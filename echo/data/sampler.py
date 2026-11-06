import numpy as np

from dataclasses import dataclass, field
from random import Random
from torch.utils.data import Sampler
from typing import Iterator, List, Literal

from echo.data.constants import OPEN_BATCHES
from echo.data.dataset import EchoDataset

# "greedy" packs utterances as they come, within max_pad_ratio; "sorted" sorts chunks of them by length first,
# which fills the token budget with far less padding (about 4% against 31% on the training data).
BatchPacking = Literal["greedy", "sorted"]


@dataclass
class _OpenBatch:
    indices: List[int] = field(default_factory=list)
    total: int = 0
    longest: int = 0

    def add(self, index: int, length: int) -> None:
        self.indices.append(index)
        self.total += length
        self.longest = max(self.longest, length)


class EchoBatchSampler(Sampler[List[int]]):
    """
    Yields the batches it composes from an :class:`EchoDataset` for the current epoch.

    With several ranks each takes every n-th batch. The composition is the same
    everywhere - it is built from a seed and the epoch - so the ranks need to
    agree on nothing but their own number.
    """

    def __init__(
        self,
        dataset: EchoDataset,
        rank: int = 0,
        world_size: int = 1,    # Number of GPUs
        max_batch_size: int = 64,
        max_batch_tokens: int = 8192,
        max_pad_ratio: float = 0.3,   # Only for "greedy" packing
        seed: int = 42,
        packing: BatchPacking = "greedy",
        packing_chunk: int = 2048,    # Utterances sorted together, only for "sorted" packing
    ) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_pad_ratio = max_pad_ratio
        self.seed = seed
        self.packing = packing
        self.packing_chunk = packing_chunk

        # A safety check for batch tokens bound correctness given loaded data.
        longest = int(dataset.tokens.max()) if len(dataset.tokens) else 0
        if longest > max_batch_tokens:
            raise ValueError(
                f"{dataset.manifest} has a {longest}-token sample, which never fits "
                f"max_batch_tokens={max_batch_tokens}"
            )

        self.epoch = 0
        self.batches = self._share(self.build_batches(self.epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.batches = self._share(self.build_batches(epoch))

    def _share(self, batches: List[List[int]]) -> List[List[int]]:
        """
        This rank's batches, and as many of them as every other rank gets.

        The remainder is dropped on purpose: a rank that runs out early leaves
        the others waiting on a gradient exchange that never comes.
        """

        usable = len(batches) - len(batches) % self.world_size

        return batches[self.rank : usable : self.world_size]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def build_batches(self, epoch: int = 0) -> List[List[int]]:
        """
        Groups the whole dataset into batches of similarly long samples.

        Performed once per training epoch.
        """

        return self._sorted_batches(epoch) if self.packing == "sorted" else self._greedy_batches(epoch)

    def _sorted_batches(self, epoch: int) -> List[List[int]]:
        """
        Random chunks of the dataset, each sorted by length and cut into batches as full as the budget allows.

        Neighbours in a sorted chunk are nearly equally long, so the batches carry almost no padding; the chunks
        and the final shuffle keep the order random from epoch to epoch.
        """

        order = np.random.default_rng(self.seed + epoch).permutation(len(self.dataset))
        batches: List[List[int]] = []

        for start in range(0, len(order), self.packing_chunk):
            chunk = sorted(order[start : start + self.packing_chunk], key=lambda i: int(self.dataset.tokens[i]))
            batch: List[int] = []
            for index in chunk:
                size = len(batch) + 1
                if batch and (size * int(self.dataset.tokens[index]) > self.max_batch_tokens or size > self.max_batch_size):
                    batches.append(batch)
                    batch = []
                batch.append(int(index))
            if batch:
                batches.append(batch)

        Random(self.seed + epoch).shuffle(batches)

        return batches

    def _greedy_batches(self, epoch: int) -> List[List[int]]:
        """
        Packs samples as they come, into a few open batches, within the padding limit.
        """

        rng = Random(self.seed + epoch)
        order = np.random.default_rng(self.seed + epoch).permutation(len(self.dataset))

        batches: List[List[int]] = []
        open_batches: List[_OpenBatch] = []

        for index in order:
            length = int(self.dataset.tokens[index])

            for batch in open_batches:
                if not self._accepts(batch, length):
                    continue

                batch.add(int(index), length)
                if self._is_full(batch):
                    batches.append(batch.indices)
                    open_batches.remove(batch)
                break
            else:
                if len(open_batches) == OPEN_BATCHES:
                    batches.append(open_batches.pop(0).indices)
                open_batches.append(_OpenBatch([int(index)], length, length))

        batches.extend(batch.indices for batch in open_batches)
        rng.shuffle(batches)

        return batches

    def _accepts(self, batch: _OpenBatch, length: int) -> bool:
        """
        A predicate, which decides whether a batch can be accepted given current limit parameters.
        """

        size = len(batch.indices) + 1
        longest = max(batch.longest, length)
        padded = size * longest

        return (
            size <= self.max_batch_size
            and padded <= self.max_batch_tokens
            and batch.total + length >= (1 - self.max_pad_ratio) * padded
        )

    def _is_full(self, batch: _OpenBatch) -> bool:
        """
        A predicate, which decides whether there is still a room in the batch,
        given current batch size limit parameters.
        """
        size = len(batch.indices)

        return (
            size >= self.max_batch_size
            or (size + 1) * batch.longest > self.max_batch_tokens
        )
