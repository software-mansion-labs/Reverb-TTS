import torch
import torch.nn.functional as F

from functools import partial
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional

from echo.data.sampler import BatchPacking, EchoBatchSampler
from echo.data.dataset import EchoDataset
from echo.data.constants import AUDIO_MASK_ID
from echo.data.preprocessor import EchoPreprocessor
from echo.utils.distributed_utils import world
from echo.utils.seed_utils import seed_worker


def collate(
    samples: List[Dict[str, Any]],
    preprocessor: Optional[EchoPreprocessor] = None,
    disable_prefix_drop: bool = False,   # Skips the unconditional-branch samples
    disable_masking: bool = False,       # Leaves the target audio intact
) -> Dict[str, List[Any]]:
    """
    Preprocesses the samples if asked to, then groups them column-wise and pads
    every sequence to the longest one in the batch.
    """

    if preprocessor is not None:
        samples = preprocessor(samples, disable_prefix_drop, disable_masking)

    # Without a preprocessor there is no prompt yet, so the audio is the whole sequence.
    lengths = [
        sample.get("length", sample["audio_tokens"].size(-1)) for sample in samples
    ]
    longest = max(lengths)

    batch = {key: [sample[key] for sample in samples] for key in samples[0]}

    # We apply the padding specifically to the audio sequence.
    batch["audio_tokens"] = [
        F.pad(audio, (0, longest - length), value=AUDIO_MASK_ID)
        for audio, length in zip(batch["audio_tokens"], lengths)
    ]

    # The mask covers the same frames, and never the padded tail.
    # We pad it for convenience, since it will interact with the padded audio tensor.
    if "mask" in batch:
        batch["mask"] = [
            F.pad(mask, (0, longest - length), value=False)
            for mask, length in zip(batch["mask"], lengths)
        ]

    # Crates validity - marking which tokens are real, so the padded tail can be left out of attention.
    positions = torch.arange(longest).unsqueeze(0)
    batch["valid"] = positions < torch.tensor(lengths).unsqueeze(1)

    return batch


class EchoDataLoader(DataLoader):
    """
    Loads :class:`EchoDataset` batches, reading the codecs in worker processes.
    """

    def __init__(
        self,
        dataset: EchoDataset,
        preprocessor: Optional[EchoPreprocessor] = None,        # Preprocessing operator (tokenization, split, masking)
        disable_prefix_drop: bool = False,   # Skips the unconditional-branch samples
        disable_masking: bool = False,       # Leaves the target audio intact
        sharded: bool = True,                # Splits the batches between ranks - an evaluation wants the whole set
        max_batch_size: int = 64,
        max_batch_tokens: int = 8192,
        max_pad_ratio: float = 0.3,
        seed: int = 42,
        packing: BatchPacking = "greedy",
        packing_chunk: int = 2048,
        num_workers: int = 4,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        timeout: float = 300.0,              # Seconds a batch may take before the loader gives up
        **kwargs: Any,
    ) -> None:
        rank, world_size = world() if sharded else (0, 1)

        super().__init__(
            dataset,
            batch_sampler=EchoBatchSampler(
                dataset,
                rank,
                world_size,
                max_batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
                max_pad_ratio=max_pad_ratio,
                seed=seed,
                packing=packing,
                packing_chunk=packing_chunk,
            ),
            collate_fn=partial(
                collate,
                preprocessor=preprocessor,
                disable_prefix_drop=disable_prefix_drop,
                disable_masking=disable_masking,
            ),
            num_workers=num_workers,
            # Torch seeds itself in every worker, but the preprocessor also draws
            # from 'random', which it leaves alone.
            worker_init_fn=seed_worker if num_workers else None,
            persistent_workers=persistent_workers if num_workers else False,
            prefetch_factor=prefetch_factor if num_workers else None,
            # A worker that wedges - on shared memory it cannot get, on a read
            # that never returns - would otherwise hold this rank forever, and
            # the other ranks wait in a collective until their watchdog takes
            # the run down ten minutes later, naming the wrong culprit.
            timeout=timeout if num_workers else 0,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Provides RNG variety by explicitely changing the epoch id, which is used as a seed
        for RNG within the EchoBatchSampler class.
        """
        
        self.batch_sampler.set_epoch(epoch)
