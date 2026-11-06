from typing import List

import numpy as np


class StringColumn:
    """
    A column of strings kept as one blob plus offsets.

    Holding 1M rows as Python strings costs hundreds of megabytes that every
    dataloader worker ends up copying; two numpy arrays do not.
    """

    def __init__(self, values: List[str]) -> None:
        encoded = [value.encode() for value in values]
        self.blob = b"".join(encoded)
        self.offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum([len(item) for item in encoded])

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> str:
        return self.blob[self.offsets[index] : self.offsets[index + 1]].decode()
