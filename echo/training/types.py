import torch

from torch.nn.parallel import DistributedDataParallel
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

from echo.model.model import EchoModel

# The config imports the aliases below, so importing it back at runtime would
# be circular - the step signature names it for type checkers alone.
if TYPE_CHECKING:
    from echo.training.config import EchoTrainingConfig

MixedPrecision = Literal["fp32", "bf16", "fp16"]
LRSchedulerType = Literal["cosine"]
AnnealShape = Literal["constant", "linear", "power"]

Student = Union[EchoModel, DistributedDataParallel]

# What one training step is given, and what it gives back: the total under
# 'loss', and whatever else it wants reported beside it. A stage brings its own
# - see echo/training/step.py - and the trainer knows nothing about which.
StepFn = Callable[
    [
        Student,
        Optional[EchoModel],
        Dict[str, torch.Tensor],
        "EchoTrainingConfig",
        torch.Tensor,
    ],
    Dict[str, torch.Tensor],
]
