import math
import torch

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Dict, List

from echo.training.config import EchoTrainingConfig
from echo.training.constants import BETAS

# Share of the run the ramp from zero up to a raised warmup start takes.
RAMP_RATIO = 0.01


# Training helper
def build_optimizer(model: torch.nn.Module, config: EchoTrainingConfig) -> Optimizer:
    """
    AdamW over the parameters that are still trainable.

    Weight decay is applied to the matrices alone.
    """

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups: List[Dict[str, Any]] = [
        {
            "params": [parameter for parameter in trainable if parameter.ndim > 1],
            "weight_decay": config.optim.weight_decay,
        },
        {
            "params": [parameter for parameter in trainable if parameter.ndim <= 1],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(groups, lr=config.optim.learning_rate, betas=BETAS)


# Training helper
def build_schedule(
    optimizer: Optimizer,
    config: EchoTrainingConfig,
    total_steps: int,   # The run's length, which the decay is spread over
) -> LambdaLR:
    """
    Linear warmup, the peak held for 'schedule.hold_ratio' of the run, then a cosine decay down to a share of it.

    With 'schedule.warmup_start_ratio' the warmup climbs from that share of the peak instead of from zero - for a run
    that carries on from a previous one's rate. A short ramp still leads up to it: the optimizer state is fresh,
    and its first steps move every weight by about the full rate.

    The rate never reaches zero: the last steps of a distillation run are still
    worth something, and a floor keeps them so.
    """

    warmup = config.warmup_steps(total_steps)
    decay_start = max(warmup, int(config.schedule.hold_ratio * total_steps))
    floor = config.schedule.min_lr_ratio
    start = config.schedule.warmup_start_ratio
    ramp = min(max(int(RAMP_RATIO * total_steps), 1), warmup) if start > 0.0 else 0

    def rate(step: int) -> float:
        # Counted from one, so the first step is not taken at a rate of zero.
        if step < ramp:
            return start * (step + 1) / ramp
        if step < warmup:
            return start + (1.0 - start) * (step + 1 - ramp) / (warmup - ramp)

        progress = max(step - decay_start, 0) / max(total_steps - decay_start, 1)

        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, rate)
