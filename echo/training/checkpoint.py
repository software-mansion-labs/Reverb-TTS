import logging
import numpy as np
import random
import shutil
import torch

from pathlib import Path
from safetensors.torch import load_model
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.utils import SAFE_WEIGHTS_NAME
from typing import Any, Dict, List, Optional, Union

from echo.model.model import EchoModel
from omnivoice.models.omnivoice import _resolve_model_path

logger = logging.getLogger(__name__)

# Everything a run needs to carry on, beside the weights themselves.
STATE_FILE = "state.pt"

# Files that make a checkpoint loadable for inference. The audio codec is left
# out on purpose: it is 768 MB, it is the same for every checkpoint, and the
# model falls back to its published copy when a checkpoint carries none.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def save_checkpoint(
    output_dir: Union[str, Path],   # The run's directory - checkpoints are written inside it
    model: EchoModel,
    optimizer: Optimizer,
    schedule: LRScheduler,
    step: int,
    epoch: int = 0,
    keep_last_n: int = 3,           # Older checkpoints are removed, 0 keeps them all
    tokenizer_from: Optional[Union[str, Path]] = None,   # Model directory or Hub id the text tokenizer is copied from
) -> Path:
    """
    Writes everything the run would need to carry on from here.
    """

    path = Path(output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
            "step": step,
            "epoch": epoch,
            "rng": _rng_state(),
        },
        path / STATE_FILE,
    )

    if tokenizer_from is not None:
        _copy_tokenizer(Path(_resolve_model_path(str(tokenizer_from))), path)

    _rotate(Path(output_dir), keep_last_n)

    return path


def load_checkpoint(
    path: Union[str, Path],
    model: EchoModel,
    optimizer: Optional[Optimizer] = None,
    schedule: Optional[LRScheduler] = None,
) -> Dict[str, int]:
    """
    Restores a run's state, and says where it left off.
    """

    # One file: save_pretrained only shards a model past 50 GB.
    load_model(model, Path(path) / SAFE_WEIGHTS_NAME)

    state = torch.load(Path(path) / STATE_FILE, map_location="cpu", weights_only=False)

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

    if schedule is not None:
        schedule.load_state_dict(state["schedule"])

        # Torch writes a loaded rate onto the optimizer at the next step, not
        # now, so a resume would otherwise spend one step at the rate the run
        # started from.
        if optimizer is not None:
            for group, rate in zip(optimizer.param_groups, schedule.get_last_lr()):
                group["lr"] = rate

    _restore_rng(state["rng"])

    return {"step": state["step"], "epoch": state["epoch"]}


def _rng_state() -> Dict[str, Any]:
    """
    The streams a training step draws from.
    """

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _copy_tokenizer(source: Path, destination: Path) -> None:
    """
    Puts the text tokenizer beside the weights, so the checkpoint can be
    evaluated without being told where it came from.
    """

    for name in TOKENIZER_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)


def _rotate(output_dir: Path, keep_last_n: int) -> None:
    """
    Removes the oldest checkpoints, keeping the last few.
    """

    if keep_last_n <= 0:
        return

    checkpoints: List[Path] = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )

    for path in checkpoints[:-keep_last_n]:
        logger.info("Removing %s", path)
        shutil.rmtree(path)
