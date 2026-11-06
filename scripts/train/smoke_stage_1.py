"""Smoke test for stage 1: does a run hold together, and what does it cost?

Runs a handful of steps on a small manifest, checks the gradients are real
numbers, then times a few batch settings and extrapolates one epoch from them.
"""

import argparse
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from echo import build_model, freeze
from echo.data.dataset import EchoDataset
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.data.sampler import EchoBatchSampler
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step
from echo.training.trainer import Trainer
from echo.utils.distributed_utils import is_main, setup, teardown, world

CONFIG = "echo/configs/training/train.json"
MANIFEST = "data/dev/dev_processed.csv"

# Batch settings that are actually run: (samples, tokens).
MEASURED = ((8, 2048), (16, 4096))

# And the ones too large to be worth running, carried over from those.
ESTIMATED = ((32, 8192), (64, 16384), (128, 32768))


def _config(args: argparse.Namespace, size: int, tokens: int) -> EchoTrainingConfig:
    """
    The stage's config, cut down to something that runs in a minute.
    """

    config = EchoTrainingConfig.from_json(args.config)

    config.data.train_manifest = str(args.manifest)
    config.data.data_root = str(args.data_root) if args.data_root else None
    config.data.max_batch_size = size
    config.data.max_batch_tokens = tokens
    config.data.num_workers = args.num_workers

    config.schedule.steps, config.schedule.epochs = 1000, None
    config.checkpoint.output_dir = str(args.output_dir)
    config.logging.log_every_steps = 10**6    # The smoke does its own reporting
    config.evaluation.eval_every_steps = 10**6
    config.checkpoint.save_every_steps = 10**6

    return config


def _trainer(config: EchoTrainingConfig, models: Tuple[Any, Any]) -> Trainer:
    student, teacher = models
    dataset = EchoDataset(config.data.train_manifest, **config.dataset_kwargs)

    return Trainer(
        config=config,
        student=student,
        teacher=teacher,
        train_loader=EchoDataLoader(
            dataset,
            EchoPreprocessor(**config.preprocessor_kwargs),
            num_workers=config.data.num_workers,
            **config.sampler_kwargs,
        ),
        loss_fn=distillation_step,
    )


def check_gradients(trainer: Trainer) -> Dict[str, float]:
    """
    One step's worth of backward, looked at before it is applied.

    A run can report a perfectly finite loss and still be broken underneath, so
    the gradients themselves are what gets checked.
    """

    terms = trainer._accumulate()

    gradients = [
        parameter.grad
        for parameter in trainer.student.parameters()
        if parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = torch.norm(torch.stack([gradient.norm() for gradient in gradients]))

    trainer._apply()

    if not finite:
        raise RuntimeError("some gradients are not finite")
    if not all(torch.isfinite(torch.tensor(value)) for value in terms.values()):
        raise RuntimeError(f"the loss is not a number: {terms}")

    return {**terms, "grad_norm": float(norm), "with_grad": len(gradients)}


def measure(trainer: Trainer, steps: int) -> Dict[str, float]:
    """
    Seconds per optimizer step, once the first one is out of the way.
    """

    _reset_peak()
    trainer.train(steps=1)   # Warmup: allocations, autotuning, the first epoch's batches

    started = time.time()
    trainer.train(steps=steps)
    elapsed = time.time() - started

    batches = len(trainer.train_loader)
    per_step = elapsed / steps

    return {
        "seconds_per_step": per_step,
        "batches_per_epoch": batches,
        "steps_per_epoch": batches / trainer.config.optim.gradient_accumulation_steps,
        "epoch_hours": per_step
        * batches
        / trainer.config.optim.gradient_accumulation_steps
        / 3600,
        "peak_memory_gb": _peak(),
    }


def _line_through(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Least squares line through (tokens, cost), as intercept and slope.

    Both the step time and the memory grow with the tokens in a batch, over a
    fixed cost - the weights, the optimizer state, the teacher - that a batch of
    no tokens would still pay. Two measured points are enough to separate them.
    """

    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    spread = sum((x - mean_x) ** 2 for x, _ in points)

    if spread == 0.0:
        raise ValueError("the measured setups need at least two different token budgets")

    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread

    return mean_y - slope * mean_x, slope


def count_batches(config: EchoTrainingConfig) -> int:
    """
    How many batches of an epoch this rank would see.

    Packing the manifest costs nothing next to running it, so this is counted
    even for the settings that are only estimated. The count is per rank, like
    the measured one: the ranks share an epoch out between them.
    """

    dataset = EchoDataset(config.data.train_manifest, **config.dataset_kwargs)
    batches = EchoBatchSampler(dataset, **config.sampler_kwargs).batches
    ranks = world()[1]

    return (len(batches) - len(batches) % ranks) // ranks


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak() -> float:
    """
    The high water mark of memory, in GiB.
    """

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30

    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30

    # The resident set, which on Linux is counted in kilobytes and on macOS in bytes.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return peak / 2**30 if peak > 2**32 else peak * 1024 / 2**30


def _row(kind: str, size: int, tokens: int, cost: Dict[str, float]) -> None:
    print(
        f"{kind:>10}{size:>9}{tokens:>8}{cost['batches_per_epoch']:>9.0f}"
        f"{cost['seconds_per_step']:>9.2f}{cost['epoch_hours']:>11.2f}"
        f"{cost['peak_memory_gb']:>12.2f}",
        flush=True,
    )


def _silence() -> None:
    """
    Keeps every rank but the first from writing the same table again.
    """

    global print

    print = lambda *args, **kwargs: None   # noqa: E731


def _pairs(text: str) -> List[Tuple[int, int]]:
    return [
        (int(size), int(tokens))
        for size, tokens in (pair.split(":") for pair in text.split(",") if pair)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="A small dataset to run on")
    parser.add_argument("--data-root", type=Path, help="Root the codec paths are relative to")
    parser.add_argument("--steps", type=int, default=5, help="Steps each setup is timed over")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/echo-smoke"))
    parser.add_argument(
        "--setups",
        default=",".join(f"{size}:{tokens}" for size, tokens in MEASURED),
        help="Batch settings to run, as samples:tokens",
    )
    parser.add_argument(
        "--estimate",
        default=",".join(f"{size}:{tokens}" for size, tokens in ESTIMATED),
        help="Batch settings to carry over from the measured ones, as samples:tokens",
    )
    args = parser.parse_args()

    setups = _pairs(args.setups)
    estimates = _pairs(args.estimate)

    device = setup()
    rank, ranks = world()

    # Every rank runs the same work; only one of them says so.
    if not is_main():
        _silence()

    print(f"device {device}  |  ranks {ranks}  |  manifest {args.manifest}\n", flush=True)

    student = build_model(_config(args, *setups[0]).models.student, train=True, device=device)
    teacher = freeze(
        build_model(
            _config(args, *setups[0]).models.teacher,
            train=True,
            device=device,
            dtype=torch.float32,
        )
    )

    # 1. Does a step hold together, and are its gradients real numbers?
    checked = check_gradients(_trainer(_config(args, *setups[0]), (student, teacher)))
    print(
        f"gradients: {checked['with_grad']} tensors, norm {checked['grad_norm']:.4g}, "
        f"loss {checked['loss']:.4f} (ce {checked['ce']:.4f}, kl {checked['kl']:.4f})\n",
        flush=True,
    )

    # 2. What does it cost, at the settings worth running?
    header = (
        f"{'':>10}{'samples':>9}{'tokens':>8}{'batches':>9}"
        f"{'s/step':>9}{'epoch (h)':>11}{'peak (GiB)':>12}"
    )
    print(header)
    print("-" * len(header))

    measured: List[Dict[str, float]] = []

    for size, tokens in setups:
        config = _config(args, size, tokens)

        try:
            cost = measure(_trainer(config, (student, teacher)), args.steps)
        except Exception as error:   # A budget too small for the longest sample, most likely
            print(f"{'run':>10}{size:>9}{tokens:>8}   {type(error).__name__}: {str(error)[:40]}")
            continue

        measured.append({**cost, "tokens": tokens})
        _row("run", size, tokens, cost)

    # 3. And what the larger ones would cost, carried over from those.
    if len(measured) < 2:
        print("\nnot enough measured setups to carry anything over")
        return

    time_at = _line_through([(row["tokens"], row["seconds_per_step"]) for row in measured])
    memory_at = _line_through([(row["tokens"], row["peak_memory_gb"]) for row in measured])

    print()

    for size, tokens in estimates:
        config = _config(args, size, tokens)

        try:
            batches = count_batches(config)
        except Exception as error:
            print(f"{'est':>10}{size:>9}{tokens:>8}   {type(error).__name__}: {str(error)[:40]}")
            continue

        per_step = time_at[0] + time_at[1] * tokens
        steps = batches / config.optim.gradient_accumulation_steps

        _row(
            "est",
            size,
            tokens,
            {
                "batches_per_epoch": batches,
                "seconds_per_step": per_step,
                "epoch_hours": per_step * steps / 3600,
                "peak_memory_gb": memory_at[0] + memory_at[1] * tokens,
            },
        )

    if ranks > 1:
        print(f"\nepoch times are for {ranks} ranks working at once")

    print(
        f"\ncarried over from {len(measured)} runs: "
        f"{memory_at[0]:.2f} GiB + {memory_at[1] * 1024:.3f} GiB per 1k tokens, "
        f"{time_at[0]:.2f} s + {time_at[1] * 1024:.3f} s per 1k tokens"
    )


if __name__ == "__main__":
    if sys.platform == "darwin":
        torch.multiprocessing.set_start_method("spawn", force=True)

    try:
        main()
    finally:
        teardown()
