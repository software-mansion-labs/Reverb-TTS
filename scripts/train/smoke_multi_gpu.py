"""Does multi-GPU training actually work here?

Runs the real models on the real manifest under torchrun, and checks the four
things that decide it: the ranks split the data, their weights stay identical
(which is the only proof gradients are being exchanged), nobody straggles, and
the rank-zero work - each evaluation sweep, and the checkpoint - fits inside
the time a collective will wait.

    torchrun --nproc_per_node=4 scripts/train/smoke_multi_gpu.py

Every check prints PASS or FAIL, and the last line is the verdict.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from echo import build_model, freeze
from echo.data.dataset import EchoDataset
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step
from echo.training.trainer import Trainer
from echo.utils.distributed_utils import is_main, setup, teardown, world

CONFIG = "echo/configs/training/train.json"

# What a collective waits before it gives up on a rank, and takes the run with
# it. Anything one rank does alone has to fit inside this.
WATCHDOG = 600.0


def gather(value: Any) -> List[Any]:
    """The same value from every rank."""

    ranks = world()[1]
    if ranks == 1:
        return [value]

    collected: List[Any] = [None] * ranks
    dist.all_gather_object(collected, value)

    return collected


def say(check: str, passed: bool, detail: str = "") -> bool:
    if is_main():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check:<34}{detail}", flush=True)

    return passed


def check_shards(loader: EchoDataLoader) -> bool:
    """Every rank must get its own share, and the same number of them."""

    batches = loader.batch_sampler.batches
    counts = gather(len(batches))
    first = gather([tuple(batch) for batch in batches[:20]])

    seen = [batch for rank in first for batch in rank]
    equal = len(set(counts)) == 1
    disjoint = len(seen) == len(set(seen))

    ok = say("ranks step the same number of times", equal, f"batches per rank: {counts}")
    ok &= say("ranks hold different batches", disjoint,
              f"{len(set(seen))} distinct of {len(seen)} sampled")

    return ok


def check_weights(trainer: Trainer) -> bool:
    """Identical weights after a step is the proof gradients were exchanged."""

    model = getattr(trainer.student, "module", trainer.student)
    signature = float(
        sum(parameter.detach().double().sum() for parameter in model.parameters())
    )
    signatures = gather(signature)
    spread = max(signatures) - min(signatures) if len(signatures) > 1 else 0.0

    return say("weights agree across ranks", spread < 1e-3,
               f"spread {spread:.3e} over {len(signatures)} ranks")


def check_pace(times: List[float], tokens: int) -> bool:
    """A rank that is slower than the others sets the pace for all of them."""

    per_step = sorted(times)[len(times) // 2]
    paces = gather(per_step)
    slowest, fastest = max(paces), min(paces)
    ranks = len(paces)

    say("step time", True,
        f"median {per_step:.2f}s | slowest rank {slowest:.2f}s, fastest {fastest:.2f}s")
    say("throughput", True,
        f"{ranks * tokens / slowest:,.0f} tokens/s over {ranks} rank(s)")

    return say("no rank straggles", slowest < 1.3 * fastest,
               f"slowest is {slowest / max(fastest, 1e-9):.2f}x the fastest")


def check_evaluation(
    trainer: Trainer,
    loaders: Dict[str, Optional[EchoDataLoader]],
) -> bool:
    """The sweeps rank zero runs alone, while the others wait at a barrier.

    Each one is timed on its own, because the question is never whether they
    work - it is which of them outlasts the collective's patience.
    """

    ok = True

    for name, loader in loaders.items():
        if loader is None:
            say(f"{name} sweep", True, "no manifest, skipped")
            continue

        started = time.time()
        if is_main():
            trainer._evaluate(loader, name)
        waited = max(gather(time.time() - started))

        ok &= say(f"{name} sweep finishes in time", waited < WATCHDOG / 2,
                  f"{waited:.0f}s of the {WATCHDOG:.0f}s a collective allows")

    return ok


def check_checkpoint(trainer: Trainer) -> bool:
    """The write rank zero does alone - weights, and the optimizer behind them."""

    started = time.time()
    written = trainer.save() if is_main() else None
    waited = max(gather(time.time() - started))

    size = (
        sum(path.stat().st_size for path in written.glob("*") if path.is_file()) / 2**30
        if written is not None
        else 0.0
    )
    sizes = gather(size)
    elsewhere = gather(written is not None)

    ok = say("checkpoint finishes in time", waited < WATCHDOG / 2,
             f"{waited:.0f}s for {max(sizes):.1f} GiB of the {WATCHDOG:.0f}s allowed")

    return ok & say("only rank zero writes", sum(elsewhere) == 1,
                    f"{sum(elsewhere)} rank(s) wrote")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, help="Overrides the config's training manifest")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--steps", type=int, default=12, help="Steps to time, after a warmup")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/echo-multi-gpu"))
    args = parser.parse_args()

    device = setup()
    rank, ranks = world()

    config = EchoTrainingConfig.from_json(args.config)
    if args.manifest is not None:
        config.data.train_manifest = str(args.manifest)
    if args.data_root is not None:
        config.data.data_root = str(args.data_root)

    config.schedule.steps, config.schedule.epochs = 10_000, None
    config.checkpoint.output_dir = str(args.output_dir)
    config.logging.log_every_steps = 10**6      # This script does its own reporting
    config.evaluation.eval_every_steps = 1
    config.checkpoint.save_every_steps = 1

    if is_main():
        print(f"\n{ranks} rank(s), {device}, {config.data.max_batch_tokens} tokens per batch")
        print(f"manifest {config.data.train_manifest}\n")

    student = build_model(config.models.student, train=True, device=device)
    teacher = freeze(build_model(config.models.teacher, train=True, device=device, dtype=torch.float32))

    def _loader(manifest: str, evaluating: bool) -> EchoDataLoader:
        return EchoDataLoader(
            EchoDataset(manifest, **config.dataset_kwargs),
            EchoPreprocessor(
                **(config.eval_preprocessor_kwargs if evaluating else config.preprocessor_kwargs)
            ),
            disable_prefix_drop=evaluating,
            disable_masking=evaluating,
            # An evaluation reads its own data: starting workers on one
            # rank, inside a collective, is what deadlocks.
            num_workers=0 if evaluating else config.data.num_workers,
            **config.sampler_kwargs,
        )

    train = _loader(config.data.train_manifest, evaluating=False)
    dev = _loader(config.data.dev_manifest, evaluating=True) if config.data.dev_manifest else None
    test = _loader(config.data.test_manifest, evaluating=True) if config.data.test_manifest else None

    trainer = Trainer(
        config=config, student=student, teacher=teacher,
        train_loader=train, loss_fn=distillation_step,
    )

    passed = check_shards(train)

    # A warmup step, then the timed ones.
    trainer.config.evaluation.eval_every_steps = 10**6
    trainer.config.checkpoint.save_every_steps = 10**6
    trainer.train(steps=1)

    times: List[float] = []
    for _ in range(args.steps):
        started = time.time()
        trainer.train(steps=1)
        times.append(time.time() - started)

    passed &= check_weights(trainer)
    passed &= check_pace(times, config.data.max_batch_tokens)
    passed &= check_evaluation(trainer, {"dev": dev, "test": test})
    passed &= check_checkpoint(trainer)

    if is_main():
        print(f"\n{'multi-GPU training works here' if passed else 'multi-GPU training is broken here'}\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        teardown()
