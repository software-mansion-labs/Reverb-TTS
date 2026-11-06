"""Stage 1 - distils a prefix-blocked student from the full-attention teacher."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from echo import build_model, freeze
from echo.data.dataset import EchoDataset
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step
from echo.training.tracking import MLflowTracker
from echo.training.trainer import Trainer
from echo.utils.distributed_utils import is_main, setup, teardown, world

CONFIG = "echo/configs/training/train.json"

# Kept beside the checkpoints, so a run's weights and its account of itself
# travel together.
LOG_FILE = "train.log"


def start_logging(output_dir: str) -> None:
    """
    Sends the log to the screen and to a file in the run's directory.

    Every rank writes its own, appending rather than truncating: a resumed run
    continues the same account, and a rank that dies leaves one behind - which
    is the only place its last words are ever recorded.
    """

    rank = world()[0]
    path = Path(output_dir) / (LOG_FILE if rank == 0 else f"train.rank{rank}.log")
    path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if is_main() else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
    )


def build_loader(
    manifest: Optional[str],
    config: EchoTrainingConfig,
    evaluating: bool = False,   # An evaluation set: a pinned split, and nothing masked or dropped
) -> Optional[EchoDataLoader]:
    """
    The loader for one of the manifests, or nothing when there is no manifest.
    """

    if manifest is None:
        return None

    dataset = EchoDataset(manifest, **config.dataset_kwargs)
    preprocessor = EchoPreprocessor(
        **(config.eval_preprocessor_kwargs if evaluating else config.preprocessor_kwargs)
    )

    return EchoDataLoader(
        dataset,
        preprocessor,
        disable_prefix_drop=evaluating,
        disable_masking=evaluating,
        # Training is shared out between the ranks; an evaluation is run by one
        # of them, and has to see the whole set to stay comparable.
        sharded=not evaluating,
        # No workers for an evaluation: it runs on one rank while the others
        # wait in a collective, and starting worker processes there means
        # forking a process that already holds CUDA and NCCL state. The sets
        # are small enough to read on the spot.
        num_workers=0 if evaluating else config.data.num_workers,
        **config.sampler_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Training stage's .json config")
    parser.add_argument("--resume", type=Path, help="Checkpoint to carry on from")
    parser.add_argument("--steps", type=int, help="Overrides the config's length, for a short run")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    if args.resume is not None:
        config.checkpoint.resume = str(args.resume)
    if args.steps is not None:
        config.schedule.steps, config.schedule.epochs = args.steps, None

    device = setup()
    start_logging(config.checkpoint.output_dir)

    # The student is trained in full precision and cast where it helps; the
    # teacher stays in fp32 too - in bf16 its fine codebooks are measurably
    # worse (echo.training.step._full_precision).
    student = build_model(config.models.student, train=True, device=device)
    teacher = freeze(
        build_model(
            config.models.teacher, train=True, device=device, dtype=torch.float32
        )
    )

    tracker = MLflowTracker(config) if config.tracking.enabled and is_main() else None

    trainer = Trainer(
        config=config,
        student=student,
        teacher=teacher,
        train_loader=build_loader(config.data.train_manifest, config),
        dev_loader=(
            build_loader(config.data.dev_manifest, config, evaluating=True)
            if config.evaluation.eval_on_dev
            else None
        ),
        test_loader=(
            build_loader(config.data.test_manifest, config, evaluating=True)
            if config.evaluation.eval_on_test
            else None
        ),
        loss_fn=distillation_step,
        tracker=tracker,
    )

    status = "FAILED"

    try:
        trainer.train()

        if is_main():
            logging.info("Finished at step %d, writing the last checkpoint", trainer.step)
            trainer.save()

        status = "FINISHED"
    finally:
        if tracker is not None:
            tracker.close(status)

        teardown()


if __name__ == "__main__":
    # macOS has no fork worth using; Linux does, and spawning would make every
    # worker re-import torch and unpickle the manifest again.
    if sys.platform == "darwin":
        torch.multiprocessing.set_start_method("spawn", force=True)

    main()
