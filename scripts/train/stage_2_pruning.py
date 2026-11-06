"""Stage 2 - retrains a pruned student: plain CE, or CE with a KL term when the config distils."""

import argparse
import logging
import sys
from pathlib import Path

import torch

from echo import EchoConfig, build_model, freeze
from echo.pruning.surgery import LayerFade, drop_layers
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step, plain_step
from echo.training.tracking import MLflowTracker
from echo.training.trainer import Trainer
from echo.utils.distributed_utils import is_main, setup, teardown
from stage_1_prefix_blocking import build_loader, start_logging

CONFIG = "echo/configs/training/train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Training stage's .json config")
    parser.add_argument("--init", type=Path, required=True, help="Checkpoint the student starts from, e.g. one written by scripts/pruning/prune.py")
    parser.add_argument("--output-dir", type=Path, help="Overrides the config's checkpoint.output_dir")
    parser.add_argument("--epochs", type=int, help="Overrides the config's length, in epochs")
    parser.add_argument("--steps", type=int, help="Overrides the config's length, in steps")
    parser.add_argument("--eval-at-end", action="store_true", help="Also evaluates after the last step")
    parser.add_argument("--fade-layer", type=int, help="Fades this layer out over pruning.fade_steps, and drops it before the last save")
    parser.add_argument("--warmup-start-ratio", type=float, help="Overrides the config's schedule.warmup_start_ratio")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    if args.output_dir is not None:
        config.checkpoint.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.schedule.steps, config.schedule.epochs = None, args.epochs
    if args.steps is not None:
        config.schedule.steps, config.schedule.epochs = args.steps, None
    if args.warmup_start_ratio is not None:
        config.schedule.warmup_start_ratio = args.warmup_start_ratio

    device = setup()
    start_logging(config.checkpoint.output_dir)

    student_config = EchoConfig.from_json(config.models.student)
    student_config.init_from = str(args.init)
    student = build_model(student_config, train=True, device=device)

    # Attached before the trainer wraps the student, so the frozen layer never enters the gradient exchange.
    pruning = config.pruning
    if args.fade_layer is not None and pruning.fade_steps == 0:
        parser.error("--fade-layer needs pruning.fade_steps in the config")
    fade = LayerFade(student, args.fade_layer, pruning.fade_steps, pruning.fade_power, pruning.fade_budget) if args.fade_layer is not None else None

    # A CE-only run never builds the teacher, unless it selects the tokens.
    teacher = (
        freeze(build_model(config.models.teacher, train=True, device=device, dtype=torch.float32))
        if config.distilling or config.selecting
        else None
    )

    # Without a teacher, evaluations compare against the phase's reference - rank zero's alone, as only it evaluates.
    reference = (
        freeze(build_model(config.pruning.reference, train=True, device=device))
        if teacher is None and config.pruning.reference is not None and is_main()
        else None
    )

    tracker = MLflowTracker(config) if config.tracking.enabled and is_main() else None

    trainer = Trainer(
        config=config,
        student=student,
        teacher=teacher,
        reference=reference,
        train_loader=build_loader(config.data.train_manifest, config),
        dev_loader=build_loader(config.data.dev_manifest, config, evaluating=True) if config.evaluation.eval_on_dev else None,
        test_loader=build_loader(config.data.test_manifest, config, evaluating=True) if config.evaluation.eval_on_test else None,
        loss_fn=distillation_step if config.distilling or config.selecting else plain_step,
        tracker=tracker,
        fade=fade,
    )

    status = "FAILED"

    try:
        trainer.train()

        # Unless the last step fell on the evaluation cadence and was evaluated already.
        if args.eval_at_end and trainer.step % config.evaluation.eval_every_steps != 0:
            trainer._evaluate_all()

        # A layer faded all the way passes its input through, so dropping it leaves the model as it is.
        if fade is not None:
            if not fade.done:
                raise SystemExit(f"layer {fade.index} only faded to alpha {fade.alpha:.3f} - lengthen the run or raise pruning.fade_budget")
            drop_layers(student, [fade.index])

        if is_main():
            logging.info("Finished at step %d, writing the last checkpoint", trainer.step)
            trainer.save()

        status = "FINISHED"
    finally:
        if tracker is not None:
            tracker.close(status)

        teardown()


if __name__ == "__main__":
    if sys.platform == "darwin":
        torch.multiprocessing.set_start_method("spawn", force=True)

    main()
