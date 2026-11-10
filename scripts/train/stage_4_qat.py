"""Stage 4 - quantization-aware training: the student computes with quantized weights (and int8 inputs, W8A8 by default) in its decoder linears, and learns the distribution of its full-precision self."""

import argparse
import logging
import sys
from pathlib import Path

import torch

from echo import EchoConfig, build_model, freeze
from echo.model.quantization import quantize
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step
from echo.training.tracking import MLflowTracker
from echo.training.trainer import Trainer
from echo.utils.config_utils import check
from echo.utils.distributed_utils import is_main, setup, teardown
from stage_1_prefix_blocking import build_loader, start_logging

CONFIG = "echo/configs/training/qat_w8a8.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Training stage's .json config")
    parser.add_argument("--init", type=Path, help="Checkpoint both the student and its frozen full-precision copy start from - the model configs' init_from if unset")
    parser.add_argument("--output-dir", type=Path, help="Overrides the config's checkpoint.output_dir")
    parser.add_argument("--epochs", type=int, help="Overrides the config's length, in epochs")
    parser.add_argument("--steps", type=int, help="Overrides the config's length, in steps")
    parser.add_argument("--eval-at-end", action="store_true", help="Also evaluates after the last step")
    parser.add_argument("--bits", type=int, default=8, help="Width of the quantized weights")
    parser.add_argument("--group-size", type=int, help="Inputs sharing one weight scale - one scale per output channel if unset")
    parser.add_argument("--activations", action=argparse.BooleanOptionalAction, default=True, help="Also quantizes the inputs of the linears to int8")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    check(config.distilling, "a QAT stage distils from the full-precision model - it needs a teacher and a KL weight")
    if args.output_dir is not None:
        config.checkpoint.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.schedule.steps, config.schedule.epochs = None, args.epochs
    if args.steps is not None:
        config.schedule.steps, config.schedule.epochs = args.steps, None

    device = setup()
    start_logging(config.checkpoint.output_dir)

    student_config = EchoConfig.from_json(config.models.student)
    if args.init is not None:
        student_config.init_from = str(args.init)
    student = build_model(student_config, train=True, device=device)

    # Before the trainer wraps the student, so the gradient exchange and the checkpointing see the quantized linears.
    # The checkpoints hold the full-precision weights - scripts/quantize.py --activations turns one into int8.
    count = quantize(student, args.activations, args.bits, args.group_size)
    logging.info("Quantized %d linears to W%dA%d, weight group size %s", count, args.bits, 8 if args.activations else 16, args.group_size)

    teacher_config = EchoConfig.from_json(config.models.teacher)
    if args.init is not None:
        teacher_config.init_from = str(args.init)
    teacher = freeze(build_model(teacher_config, train=True, device=device, dtype=torch.float32))

    tracker = MLflowTracker(config) if config.tracking.enabled and is_main() else None

    trainer = Trainer(
        config=config,
        student=student,
        teacher=teacher,
        train_loader=build_loader(config.data.train_manifest, config),
        dev_loader=build_loader(config.data.dev_manifest, config, evaluating=True) if config.evaluation.eval_on_dev else None,
        test_loader=build_loader(config.data.test_manifest, config, evaluating=True) if config.evaluation.eval_on_test else None,
        loss_fn=distillation_step,
        tracker=tracker,
    )

    status = "FAILED"

    try:
        trainer.train()

        # Unless the last step fell on the evaluation cadence and was evaluated already.
        if args.eval_at_end and trainer.step % config.evaluation.eval_every_steps != 0:
            trainer._evaluate_all()

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
