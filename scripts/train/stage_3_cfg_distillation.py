"""Stage 3 - CFG distillation: the student learns, in one conditional pass, the guided mixture a frozen copy of itself makes."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

from echo import EchoConfig, build_model, freeze
from echo.training.config import EchoTrainingConfig
from echo.training.step import cfg_step
from echo.training.tracking import MLflowTracker
from echo.training.trainer import Trainer
from echo.training.types import Student
from echo.model.model import EchoModel
from echo.utils.config_utils import check
from echo.utils.distributed_utils import is_main, setup, teardown
from stage_1_prefix_blocking import build_loader, start_logging

CONFIG = "echo/configs/training/train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Training stage's .json config")
    parser.add_argument("--init", type=Path, required=True, help="Checkpoint both the student and its frozen guided copy start from")
    parser.add_argument("--output-dir", type=Path, help="Overrides the config's checkpoint.output_dir")
    parser.add_argument("--epochs", type=int, help="Overrides the config's length, in epochs")
    parser.add_argument("--steps", type=int, help="Overrides the config's length, in steps")
    parser.add_argument("--eval-at-end", action="store_true", help="Also evaluates after the last step")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    check(config.guiding, "a CFG distillation stage needs guidance.scale > 0")
    if args.output_dir is not None:
        config.checkpoint.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.schedule.steps, config.schedule.epochs = None, args.epochs
    if args.steps is not None:
        config.schedule.steps, config.schedule.epochs = args.steps, None

    device = setup()
    start_logging(config.checkpoint.output_dir)

    # The student holds the guidance in its weights once trained, so it is saved to generate unguided.
    student_config = EchoConfig.from_json(config.models.student)
    student_config.init_from = str(args.init)
    student_config.cfg.mode = "student"
    student = build_model(student_config, train=True, device=device)

    # The teacher is the same checkpoint, frozen: cfg_step asks it for both branches and mixes them.
    teacher_config = EchoConfig.from_json(config.models.teacher)
    teacher_config.init_from = str(args.init)
    teacher = freeze(build_model(teacher_config, train=True, device=device, dtype=torch.float32))

    tracker = MLflowTracker(config) if config.tracking.enabled and is_main() else None

    trainer = None

    def annealed_step(
        student: Student,
        teacher: EchoModel,
        inputs: Dict[str, torch.Tensor],
        config: EchoTrainingConfig,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # The guidance the mixture is taken at rises along the run, as 'guidance' prescribes.
        teacher.echo_config.cfg.guidance_scale = config.guidance.at(trainer.step, trainer.total_steps)
        return cfg_step(student, teacher, inputs, config, weights)

    trainer = Trainer(
        config=config,
        student=student,
        teacher=teacher,
        train_loader=build_loader(config.data.train_manifest, config),
        dev_loader=build_loader(config.data.dev_manifest, config, evaluating=True) if config.evaluation.eval_on_dev else None,
        test_loader=build_loader(config.data.test_manifest, config, evaluating=True) if config.evaluation.eval_on_test else None,
        loss_fn=annealed_step,
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
