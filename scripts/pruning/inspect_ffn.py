"""Prints what shrinking every layer's FFN to each size would cost, in ΔKL to the teacher per CFG branch. Writes nothing."""

import argparse
from pathlib import Path

from echo.pruning.calibration import calibration_inputs, calibration_loader, ffn_grams, load_student, load_teacher
from echo.pruning.search import ffn_curve
from echo.training.config import EchoTrainingConfig
from echo.utils.distributed_utils import setup

CONFIG = "echo/configs/training/train.json"
SIZES = "2816,2560,2304,2048,1792,1536"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Pruning stage's .json config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The model to inspect")
    parser.add_argument("--sizes", default=SIZES, help="FFN sizes to try, comma separated")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    device = setup()
    model = load_student(config, args.checkpoint, device)

    # The refit is fitted on one set and scored on another, or it would be graded on its own fit.
    loader, batches = calibration_loader(config), config.pruning.calibration_batches
    fitting = calibration_inputs(loader, batches, device)
    inputs = calibration_inputs(loader, batches, device, skip=batches)
    teacher = load_teacher(config, device)
    curve = ffn_curve(model, teacher, inputs, ffn_grams(model, fitting), [int(size) for size in args.sizes.split(",")])

    current = model.llm.config.intermediate_size
    layer_ffn = 3 * model.llm.config.hidden_size * len(model.llm.layers)
    print(f"\nFFN {current} per layer, {len(inputs)} scoring inputs\n")
    print(f"{'size':>6}{'removed':>9}{'params':>10}{'dKL cond':>11}{'dKL uncond':>12}")
    for size, damage in curve.items():
        print(
            f"{size:>6}{1 - size / current:>9.0%}{-(current - size) * layer_ffn / 1e6:>9.1f}M"
            f"{damage['cond']:>+11.4f}{damage['uncond']:>+12.4f}"
        )


if __name__ == "__main__":
    main()
