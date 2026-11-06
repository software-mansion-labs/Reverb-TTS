"""Prints what removing each layer would cost, and which layer the next pruning step would take. Writes nothing.

With --fade-top, also how the cheapest layers' damage builds up as their outputs are scaled from 1 towards 0.
"""

import argparse
from pathlib import Path
from typing import Dict, List

from echo.pruning.calibration import branch_kl, calibration_inputs, calibration_loader, load_student, load_teacher
from echo.pruning.search import Damage, LayerChoice, fade_curve, layer_costs, next_layer
from echo.training.config import EchoTrainingConfig
from echo.utils.distributed_utils import setup

CONFIG = "echo/configs/training/train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Pruning stage's .json config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The model to inspect")
    parser.add_argument("--fade-top", type=int, default=0, help="Measures the fade curve of this many cheapest layers")
    parser.add_argument("--alphas", default="0.75,0.5,0.25", help="Output scales the fade curve is measured at")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    device = setup()
    model = load_student(config, args.checkpoint, device)
    teacher = load_teacher(config, device)
    batches = config.pruning.calibration_batches
    inputs = calibration_inputs(calibration_loader(config), batches, device, skip=batches)

    distance = branch_kl(model, teacher, inputs)
    costs = layer_costs(model, teacher, inputs, config.pruning.protected_layers)
    choice = next_layer(costs, distance)

    kept = model.echo_config.architecture.kept_layers or list(range(len(model.llm.layers)))
    print(f"\n{len(model.llm.layers)} layers, {len(inputs)} scoring inputs")
    print(f"KL to the teacher now: cond {distance['cond']:.4f}  uncond {distance['uncond']:.4f}\n")
    print(f"{'layer':>6}{'orig':>6}{'dKL cond':>11}{'dKL uncond':>12}")
    for layer, damage in sorted(costs.items(), key=lambda item: item[1]["cond"]):
        print(f"{layer:>6}{kept[layer]:>6}{damage['cond']:>+11.4f}{damage['uncond']:>+12.4f}")

    print_choice(choice, kept)

    alphas = [float(alpha) for alpha in args.alphas.split(",")]
    for layer in sorted(costs, key=lambda index: costs[index]["cond"])[: args.fade_top]:
        curve = {**fade_curve(model, teacher, inputs, layer, alphas, distance), 0.0: costs[layer]}
        print_curve(layer, kept[layer], curve)


def print_curve(layer: int, original: int, curve: Dict[float, Damage]) -> None:
    cut = curve[0.0]["cond"]
    print(f"fade of layer {layer} (originally {original})")
    print(f"{'alpha':>8}{'dKL cond':>11}{'dKL uncond':>12}{'of the cut':>12}")
    for alpha, damage in sorted(curve.items(), reverse=True):
        print(f"{alpha:>8.2f}{damage['cond']:>+11.4f}{damage['uncond']:>+12.4f}{damage['cond'] / cut:>12.0%}")
    print()


def print_choice(choice: LayerChoice, kept: List[int]) -> None:
    print(f"\nnext layer: {choice.layer} (originally {kept[choice.layer]})")
    print(f"  step   dKL cond {choice.step['cond']:+.4f}")
    print(f"  total  KL cond  {choice.total['cond']:.4f}   (against the teacher)\n")


if __name__ == "__main__":
    main()
