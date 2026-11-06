import torch

from dataclasses import dataclass
from typing import Dict, List, Sequence

from echo.model.model import EchoModel
from echo.pruning.calibration import CalibrationInput, branch_kl
from echo.pruning.surgery import faded_layer, shrunk_ffn, without_layers

# Damage is measured as KL(teacher || model), and decided on the conditional branch alone. The unconditional one is
# reported beside it, and the CFG mixture not at all.
Damage = Dict[str, float]   # KL, or its change, per branch: 'cond', 'uncond'


@dataclass
class LayerChoice:
    layer: int        # Index in the current model
    step: Damage      # ΔKL of this removal, against the current model
    total: Damage     # KL to the teacher after it


def ffn_curve(
    model: EchoModel,
    teacher: EchoModel,
    inputs: List[CalibrationInput],
    grams: List[torch.Tensor],
    sizes: Sequence[int],
) -> Dict[int, Damage]:
    """
    ΔKL of shrinking every layer's FFN to each of 'sizes', one size at a time.
    """

    base = branch_kl(model, teacher, inputs)
    curve = {}

    for size in sizes:
        with shrunk_ffn(model, size, grams):
            curve[size] = _delta(branch_kl(model, teacher, inputs), base)

    return curve


def layer_costs(
    model: EchoModel,
    teacher: EchoModel,
    inputs: List[CalibrationInput],
    protected: Sequence[int] = (0, -1),   # Never candidates - negative indices count from the end
) -> Dict[int, Damage]:
    """
    ΔKL to the teacher of removing each unprotected layer on its own.
    """

    depth = len(model.llm.layers)
    fixed = {index % depth for index in protected}
    base = branch_kl(model, teacher, inputs)
    costs = {}

    for layer in range(depth):
        if layer not in fixed:
            with without_layers(model, [layer]):
                costs[layer] = _delta(branch_kl(model, teacher, inputs), base)

    return costs


def fade_curve(
    model: EchoModel,
    teacher: EchoModel,
    inputs: List[CalibrationInput],
    layer: int,
    alphas: Sequence[float],   # Scales of the layer's outputs to measure, 0 being its removal
    base: Damage,              # KL of the unchanged model to the teacher
) -> Dict[float, Damage]:
    """
    ΔKL to the teacher of scaling one layer's outputs by each of 'alphas': where a fade towards removal does its damage.
    """

    curve = {}

    for alpha in alphas:
        with faded_layer(model, layer, alpha):
            curve[alpha] = _delta(branch_kl(model, teacher, inputs), base)

    return curve


def next_layer(
    costs: Dict[int, Damage],
    distance: Damage,   # KL of the current model to the teacher
) -> LayerChoice:
    """
    The layer whose removal moves the conditional branch least from the teacher, with what it costs.
    """

    layer = min(costs, key=lambda index: costs[index]["cond"])
    step = costs[layer]
    total = {key: value + distance[key] for key, value in step.items()}

    return LayerChoice(layer=layer, step=step, total=total)


def _delta(after: Damage, before: Damage) -> Damage:
    return {key: after[key] - before[key] for key in after}
