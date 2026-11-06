import torch

from dataclasses import dataclass
from torch import nn
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from echo.data.constants import IGNORE_INDEX
from echo.data.dataset import EchoDataset
from echo.data.inputs import build_model_inputs, build_unconditional_inputs
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.eval.constants import EVAL_SEED, POLISH_STATE, TRAJECTORY
from echo.eval.single_step import _draw_masks
from echo.masking.audio_masking import codebook_weights
from echo.masking.span_masking import gather_target
from echo.model.config import EchoConfig
from echo.model.factory import build_model
from echo.model.helpers import freeze
from echo.model.model import EchoModel
from echo.training.config import EchoTrainingConfig
from echo.utils.loss_utils import weighted_kl

# Sampler states every calibration batch is masked at - fully masked, the last main pass, the polish pass: the
# per-codebook masking rates, and whether it is the polish state.
CALIBRATION_STATES: Dict[str, Tuple[Sequence[float], bool]] = {
    "s8_f0": (TRAJECTORY["s8_f0"][0], False),
    "s8_f7": (TRAJECTORY["s8_f7"][0], False),
    "s8_polish": (POLISH_STATE, True),
}


@dataclass
class CalibrationInput:
    """
    One masked batch, as both CFG branches see it.
    """

    cond: Dict[str, torch.Tensor]     # EchoModel.forward inputs, labels included
    uncond: Dict[str, torch.Tensor]   # The same targets alone, with neither text nor prefix
    labels: torch.Tensor              # [B, C, T] labels over the target span, shared by both branches
    cond_valid: torch.Tensor          # [B, L] real tokens of the conditional pass
    uncond_valid: torch.Tensor        # [B, T] real tokens of the unconditional pass


def load_student(
    config: EchoTrainingConfig,
    checkpoint: Union[str, Path],   # Weights to load - the stage's student config supplies everything else
    device: torch.device,
) -> EchoModel:
    """
    The stage's student, loaded from a given checkpoint and set up for measuring.
    """

    student = EchoConfig.from_json(config.models.student)
    student.init_from = str(checkpoint)
    model = build_model(student, train=True, device=device)

    return model.eval().requires_grad_(False)


def load_teacher(config: EchoTrainingConfig, device: torch.device) -> EchoModel:
    """
    The stage's teacher, the model every pruning decision is measured against.
    """

    if config.models.teacher is None:
        raise ValueError("pruning measures the damage against the teacher, so models.teacher has to be set")

    return freeze(build_model(config.models.teacher, train=True, device=device))


def calibration_loader(config: EchoTrainingConfig) -> EchoDataLoader:
    """
    The dev set, split and batched the way evaluations see it: a pinned prefix, nothing masked.
    """

    return EchoDataLoader(
        EchoDataset(config.data.dev_manifest, **config.dataset_kwargs),
        EchoPreprocessor(**config.eval_preprocessor_kwargs),
        disable_prefix_drop=True,
        disable_masking=True,
        num_workers=0,
        **config.sampler_kwargs,
    )


def calibration_inputs(
    loader: EchoDataLoader,
    batches: int,                 # How many batches of the loader to take
    device: torch.device,
    skip: int = 0,                # Batches passed over first - keeps a scoring set apart from a fitting one
    states: Dict[str, Tuple[Sequence[float], bool]] = CALIBRATION_STATES,
) -> List[CalibrationInput]:
    """
    'batches' batches of the loader, each masked at every state in 'states', with seeded masks.
    """

    inputs = []

    for index, batch in enumerate(loader):
        if index < skip:
            continue
        if index >= skip + batches:
            break

        for state_index, (rates, polish) in enumerate(states.values()):
            masks = _draw_masks(batch, rates, (EVAL_SEED, index, state_index, 0))
            cond = build_model_inputs(batch, masks, [polish] * len(masks), device=device)
            spans = (cond["prefix_lengths"], cond["target_lengths"])
            uncond = build_unconditional_inputs(cond["input_ids"], *spans)
            labels, valid = gather_target(cond["labels"], *spans)

            inputs.append(
                CalibrationInput(
                    cond=cond,
                    uncond=uncond,
                    labels=labels.masked_fill(~valid.unsqueeze(1), IGNORE_INDEX),
                    cond_valid=batch["valid"].to(device),
                    uncond_valid=uncond["audio_mask"],
                )
            )

    return inputs


def ffn_grams(model: EchoModel, inputs: List[CalibrationInput]) -> List[torch.Tensor]:
    """
    Per layer, the Gram of the down_proj inputs over the real tokens of both branches.

    Taking both matters: a neuron only the unconditional branch relies on would
    otherwise look idle, and be the first to go.
    """

    return _input_grams(model, inputs, [layer.mlp.down_proj for layer in model.llm.layers])


def attention_grams(model: EchoModel, inputs: List[CalibrationInput]) -> List[torch.Tensor]:
    """
    Per layer, the Gram of the o_proj inputs - every query head's output, side by side - over both branches.
    """

    return _input_grams(model, inputs, [layer.self_attn.o_proj for layer in model.llm.layers])


def _input_grams(model: EchoModel, inputs: List[CalibrationInput], modules: List[nn.Linear]) -> List[torch.Tensor]:
    device = next(model.parameters()).device
    grams = [torch.zeros(module.in_features, module.in_features, device=device) for module in modules]
    state = {}

    def hook(index):
        def accumulate(module, args):
            activations = args[0][state["valid"]].float()
            grams[index] += activations.T @ activations
        return accumulate

    handles = [module.register_forward_pre_hook(hook(index)) for index, module in enumerate(modules)]

    try:
        with torch.inference_mode():
            for item in inputs:
                state["valid"] = item.cond_valid
                _cond_logits(model, item)
                state["valid"] = item.uncond_valid
                _uncond_logits(model, item)
    finally:
        for handle in handles:
            handle.remove()

    return grams


def branch_kl(model: EchoModel, teacher: EchoModel, inputs: List[CalibrationInput]) -> Dict[str, float]:
    """
    KL(teacher || model) on the masked positions, of the conditional branch and of the unconditional one.

    Measured against the teacher rather than the true tokens: a cut that keeps fitting the data while drifting
    from the teacher loses the generalization the teacher has.
    """

    weights = torch.tensor(codebook_weights(), device=inputs[0].labels.device)
    totals = {"cond": 0.0, "uncond": 0.0}

    with torch.inference_mode():
        for item in inputs:
            # A calibration batch holds a single state, which scores only some of the codebooks: their weights are
            # renormalized to sum to 1, as the evaluation's per-state weights do.
            scored = weights * (item.labels != IGNORE_INDEX).any(dim=2).any(dim=0)
            scored = scored / scored.sum()

            spans = (item.cond["prefix_lengths"], item.cond["target_lengths"])
            cond, _ = gather_target(_cond_logits(model, item), *spans)
            teacher_cond, _ = gather_target(_cond_logits(teacher, item), *spans)
            totals["cond"] += float(weighted_kl(cond, teacher_cond, item.labels, scored))
            totals["uncond"] += float(
                weighted_kl(_uncond_logits(model, item), _uncond_logits(teacher, item), item.labels, scored)
            )

    return {key: total / len(inputs) for key, total in totals.items()}


def _cond_logits(model: EchoModel, item: CalibrationInput) -> torch.Tensor:
    inputs = {key: value for key, value in item.cond.items() if key != "labels"}

    return model(**inputs, unconditional=False).logits


def _uncond_logits(model: EchoModel, item: CalibrationInput) -> torch.Tensor:
    # No prefix to block, and no second branch to run: a plain pass over the target alone.
    return model(**item.uncond, prefix_blocked=False, unconditional=False).logits
