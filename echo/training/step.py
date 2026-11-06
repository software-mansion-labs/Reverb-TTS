import torch

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from echo.data.constants import AUDIO_MASK_ID, IGNORE_INDEX
from echo.masking.constants import MAIN_CODEBOOKS
from echo.masking.span_masking import gather_target
from echo.model.model import EchoModel
from echo.training.config import EchoTrainingConfig
from echo.training.types import Student
from echo.utils.cfg_utils import guided_log_probs
from echo.utils.config_utils import check
from echo.utils.loss_utils import reducible_tokens, weighted_ce, weighted_kl, weighted_kl_logp


# Training step - for CE only training (teacher does not run at all)
# NOTE: teacher arg is still being passed just for compatibility
def plain_step(
    student: Student,
    teacher: Optional[EchoModel],   # Unused - this step has nothing to distil from
    inputs: Dict[str, torch.Tensor],
    config: EchoTrainingConfig,
    weights: torch.Tensor,          # [C] weight of every codebook
) -> Dict[str, torch.Tensor]:
    """
    Plain training: the model is scored only against the ground-truth tokens that were masked.
    """

    logits = student(**inputs).logits
    ce = weighted_ce(logits, inputs["labels"], _by_path(weights, config.loss.main.ce, config.loss.polish.ce))

    return {"loss": ce, "ce": ce, "kl": torch.zeros((), device=logits.device, dtype=logits.dtype)}


# Training step - standard distillation one (teacher present, CE also possible)
def distillation_step(
    student: Student,
    teacher: EchoModel,
    inputs: Dict[str, torch.Tensor],
    config: EchoTrainingConfig,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Distillation from a teacher that sees the same batch, with CE beside it if configured.

    The main-path (0-1) and the polish (2-7) codebooks weigh both terms on their own (loss.main, loss.polish).
    Token selection, if any, narrows the CE alone - the KL keeps every scored token.
    """

    main, polish = config.loss.main, config.loss.polish
    labels = inputs["labels"]
    teacher_logits = None

    if max(main.kl, polish.kl) > 0.0 or min(main.ce_selection, polish.ce_selection) < 1.0:
        with torch.no_grad(), _full_precision(inputs["input_ids"].device.type):
            # Only the conditional branch is wanted here. A teacher configured
            # for CFG would otherwise run its unconditional pass as well, and
            # this step would throw the result away.
            teacher_logits = teacher(**inputs, unconditional=False).logits

    logits = student(**inputs).logits
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)

    ce = zero
    if max(main.ce, polish.ce) > 0.0:
        selected = _selected(logits, teacher_logits, labels, main.ce_selection, polish.ce_selection)
        ce = weighted_ce(logits, selected, _by_path(weights, main.ce, polish.ce))

    kl = zero
    if max(main.kl, polish.kl) > 0.0:
        kl = weighted_kl(logits, teacher_logits, labels, _by_path(weights, main.kl, polish.kl), config.loss.temperature, config.loss.reverse_kl)

    return {"loss": ce + kl, "ce": ce, "kl": kl}


# Training step - CFG specific one, needs guided_log_probs calculation
def cfg_step(
    student: Student,
    teacher: EchoModel,   # A teacher in 'cfg.mode: teacher', which returns both branches
    inputs: Dict[str, torch.Tensor],
    config: EchoTrainingConfig,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Distillation of the guided mixture into a single pass.

    The teacher runs conditioned and unconditioned, and its two branches are
    mixed (guided_log_probs) into the distribution inference would have sampled from. 
    """

    check(
        config.loss.temperature == 1.0,
        "a CFG teacher hands over a normalized distribution, so "
        f"loss.temperature has to be 1.0, got {config.loss.temperature}",
    )

    echo_config = _unwrap(teacher).echo_config
    spans = (inputs["prefix_lengths"], inputs["target_lengths"])

    with torch.no_grad(), _full_precision(inputs["input_ids"].device.type):
        branches = teacher(**inputs, unconditional=True)
        target = guided_log_probs(
            branches.cond_logits,
            branches.uncond_logits,
            echo_config.cfg.guidance_scale,
            mask_id=AUDIO_MASK_ID,
        )

    # The student's own span, in the coordinates the mixture already uses. Rows
    # with a shorter target end in repeated positions, which the labels drop.
    logits, valid = gather_target(student(**inputs).logits, *spans)
    labels, _ = gather_target(inputs["labels"], *spans)

    labels = labels.masked_fill(~valid.unsqueeze(1), IGNORE_INDEX)
    main, polish = config.loss.main, config.loss.polish
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    ce = weighted_ce(logits, labels, _by_path(weights, main.ce, polish.ce)) if max(main.ce, polish.ce) > 0.0 else zero
    kl = weighted_kl_logp(logits, target, labels, _by_path(weights, main.kl, polish.kl), config.loss.reverse_kl)
    terms = {"loss": ce + kl, "ce": ce, "kl": kl}

    # Reported beside the loss, so an annealed run shows where its guidance stands.
    return {**terms, "guidance": torch.tensor(echo_config.cfg.guidance_scale, device=logits.device)}


# Helper
def _unwrap(model: Student) -> EchoModel:
    """
    The model itself, whatever it was wrapped in.
    """

    return getattr(model, "module", model)


# Helper
def _selected(
    logits: torch.Tensor,                      # [B, C, L, V] of the student
    teacher_logits: Optional[torch.Tensor],    # [B, C, L, V], None when no path selects
    labels: torch.Tensor,                      # [B, C, L]
    main: float,                               # Share of the main-path codebooks' (0-1) tokens kept
    polish: float,                             # Share of the polish codebooks' (2-7) tokens kept
) -> torch.Tensor:
    """
    The labels the CE is taken on: each path keeps its share of the tokens where the student trails the teacher most.
    """

    if teacher_logits is None or min(main, polish) >= 1.0:
        return labels

    selected = reducible_tokens(logits, teacher_logits, labels, polish, MAIN_CODEBOOKS).clone()
    selected[:, :MAIN_CODEBOOKS] = reducible_tokens(logits, teacher_logits, labels, main)[:, :MAIN_CODEBOOKS]

    return selected


# Helper
def _by_path(
    weights: torch.Tensor,   # [C] weight of every codebook
    main: float,             # Scale of the main-path codebooks (0-1)
    polish: float,           # Scale of the polish codebooks (2-7)
) -> torch.Tensor:
    """
    The codebook weights of one loss term, scaled per path.
    """

    scale = torch.full_like(weights, main)
    scale[MAIN_CODEBOOKS:] = polish

    return weights * scale


# Helper
@contextmanager
def _full_precision(device_type: str) -> Iterator[None]:
    """
    The teacher's forward in fp32, outside the student's autocast, with TF32 matmuls.

    Under bf16 the teacher's fine codebooks lose 0.02-0.04 nats of CE (c6/c7), a bias its KL target and token
    selection would pass on. TF32 costs nothing measurable, and is faster than full fp32.
    """

    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")

    try:
        with torch.autocast(device_type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
