import math

import torch
import torch.nn.functional as F

from typing import Dict, Optional

from echo.data.constants import IGNORE_INDEX


def weighted_ce(
    logits: torch.Tensor,   # [B, C, L, V]
    labels: torch.Tensor,   # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    weights: torch.Tensor,  # [C] weight of every codebook
) -> torch.Tensor:
    """
    Cross entropy over the scored positions, reduced per codebook and then
    weighted - the same loss the base model computes when given labels.

    It is recomputed here rather than read off the model, so a subset of the
    batch can be scored on its own without another forward pass.
    """

    per_token = F.cross_entropy(
        logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX
    )

    return _weighted_mean(per_token, labels, weights)


def weighted_kl(
    student_logits: torch.Tensor,   # [B, C, L, V]
    teacher_logits: torch.Tensor,   # [B, C, L, V]
    labels: torch.Tensor,           # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    weights: torch.Tensor,          # [C] weight of every codebook
    temperature: float = 1.0,       # Softens both sides, and the loss is scaled back by its square
    reverse: bool = False,          # KL(student || teacher) instead: mode-seeking rather than mass-covering
) -> torch.Tensor:
    """
    KL(teacher || student), reduced exactly the way the cross entropy is: a mean
    over the scored positions of each codebook, then a weighted sum over them.
    """

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    if reverse:
        divergence = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    else:
        divergence = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    # Softening shrinks the gradients by roughly the square of the temperature,
    # so the term is scaled back to keep its weight against the cross entropy.
    return _weighted_mean(divergence, labels, weights) * temperature**2


def weighted_kl_logp(
    student_logits: torch.Tensor,     # [B, C, L, V]
    teacher_log_probs: torch.Tensor,  # [B, C, L, V] a distribution, not scores - already normalized
    labels: torch.Tensor,             # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    weights: torch.Tensor,            # [C] weight of every codebook
    reverse: bool = False,            # KL(student || teacher) instead: mode-seeking rather than mass-covering
) -> torch.Tensor:
    """
    The same divergence, against a teacher that hands over log probabilities
    rather than logits - which is what a CFG teacher's mixture is.
    """

    if not reverse:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()

        divergence = torch.where(
            teacher_probs > 0,
            teacher_probs * (teacher_log_probs - student_log_probs),
            torch.zeros_like(teacher_probs),
        ).sum(dim=-1)

        return _weighted_mean(divergence, labels, weights)

    # Tokens the teacher rules out entirely (the mask token of a guided mixture) are ruled out for the student
    # too, as the sampler never commits them - otherwise any mass there costs an infinite divergence. The rest
    # of the teacher's log probs get a floor, so a token it finds all but impossible costs a bounded penalty.
    # Excluded entries are then zeroed as finite values, never as 0 * -inf, which would turn the gradient into NaN.
    excluded = torch.isneginf(teacher_log_probs)
    student_log_probs = F.log_softmax(student_logits.masked_fill(excluded, float("-inf")), dim=-1)
    student_probs = student_log_probs.exp()

    gap = (student_log_probs - teacher_log_probs.clamp(min=-30.0)).masked_fill(excluded, 0.0)
    divergence = (student_probs * gap).sum(dim=-1)

    return _weighted_mean(divergence, labels, weights)


def distillation_loss(
    student_logits: torch.Tensor,               # [B, C, L, V]
    labels: torch.Tensor,                       # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    weights: torch.Tensor,                      # [C] weight of every codebook
    teacher_logits: Optional[torch.Tensor] = None,     # [B, C, L, V] of a plain teacher
    teacher_log_probs: Optional[torch.Tensor] = None,  # [B, C, L, V] mixture of a CFG teacher
    ce_weight: float = 0.0,
    kl_weight: float = 1.0,
    temperature: float = 1.0,   # Only reaches the logits teacher - see 'weighted_kl_logp'
    reverse_kl: bool = False,   # KL(student || teacher) instead of KL(teacher || student)
) -> Dict[str, torch.Tensor]:
    """
    A composition of CE loss and KL divergence loss - controlled by the weights.

    Defaults to KLD-only type of loss.
    """

    if kl_weight > 0.0 and (teacher_logits is None) == (teacher_log_probs is None):
        raise ValueError(
            "a positive kl_weight needs exactly one of 'teacher_logits' or "
            "'teacher_log_probs'"
        )

    zero = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    ce = (
        weighted_ce(student_logits, labels, weights) if ce_weight > 0.0 else zero
    )

    if kl_weight <= 0.0:
        kl = zero
    elif teacher_logits is not None:
        kl = weighted_kl(student_logits, teacher_logits, labels, weights, temperature, reverse_kl)
    else:
        kl = weighted_kl_logp(student_logits, teacher_log_probs, labels, weights, reverse_kl)

    return {"loss": ce_weight * ce + kl_weight * kl, "ce": ce, "kl": kl}


def reducible_tokens(
    student_logits: torch.Tensor,     # [B, C, L, V]
    reference_logits: torch.Tensor,   # [B, C, L, V] of a model the student learns towards
    labels: torch.Tensor,             # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    keep: float,                      # Share of every codebook's scored tokens left scored
    first_codebook: int = 0,          # Codebooks below this keep all their tokens
) -> torch.Tensor:
    """
    The labels with only the tokens where the student trails the reference most
    left scored (the highest student CE minus reference CE, per codebook).
    """

    if keep >= 1.0:
        return labels

    with torch.no_grad():
        excess = F.cross_entropy(
            student_logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX
        ) - F.cross_entropy(
            reference_logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX
        )

    selected = labels.clone()
    for codebook in range(first_codebook, labels.shape[1]):
        rows, positions = (labels[:, codebook] != IGNORE_INDEX).nonzero(as_tuple=True)
        values = excess[rows, codebook, positions]
        dropped = torch.ones_like(values, dtype=torch.bool)
        dropped[values.topk(math.ceil(keep * values.numel())).indices] = False
        selected[rows[dropped], codebook, positions[dropped]] = IGNORE_INDEX

    return selected


def _weighted_mean(
    per_token: torch.Tensor,   # [B, C, L] the quantity being reduced
    labels: torch.Tensor,      # [B, C, L], IGNORE_INDEX outside the scored (masked) positions
    weights: torch.Tensor,     # [C] weight of every codebook
) -> torch.Tensor:
    """
    A mean over the scored positions of every codebook, then a weighted sum over
    the codebooks - the reduction both losses share.
    """

    scored = (labels != IGNORE_INDEX).float()
    per_codebook = (per_token * scored).sum(dim=(0, 2)) / scored.sum(dim=(0, 2)).clamp(
        min=1.0
    )

    return (per_codebook * weights).sum()
