import random
import torch

from typing import Any, Dict, List, Optional, Sequence

from echo.data.constants import AUDIO_MASK_ID, IGNORE_INDEX
from echo.data.inputs import build_model_inputs
from echo.data.loader import EchoDataLoader
from echo.eval.constants import EVAL_SEED, POLISH_STATE, REPEATS, TRAJECTORY
from echo.eval.misc import print_summary
from echo.eval.types import Breakdown, Row, Trajectory
from echo.masking.audio_masking import _draw_target_mask, codebook_weights
from echo.masking.constants import DEFAULT_POLISH_SHARE, DEFAULT_POLISH_WEIGHTS
from echo.masking.span_masking import gather_target
from echo.model.model import EchoModel
from echo.utils.cfg_utils import guided_log_probs
from echo.utils.seed_utils import seed_of
from echo.utils.loss_utils import weighted_ce, weighted_kl, weighted_kl_logp


def evaluate_single_step(
    student: EchoModel,
    teacher: Optional[EchoModel],            # Optional, its presence adds the KL divergence to the report
    data_loader: EchoDataLoader,
    max_batches: int = 50,
    trajectory: Trajectory = TRAJECTORY,     # Traced main-path states of the deployed sampler; the sweep adds the polish state
    repeats: int = REPEATS,                  # Masks drawn per sample at every grid point
    seed: int = EVAL_SEED,
    per_language: bool = False,              # Adds a breakdown by language, pooled like the sampler's row
    guidance_scale: Optional[float] = None,  # Scores against the teacher's guided (CFG) mixture at this w instead
    polish_share: float = DEFAULT_POLISH_SHARE,                  # Weight of the polish row in the sampler's row
    polish_weights: Sequence[float] = DEFAULT_POLISH_WEIGHTS,    # Loss weights of codebooks 2-7 at the polish state
) -> Dict[Any, Row]:
    """
    Scores a single denoising step at the states the deployed sampler feeds the model.

    The default loss is CE. If `teacher` model is given, scores also it's CE and the KL divergence between them.

    Rows are keyed by state: the traced main-path ones ('s8_f7'), scored on codebooks 0-1, and the polish one
    ('s8_polish'), scored on codebooks 2-7. 's8_main' pools the main-path states by the share of codebook 0-1 tokens
    each one decides. The last row, named after the sampler ('s8'), joins the main-path and the polish rows as
    training splits the loss: 1 - polish_share to polish_share.

    With 'per_language' the same numbers come back under a 'languages' key, one
    entry per language, pooled like the sampler's row.

    With 'guidance_scale' the teacher's side is the distribution its guided sampler decodes from, and both
    sides are scored on the target span - what a CFG-distilled student is trained towards.
    """

    # Each state is scored on the codebooks its pass decides, weighted as that sample type is in training.
    weights = {
        False: torch.tensor(codebook_weights(0.0, polish_weights)),
        True: torch.tensor(codebook_weights(1.0, polish_weights)),
    }
    device = next(student.parameters()).device

    sampler = next(iter(trajectory)).rsplit("_", 1)[0]
    states = {point: (rates, False) for point, (rates, _) in trajectory.items()}
    states[f"{sampler}_polish"] = (POLISH_STATE, True)
    grid = list(states)

    # Per grid point sums of loss * positions, and of positions.
    totals = {
        point: {"ce": 0.0, "ce_teacher": 0.0, "kl": 0.0, "positions": 0.0} for point in grid
    }

    # Per grid point counts of the masked slots of every codebook, and of the
    # target slots they were drawn from - their ratio is the masking profile.
    masked = {point: torch.zeros(len(POLISH_STATE)) for point in grid}
    slots = {point: 0.0 for point in grid}

    # The same sums per language and grid point.
    languages: Dict[str, Breakdown] = {}

    was_training = student.training
    student.eval()
    if teacher is not None:
        teacher.eval()

    # The sweep draws from the global RNG, so the caller's stream is put back
    # afterwards - otherwise every batch following an evaluation replays it.
    python_state, torch_state = random.getstate(), torch.get_rng_state()

    try:
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= max_batches:
                break

            # 1 sample -> N samples with different masking ratios (N = number of grid points)
            for point_index, (point, (rates, polish)) in enumerate(states.items()):
                for repeat in range(repeats):
                    masks = _draw_masks(batch, rates, (seed, batch_index, point_index, repeat))
                    inputs = build_model_inputs(batch, masks, [polish] * len(masks), device=device)
                    labels = inputs["labels"]

                    with torch.inference_mode():
                        student_logits = student(**inputs).logits
                        if teacher is not None and guidance_scale is not None:
                            branches = teacher(**inputs, unconditional=True)
                            teacher_logits = guided_log_probs(
                                branches.cond_logits, branches.uncond_logits, guidance_scale, mask_id=AUDIO_MASK_ID
                            )
                            # The mixture lives on the target span, so the student and the labels are cut to it.
                            spans = (inputs["prefix_lengths"], inputs["target_lengths"])
                            student_logits, valid = gather_target(student_logits, *spans)
                            labels = gather_target(labels, *spans)[0].masked_fill(~valid.unsqueeze(1), IGNORE_INDEX)
                        else:
                            teacher_logits = (
                                teacher(**inputs).logits if teacher is not None else None
                            )

                    guided = guidance_scale is not None

                    scores = _score(
                        student_logits, teacher_logits, labels, weights[polish].to(device), guided
                    )
                    for key, value in scores.items():
                        totals[point][key] += value

                    for language in set(batch["language"]) if per_language else ():
                        rows = torch.tensor(
                            [name == language for name in batch["language"]],
                            device=labels.device,
                        )
                        scores = _score(
                            student_logits,
                            teacher_logits,
                            torch.where(rows[:, None, None], labels, IGNORE_INDEX),
                            weights[polish].to(device),
                            guided,
                        )
                        totals_of = languages.setdefault(language, {}).setdefault(
                            point,
                            {"ce": 0.0, "ce_teacher": 0.0, "kl": 0.0, "positions": 0.0},
                        )
                        for key, value in scores.items():
                            totals_of[key] += value

                    masked[point] += torch.stack([mask.sum(dim=1) for mask in masks]).sum(0)
                    slots[point] += float(sum(batch["target_length"]))
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        student.train(was_training)

    results = {
        point: {
            **_pooled(total),
            "mask_ratios": (masked[point] / max(slots[point], 1.0)).tolist(),
        }
        for point, total in totals.items()
    }

    results = _pooled_sampler(results, trajectory, sampler, polish_share)

    by_language = (
        {
            language: _pooled_sampler(
                {point: _pooled(total) for point, total in points.items()}, trajectory, sampler, polish_share
            )[sampler]
            for language, points in sorted(languages.items())
        }
        if per_language
        else None
    )

    print_summary(results, teacher is not None, by_language)

    return {**results, "languages": by_language} if per_language else results


def _pooled(total: Dict[str, float]) -> Dict[str, float]:
    """
    Sums of loss * positions turned back into the losses they stand for.
    """

    positions = max(total["positions"], 1.0)

    return {
        "ce": total["ce"] / positions,
        "ce_teacher": total["ce_teacher"] / positions,
        "ce_delta": (total["ce"] - total["ce_teacher"]) / positions,
        "kl": total["kl"] / positions,
        "positions": total["positions"],
    }


def _pooled_sampler(
    results: Dict[Any, Row],
    trajectory: Trajectory,
    sampler: str,          # The sampler's name, prefixing its state rows ('s8')
    polish_share: float,
) -> Dict[Any, Row]:
    """
    The state rows, then the main-path pool and the sampler's row - main path and polish split by 'polish_share'.
    """

    main, polish = f"{sampler}_main", f"{sampler}_polish"
    pooled = {**results, main: _weighted(results, trajectory)}
    pooled[sampler] = _weighted(pooled, {main: ((), 1.0 - polish_share), polish: ((), polish_share)})

    return pooled


def _weighted(results: Dict[Any, Row], trajectory: Trajectory) -> Row:
    """
    The traced states' rows averaged by the share of the grid each state decides.
    """

    share = sum(weight for _, weight in trajectory.values())
    rows = [(results[label], weight / share) for label, (_, weight) in trajectory.items()]

    row: Row = {key: sum(r[key] * w for r, w in rows) for key in ("ce", "ce_teacher", "ce_delta", "kl")}
    row["positions"] = sum(r["positions"] for r, _ in rows)
    # The per-language rows carry no masking profile.
    if "mask_ratios" in rows[0][0]:
        row["mask_ratios"] = [sum(r["mask_ratios"][c] * w for r, w in rows) for c in range(len(rows[0][0]["mask_ratios"]))]

    return row


def _draw_masks(
    batch: Dict[str, List[Any]],
    rates: Sequence[float],   # The masking rate of every codebook
    keys: Sequence[int],      # Identifies the draw within the sweep, making it reproductible
) -> List[torch.Tensor]:
    """
    One [C, T] mask per sample, all at the same per-codebook rates.

    Each sample gets its own seed, so the draws stay independent of one another
    and of how the samples happened to be batched.
    """

    masks = []

    for index, audio in enumerate(batch["audio_tokens"]):
        torch.manual_seed(seed_of(*keys, index))

        # The prefix is what precedes the target, minus the prompt and the text.
        prefix_frames = (
            batch["prefix_length"][index]
            - batch["prompt_tokens"][index].numel()
            - batch["text_tokens"][index].numel()
        )
        mask = _draw_target_mask(torch.tensor(rates), audio, prefix_frames)

        # Padding sits behind the target, and is never scored.
        frames = prefix_frames + batch["target_length"][index]
        mask[:, frames:] = False

        masks.append(mask)

    return masks


def _score(
    student_logits: torch.Tensor,             # [B, C, L, V]
    teacher_logits: Optional[torch.Tensor],   # [B, C, L, V], None without a teacher
    labels: torch.Tensor,                     # [B, C, L], IGNORE_INDEX off the scored positions
    weights: torch.Tensor,                    # [C] weight of every codebook
    guided: bool = False,                     # The teacher's side is a guided mixture's log probs, not logits
) -> Dict[str, float]:
    """
    CE of both models, and their divergence, summed over the positions they
    were measured on.

    Narrowing 'labels' to some of the rows scores that subset alone, which is
    how the per language numbers are taken without a second pass.
    """

    positions = float((labels != IGNORE_INDEX).sum())
    teacher_ce, kl = 0.0, 0.0

    if teacher_logits is not None:
        teacher_ce = float(weighted_ce(teacher_logits, labels, weights))
        divergence = weighted_kl_logp if guided else weighted_kl
        kl = float(divergence(student_logits, teacher_logits, labels, weights))

    return {
        "ce": float(weighted_ce(student_logits, labels, weights)) * positions,
        "ce_teacher": teacher_ce * positions,
        "kl": kl * positions,
        "positions": positions,
    }
