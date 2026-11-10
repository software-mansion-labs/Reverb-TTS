import torch

from typing import Dict, List, Optional, Sequence, Tuple

from echo.masking.constants import (
    DECISION_SHARES,
    DEFAULT_MIN_PASS_SHARE,
    DEFAULT_POLISH_SHARE,
    DEFAULT_POLISH_WEIGHTS,
    DEFAULT_TAIL_MASKED,
    EARLY_PASSES,
    MAIN_CODEBOOKS,
    MAIN_PATH_SAMPLERS,
    TRAJECTORY_C0,
    TRAJECTORY_C1,
    TRAJECTORY_JITTER,
    TRAJECTORY_SHIFT,
)


# Audio masking - a state the two-stage sampler's main path visits
def build_main_stage_audio_mask(
    audio_tokens: torch.Tensor,                       # [C, T]
    prefix_frames: int,                               # Frames the prefix keeps, never masked
    stage_shares: Optional[Sequence[float]] = None,   # The v0.0.2 schedule: shares of the (early, middle, penultimate, final, tail) draws
    tail_masked: float = DEFAULT_TAIL_MASKED,         # Least masked fraction of codebook 0 a tail draw reaches (v0.0.2 schedule)
    samplers: Optional[Dict[str, float]] = None,      # Weights of the traced samplers in MAIN_PATH_SAMPLERS - None weighs them equally
    min_pass_share: float = DEFAULT_MIN_PASS_SHARE,   # Least probability of any pass of the drawn sampler
) -> torch.Tensor:
    """
    Draws a [C, T] mask at a main-path state: codebooks 0-1 partly masked, the rest fully.

    By default a state of one of the deployed samplers: the sampler drawn by its weight, then one of its passes by the
    share of tokens that pass decides (at least 'min_pass_share'), at that pass's traced masked fractions plus their
    traced spread. With 'stage_shares' the v0.0.2 recipe's schedule instead: a fitted trajectory of the pre-RL samplers.
    """

    if stage_shares is None:
        rates = torch.ones(audio_tokens.size(0))
        rates[:MAIN_CODEBOOKS] = torch.tensor(_draw_sampler_state(samplers, min_pass_share))
        return _draw_target_mask(rates, audio_tokens, prefix_frames)

    steps = list(DECISION_SHARES)
    n = steps[torch.randint(len(steps), (1,)).item()]
    k, tail = _draw_pass(n, stage_shares)

    if tail:
        last = _trajectory_c0((n - 1) / n)
        c0 = tail_masked + torch.rand(1).item() * (last - tail_masked)
    else:
        c0 = _trajectory_c0(k / n)
    c1 = _trajectory_c1(c0)

    # The first pass always sees everything masked.
    if k > 0 or tail:
        sd = TRAJECTORY_JITTER[0] + TRAJECTORY_JITTER[1] * (1 - c0)
        c0, c1 = (min(max(c + sd * torch.randn(1).item(), 0.0), 1.0) for c in (c0, c1))

    rates = torch.ones(audio_tokens.size(0))
    rates[:MAIN_CODEBOOKS] = torch.tensor([c0, c1])

    return _draw_target_mask(rates, audio_tokens, prefix_frames)


# Audio masking - the state of the two-stage sampler's polish pass
def build_polish_stage_audio_mask(
    audio_tokens: torch.Tensor,   # [C, T]
    prefix_frames: int,           # Frames the prefix keeps, never masked
) -> torch.Tensor:
    """
    Draws a [C, T] mask at the polish state: codebooks 0-1 complete, the rest fully masked.
    """

    rates = torch.ones(audio_tokens.size(0))
    rates[:MAIN_CODEBOOKS] = 0.0

    return _draw_target_mask(rates, audio_tokens, prefix_frames)


def codebook_weights(
    polish_share: float = DEFAULT_POLISH_SHARE,
    polish_weights: Sequence[float] = DEFAULT_POLISH_WEIGHTS,
) -> List[float]:
    """
    Loss weight of every codebook: the main-path codebooks share 1 - polish_share equally, the polished ones
    share polish_share by 'polish_weights'.

    A share of 0 or 1 gives the weights of one stage alone.
    """

    total = sum(polish_weights)

    return [(1 - polish_share) / MAIN_CODEBOOKS] * MAIN_CODEBOOKS + [polish_share * w / total for w in polish_weights]


def pass_probabilities(sampler: str, min_pass_share: float = DEFAULT_MIN_PASS_SHARE) -> List[float]:
    """
    Probability of every pass of a traced sampler: its decision share, raised to at least 'min_pass_share', with the
    passes above the floor scaled down so that they keep their proportions and the total stays 1.
    """

    total = sum(MAIN_PATH_SAMPLERS[sampler]["decisions"])   # 1 up to the rounding of the traced shares
    shares = [s / total for s in MAIN_PATH_SAMPLERS[sampler]["decisions"]]
    floored = [s < min_pass_share for s in shares]
    rest = sum(s for s, low in zip(shares, floored) if not low)
    free = 1.0 - min_pass_share * sum(floored)

    return [min_pass_share if low else s * free / rest for s, low in zip(shares, floored)]


def _draw_sampler_state(samplers: Optional[Dict[str, float]], min_pass_share: float) -> Tuple[float, float]:
    """
    Masked fractions of codebooks 0-1 at a drawn pass of a drawn sampler.
    """

    names = list(samplers or MAIN_PATH_SAMPLERS)
    weights = torch.tensor([float((samplers or {}).get(n, 1.0)) for n in names])
    sampler = names[torch.multinomial(weights, 1).item()]
    k = torch.multinomial(torch.tensor(pass_probabilities(sampler, min_pass_share)), 1).item()

    masked, spread = MAIN_PATH_SAMPLERS[sampler]["masked"][k], MAIN_PATH_SAMPLERS[sampler]["spread"][k]

    return tuple(min(max(m + s * torch.randn(1).item(), 0.0), 1.0) for m, s in zip(masked, spread))


def _draw_pass(n: int, stage_shares: Sequence[float]) -> Tuple[int, bool]:
    """
    A pass of the N-step sampler, and whether the draw goes to the tail past its last pass.
    """

    shares = torch.tensor(DECISION_SHARES[n])

    stages = [range(EARLY_PASSES), range(EARLY_PASSES, n - 2), [n - 2], [n - 1], [n - 1]]
    stage = torch.multinomial(torch.tensor(stage_shares, dtype=torch.float32), 1).item()

    # Within a stage the later passes, which decide more, still lean ahead.
    passes = list(stages[stage])
    k = passes[torch.multinomial(shares[passes] ** 0.5, 1).item()]

    return k, stage == len(stages) - 1


def _trajectory_c0(u: float) -> float:
    """
    Masked fraction of codebook 0 at main-path progress u.
    """

    a, e = TRAJECTORY_C0
    tau = TRAJECTORY_SHIFT * u / (1 - (1 - TRAJECTORY_SHIFT) * u)

    return 1 - a * tau ** e


def _trajectory_c1(c0: float) -> float:
    """
    Masked fraction of codebook 1 at the state where codebook 0 has 'c0' masked.
    """

    b, q = TRAJECTORY_C1

    return 1 - b * (1 - c0) ** q


def _draw_target_mask(
    rates: torch.Tensor,          # [C] masking rate of every codebook
    audio_tokens: torch.Tensor,
    prefix_frames: int,
) -> torch.Tensor:
    """
    Flips each target slot with the probability its codebook was given; the
    prefix is always left alone.
    """

    mask = torch.rand(audio_tokens.shape) < rates.unsqueeze(1)
    mask[:, :prefix_frames] = False

    return mask
