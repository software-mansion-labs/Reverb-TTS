import torch

from typing import List, Optional, Sequence, Tuple

from echo.masking.constants import (
    DECISION_SHARES,
    DEFAULT_POLISH_SHARE,
    DEFAULT_POLISH_WEIGHTS,
    DEFAULT_TAIL_MASKED,
    EARLY_PASSES,
    MAIN_CODEBOOKS,
    TRAJECTORY_C0,
    TRAJECTORY_C1,
    TRAJECTORY_JITTER,
    TRAJECTORY_SHIFT,
)


# Audio masking - a state the two-stage sampler's main path visits
def build_main_stage_audio_mask(
    audio_tokens: torch.Tensor,                       # [C, T]
    prefix_frames: int,                               # Frames the prefix keeps, never masked
    stage_shares: Optional[Sequence[float]] = None,   # Shares of the (early, middle, penultimate, final, tail) draws
    tail_masked: float = DEFAULT_TAIL_MASKED,         # Least masked fraction of codebook 0 a tail draw reaches
    decision_power: float = 1.0,                      # Passes drawn by decision share to this power - 0 draws them uniformly
) -> torch.Tensor:
    """
    Draws a [C, T] mask at a main-path state: codebooks 0-1 partly masked (according to some trajectory point), the rest fully.
    """

    steps = list(DECISION_SHARES)
    n = steps[torch.randint(len(steps), (1,)).item()]
    k, tail = _draw_pass(n, stage_shares, decision_power)

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


def _draw_pass(n: int, stage_shares: Optional[Sequence[float]], decision_power: float) -> Tuple[int, bool]:
    """
    A pass of the N-step sampler, and whether the draw goes to the tail past its last pass.
    """

    shares = torch.tensor(DECISION_SHARES[n])

    if stage_shares is None:
        return torch.multinomial(shares ** decision_power, 1).item(), False

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
