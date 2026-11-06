"""
Which pairs of a group's takes DPO trains on (knowledge/training/rl_dpo.md):
    - CTC pairs: a CTC gap of at least ctc_margin units, a unit being max(0.3, 0.5 x the group's median CTC loss), or
      half a unit when the higher-CTC take dropped a word;
    - critic pairs: acceptable takes that CTC leaves undecided, with a critic gap of at least critic_margin;
largest gap first, at most ctc_pairs / critic_pairs per prompt, each take in at most take_uses pairs per kind. The
thresholds come from the group, i.e. from the round's starting model, so they tighten as the model improves.
"""

import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from echo.rl.config import PairsConfig


@dataclass
class Pair:
    groups: str           # The groups directory both takes are in
    prompt: str           # The prompt's id
    winner: str           # Take keys, <prompt id>_<seed>
    loser: str
    kind: str             # 'ctc' or 'critic'
    weight: float = 1.0   # Factor on the pair's loss


def dropped_words(words: List[Optional[list]], word_drop: float) -> np.ndarray:
    """
    critic.ctc_gate's word condition alone: no alignment, or a word more than word_drop below its group median.
    """

    aligned = [np.array(w) for w in words if w is not None]
    n = min((len(w) for w in aligned), default=0)
    med = np.median(np.array([w[:n] for w in aligned]), axis=0) if aligned else None
    return np.array([w is None or med is None or float((med[:n] - np.array(w)[:n]).max()) > word_drop for w in words])


def word_evidence(group: dict, a: int, b: int, word_drop: float) -> bool:
    """
    The CTC filter: the group has a gated take, and a word of loser b is more than word_drop worse than in winner a
    (the user's listening, 2026-11-03: CTC pairs without either were mostly equal to the ear).
    """

    wa, wb = group["words"][a], group["words"][b]
    if not any(group["ctc_gated"]) or wa is None:
        return False
    n = min(len(wa), len(wb)) if wb is not None else 0
    return wb is None or float((np.array(wa[:n]) - np.array(wb[:n])).max()) > word_drop


def group_pairs(group: dict, config: PairsConfig) -> List[Tuple[int, int, str]]:
    """
    (winner, loser, kind) take indices of one scored group.
    """

    loss, s = np.array(group["ctc_loss"]), np.array(group["score"])
    med = float(np.median(loss))
    unit = max(0.3, 0.5 * med)
    drop = dropped_words(group["words"], config.word_drop)

    def by_ctc(a: int, b: int) -> bool:          # a has the lower CTC loss
        gap = loss[b] - loss[a]
        return gap >= config.ctc_margin * unit or (gap >= 0.5 * unit and drop[b])

    def greedy(candidates: list, cap: int) -> List[Tuple[int, int]]:
        """
        Largest gap first, each take in at most take_uses of the chosen pairs.
        """

        uses, picked = {}, []
        for _, a, b in sorted(candidates, reverse=True):
            if len(picked) == cap:
                break
            if uses.get(a, 0) >= config.take_uses or uses.get(b, 0) >= config.take_uses:
                continue
            uses[a], uses[b] = uses.get(a, 0) + 1, uses.get(b, 0) + 1
            picked.append((a, b))
        return picked

    n = len(loss)
    candidates = [(loss[b] - loss[a], a, b) for a in range(n) for b in range(n) if loss[a] < loss[b] and by_ctc(a, b)
                  and (not config.ctc_filter or word_evidence(group, a, b, config.word_drop))]
    pairs = [(a, b, "ctc") for a, b in greedy(candidates, config.ctc_pairs)]
    ok = np.flatnonzero((loss <= config.acceptable_ctc) & (loss <= med + unit))
    candidates = [(abs(s[a] - s[b]), a, b) for a in ok for b in ok if loss[a] <= loss[b] and a != b and not by_ctc(a, b)
                  and abs(s[a] - s[b]) >= config.critic_margin]
    return pairs + [(a, b, "critic") if s[a] > s[b] else (b, a, "critic") for a, b in greedy(candidates, config.critic_pairs)]


def load_pairs(groups: Path, config: PairsConfig, ctc_only: bool = False) -> List[Pair]:
    """
    The pairs of every scored group in a groups directory, in prompt order; with ctc_only its CTC pairs alone.
    """

    found = []
    for f in sorted((groups / "scored").glob("*.json")):
        group = json.loads(f.read_text())
        found += [Pair(str(groups), f.stem, group["keys"][w], group["keys"][lo], kind)
                  for w, lo, kind in group_pairs(group, config) if not (kind == "critic" and ctc_only)]
    return found


def weigh(pairs: List[Pair], ctc_share: float) -> float:
    """
    Weights the CTC pairs so they carry ctc_share of the loss, clipped to 1-2 (1 when ctc_share is 0), and returns
    that weight.
    """

    ctc, critic = sum(p.kind == "ctc" for p in pairs), sum(p.kind == "critic" for p in pairs)
    weight = min(2.0, max(1.0, ctc_share * critic / ((1 - ctc_share) * ctc))) if ctc_share else 1.0
    for p in pairs:
        p.weight = weight if p.kind == "ctc" else 1.0
    return weight


def write_pairs(path: Path, pairs: List[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")


def read_pairs(path: Path) -> List[Pair]:
    return [Pair(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
