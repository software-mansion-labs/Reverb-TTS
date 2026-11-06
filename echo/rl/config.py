from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from echo.training.config import TrackingConfig
from echo.utils.config_utils import build_config, read_json


# Subconfig - which pairs of a group's takes DPO trains on (echo.rl.pairs)
@dataclass
class PairsConfig:
    ctc_pairs: int = 4            # CTC pairs per prompt, at most
    critic_pairs: int = 6         # Critic pairs per prompt, at most
    take_uses: int = 2            # Pairs a take may appear in, per kind
    ctc_margin: float = 1.0       # CTC gap a pair needs, in units of max(0.3, 0.5 x the group's median CTC loss)
    word_drop: float = 0.4        # A word this far below its group median counts as dropped (critic.ctc_gate's rule)
    critic_margin: float = 4.0    # Critic gap a pair of takes CTC leaves undecided needs
    acceptable_ctc: float = 1.0   # Critic pairs only between takes with a CTC loss at most this, and the median + unit
    ctc_filter: bool = False      # CTC pairs only from groups with a gated take, and with a loser word > word_drop worse
    ctc_share: float = 0.0        # CTC pairs weighted to this share of the loss (weight 1 to 2), 0 leaves them at 1


# Subconfig - the DPO run (echo.rl.dpo)
@dataclass
class TrainConfig:
    lr: float = 1e-5              # Peak learning rate, after a linear warm-up
    lr_min: float = 0.1           # Final learning rate of the cosine decay, as a share of lr
    warmup: int = 20              # Warm-up steps
    pairs_per_step: int = 32
    epochs: int = 1
    beta: float = 5.0             # DPO temperature on a pair's margin
    kl: float = 1.0               # Weight of KL(reference || policy) at the masked slots
    winner_anchor: float = 2.0    # Weight of the squared winner log-ratio, which keeps winners from falling with losers
    masks: int = 8                # Masks a take is scored under
    short_masks: int = 16         # Masks for takes under short_frames frames, whose score is the noisiest
    short_frames: int = 100
    probe_groups: Optional[str] = "data/rl/probe"   # Scored groups of held-out prompts for the pair probe, None for none
    probe_every: int = 50         # Steps between probes
    save_at: List[int] = field(default_factory=list)   # Steps after which the model is also saved


@dataclass
class RLConfig:
    """
    One DPO recipe: its pair rules, its training run and where it reports, loaded from a single .json file.
    """

    name: str = "base"
    notes: str = ""
    pairs: PairsConfig = field(default_factory=PairsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RLConfig":
        return build_config(cls, read_json(path), strict=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
