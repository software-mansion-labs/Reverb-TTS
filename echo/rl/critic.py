"""
The RL critic, a versioned package folder (runs/critic/vN: critic.py, its weights, README) loaded by path. It reads
takes through its own checkpoint, whatever the policy is, and its scores mean something only within one group.
"""

import importlib.util
import sys

from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np

CHECKPOINT = Path("runs/final/fp32")   # The Echo checkpoint both critic versions were fitted with


def load_package(package: Path) -> ModuleType:
    """
    The package's critic.py, imported once per process as the module `critic`.
    """

    loaded = sys.modules.get("critic")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != (package / "critic.py").resolve():
            raise ValueError(f"critic {loaded.__file__} is loaded already, not {package}")
        return loaded
    sys.path.insert(0, str(package))   # critic.py imports its own ctc_alignment.py
    spec = importlib.util.spec_from_file_location("critic", package / "critic.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["critic"] = module
    spec.loader.exec_module(module)
    return module


class GroupCritic:
    """
    A critic package's critic, CTC gate and confidence margin.
    """

    def __init__(self, package: Path, checkpoint: Path = CHECKPOINT) -> None:
        self.module = load_package(package)
        self.critic = self.module.Critic(checkpoint=checkpoint, weights=package / "critic_weights.npz")
        self.margin = self.module.MARGIN

    def score(self, prompt: dict, codes: List[np.ndarray], keys: List[str]) -> Dict[str, list]:
        """
        One group of takes of a prompt (with model_inputs()' reference): critic score, CTC loss, per-word
        alignment and CTC gate per take. The keys seed the critic's own masks, so a take keeps its score.
        """

        takes = [self.critic.prepare(c) for c in codes]
        group = self.critic.features(takes, prompt["text_language"], prompt["reference"].numpy(),
                                     prompt["reference_transcript"], prompt["text"], keys=keys)
        return dict(keys=keys, score=self.critic.scores(group).tolist(), ctc_loss=group["ctc_loss"].tolist(),
                    words=[None if w is None else w.tolist() for w in group["words"]],
                    ctc_gated=self.module.ctc_gate(group["ctc_loss"], group["words"]).tolist())
