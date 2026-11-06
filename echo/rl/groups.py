"""
A groups directory: the takes of a set of prompts, generated together, and their scores. Every step after
generation reads one.

    prompts.csv   the prompts (an id, then echo.rl.prompts.FIELDS)
    seeds.json    {prompt id: the seeds of its takes}
    takes/        <prompt id>_<seed>.npy, int16 [8, frames]
    scored/       <prompt id>.json: keys, score, ctc_loss, words and ctc_gated per take (echo.rl.critic)
    critic.json   the critic package the groups were scored with
"""

import json

from pathlib import Path
from typing import Dict, List

import numpy as np

from echo.rl.prompts import FIELDS, read_rows, write_rows


def write_groups(groups: Path, rows: List[Dict[str, str]], seeds: Dict[str, List[int]]) -> None:
    write_rows(groups / "prompts.csv", rows, ["id"] + FIELDS)
    (groups / "seeds.json").write_text(json.dumps(seeds))


def read_prompts(groups: Path) -> List[Dict[str, str]]:
    return read_rows(groups / "prompts.csv")


def read_seeds(groups: Path) -> Dict[str, List[int]]:
    return json.loads((groups / "seeds.json").read_text())


def read_take(groups: Path, key: str) -> np.ndarray:
    return np.load(groups / "takes" / f"{key}.npy")


def read_scored(groups: Path, prompt_id: str) -> dict:
    return json.loads((groups / "scored" / f"{prompt_id}.json").read_text())
