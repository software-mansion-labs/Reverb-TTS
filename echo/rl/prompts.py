"""
RL prompts: (reference voice, target text) rows in pipe-separated files, and the model inputs they become.
"""

import csv

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

from echo.rl.sampler import estimate_frames

FIELDS = ["reference", "reference_transcript", "speaker_id", "language", "text", "text_language", "pool", "subcategory"]

RL_DATA = Path("data/rl")   # The reference codec paths of RL prompt files are relative to it
HELDOUT = RL_DATA / "heldout.csv"
INDIVIDUAL = RL_DATA / "targets" / "individual.csv"
TEST_REFERENCES = Path("data/test/test_in_processed.csv")
TEST_TARGETS = Path("data/test/test_out.csv")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))


def write_rows(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["|".join(fields)] + ["|".join(str(r[f]) for f in fields) for r in rows]) + "\n",
                    encoding="utf-8")


def reserved_texts() -> Set[str]:
    """
    Texts RL never trains on: the held-out and test evaluation texts, and the texts the user ranked (their
    individual-pool rows carry an origin, <voice>-t<n>), on which the critic was fitted.
    """

    reserved = {r["text"] for r in read_rows(HELDOUT)} | {r["text"] for r in read_rows(TEST_TARGETS)}
    return reserved | {r["text"] for r in read_rows(INDIVIDUAL) if r["origin"] != "new"}


def prompt_text(row: Dict[str, str]) -> str:
    return (f"<|denoise|><|lang_start|>{row['text_language']}<|lang_end|><|instruct_start|>None<|instruct_end|>"
            f"<|text_start|>{row['reference_transcript']} {row['text']}<|text_end|>")


def model_inputs(rows: List[Dict[str, str]], tokenizer, root: Path = RL_DATA) -> List[dict]:
    """
    Each row with its text tokens (ids), reference codes [8, R] (from root / reference) and target length (frames).
    """

    prompts = []
    for r in rows:
        reference = torch.from_numpy(np.load(root / r["reference"]).astype(np.int64))
        prompts.append(dict(r, ids=tokenizer(prompt_text(r), return_tensors="pt").input_ids[0], reference=reference,
                            frames=estimate_frames(r["text"], r["reference_transcript"], reference.size(1))))
    return prompts


def test_rows() -> List[Dict[str, str]]:
    """
    The test set as prompts t000-t170: every target of test_out.csv with its speaker's reference (relative to
    data/test).
    """

    references = {r["speaker_id"]: r for r in read_rows(TEST_REFERENCES)}
    rows = []
    for i, t in enumerate(read_rows(TEST_TARGETS)):
        r = references[t["speaker_id"]]
        rows.append(dict(id=f"t{i:03d}", reference=r["name"], reference_transcript=r["transcript"],
                         speaker_id=t["speaker_id"], language=t["language"], text=t["text"],
                         text_language=t["language"]))
    return rows
