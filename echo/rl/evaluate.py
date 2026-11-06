"""
Evaluation of an RL model with the deployed sampler (s8, no remasking, 2-pass polish) and seeds fixed by position, so
two models' takes of a prompt pair up one to one:
    held   the 64 held-out vulnerable prompts (data/rl/heldout.csv) x 16 seeds, never trained on;
    test   the test set's 171 targets x 3 seeds.
Takes are decoded and trimmed as the critic prepares them and transcribed by Whisper (echo.eval.metrics); test takes
also get their WavLM similarity to the reference. WER is read with and without numbers: Whisper writes numbers as
digits, which is most of every model's errors, so "without numbers" leaves out the errors of takes whose only errors
are numbers.
"""

import difflib
import math
import time
import unicodedata

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from echo.rl.critic import GroupCritic, load_package
from echo.rl.prompts import HELDOUT, model_inputs, read_rows, test_rows
from echo.rl.sampler import jobs_for, pack

SETS = ("held", "test")
TEST_ROOT = Path("data/test")   # The test references' codec paths are relative to it
ROW_FIELDS = ["prompt", "seed", "language", "errors", "words", "numbers", "similarity", "hypothesis"]


def eval_jobs(name: str, tokenizer, offset: int = 0) -> Tuple[List[dict], List[List[dict]]]:
    """
    The takes of one evaluation set, and the batches they are generated in. `offset` shifts every seed, for fresh
    takes of the same prompts.
    """

    if name == "held":
        prompts = model_inputs(read_rows(HELDOUT), tokenizer)
        seeds = {p["id"]: [777000 + 100 * int(p["id"][1:]) + k + offset for k in range(16)] for p in prompts}
        jobs = jobs_for(prompts, seeds)
        return jobs, [jobs[i:i + 8] for i in range(0, len(jobs), 8)]   # 8 at a time in prompt order, as ever since round 1
    prompts = model_inputs(test_rows(), tokenizer, TEST_ROOT)
    seeds = {p["id"]: [4200 + 10 * i + k + offset for k in range(3)] for i, p in enumerate(prompts)}
    jobs = jobs_for(prompts, seeds)
    return jobs, pack(jobs)


def only_numbers(reference: str, hypothesis: str) -> bool:
    """
    Whether every edit between the two texts touches a word with a digit in it: Whisper writing a number as digits.
    """

    def words(text: str) -> List[str]:
        return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.lower()).split()

    ref, hyp = words(reference), words(hypothesis)
    edits = [o for o in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes() if o[0] != "equal"]
    return bool(edits) and all(any(ch.isdigit() for w in ref[i1:i2] + hyp[j1:j2] for ch in w) for _, i1, i2, j1, j2 in edits)


def transcribe_takes(jobs: List[dict], takes: Path, package: Path, similarity: bool) -> List[dict]:
    """
    One row per take: Whisper's word errors, whether they are only numbers, and with `similarity` the WavLM
    similarity to the decoded reference.
    """

    from transformers import HiggsAudioV2TokenizerModel

    from echo.eval.metrics import speaker_similarity, transcribe, word_errors

    trim = load_package(package).trim_silence
    codec = HiggsAudioV2TokenizerModel.from_pretrained("eustlb/higgs-audio-v2-tokenizer").cuda().eval()
    references, rows, start = {}, [], time.time()
    with torch.no_grad():
        for n, j in enumerate(jobs, 1):
            codes = torch.from_numpy(np.load(takes / f"{j['key']}.npy").astype(np.int64))[None].cuda()
            wav = codec.decode(codes).audio_values[0, 0].float().cpu().numpy()
            wav = trim(wav[: -int(0.2 * 24000)], 24000)
            hypothesis = transcribe([wav], [j["text_language"]], 24000)[0]
            errors, words = word_errors(j["text"], hypothesis)
            row = dict(prompt=j["id"], seed=j["seed"], language=j["text_language"], errors=errors, words=words,
                       numbers=int(errors > 0 and only_numbers(j["text"], hypothesis)), similarity="", hypothesis=hypothesis)
            if similarity:
                if j["id"] not in references:
                    references[j["id"]] = codec.decode(j["reference"][None].cuda()).audio_values[0, 0].float().cpu().numpy()
                row["similarity"] = speaker_similarity(wav, references[j["id"]], 24000)
            rows.append(row)
            if n % 64 == 0 or n == len(jobs):
                print(f"  transcribed {n}/{len(jobs)} takes, {time.time() - start:.0f}s", flush=True)
    return rows


def wer(rows: List[dict]) -> Dict[str, float]:
    errors, words = sum(int(r["errors"]) for r in rows), sum(int(r["words"]) for r in rows)
    numbers = sum(int(r["errors"]) for r in rows if int(r["numbers"]))
    out = dict(wer=errors / words, wer_no_numbers=(errors - numbers) / words,
               takes_with_error=float(np.mean([int(r["errors"]) > 0 for r in rows])))
    if all(r["similarity"] != "" for r in rows):
        out["similarity"] = float(np.mean([float(r["similarity"]) for r in rows]))
    return out


def sign_p(better: int, worse: int) -> float:
    """
    Two-sided sign test of `better` against `worse` paired takes.
    """

    n, k = better + worse, min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(baseline: List[dict], rows: List[dict]) -> Dict[str, float]:
    """
    Same-seed takes of the model against the baseline's: how many have fewer word errors, how many more.
    """

    before = {(r["prompt"], int(r["seed"])): int(r["errors"]) for r in baseline}
    deltas = [int(r["errors"]) - before[(r["prompt"], int(r["seed"]))] for r in rows]
    better, worse = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
    return dict(better=better, worse=worse, equal=len(deltas) - better - worse, p=sign_p(better, worse),
                **{f"baseline_{k}": v for k, v in wer(baseline).items()})


def critic_pairs(critic: GroupCritic, jobs: List[dict], before: Path, after: Path) -> Dict[str, float]:
    """
    Each test prompt's takes of both models scored as one group: confident same-seed pairs and how many the model
    wins, the paired change in CTC loss, and the CTC-gated takes of each.
    """

    prompts: Dict[str, List[dict]] = {}
    for j in jobs:
        prompts.setdefault(j["id"], []).append(j)
    gated, loss, confident, wins = {"before": [], "after": []}, {"before": [], "after": []}, 0, 0
    for n, (pid, takes) in enumerate(prompts.items(), 1):
        keys = [(m, j["seed"]) for m in ("before", "after") for j in takes]
        codes = [np.load((before if m == "before" else after) / f"{pid}_{s}.npy") for m, s in keys]
        group = critic.score(takes[0], codes, [f"{pid}_{m}_{s}" for m, s in keys])
        for i, (m, _) in enumerate(keys):
            gated[m].append(bool(group["ctc_gated"][i]))
            loss[m].append(group["ctc_loss"][i])
        for i in range(len(takes)):
            a, b = group["score"][i], group["score"][i + len(takes)]
            if abs(b - a) >= critic.margin:
                confident += 1
                wins += b > a
        if n % 20 == 0:
            print(f"  critic: {n}/{len(prompts)} prompts", flush=True)
    b, a = np.array(gated["before"]), np.array(gated["after"])
    d = np.array(loss["after"]) - np.array(loss["before"])
    return dict(confident=confident, wins=int(wins), ctc_change=float(d.mean()), ctc_change_se=float(d.std() / np.sqrt(len(d))),
                gated_baseline=int(b.sum()), gated=int(a.sum()), gated_only_baseline=int((b & ~a).sum()),
                gated_only_model=int((a & ~b).sum()))
