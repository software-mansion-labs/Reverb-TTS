"""Turns the user's blind rankings (data/rl/preferences/tiers.md) into encoded A/B pairs for evaluating critics.

    cd /workspace/Echo && /venv/main/bin/python scripts/critic/preference_pairs.py --out <dir>

Each ranked take's wav (data/rl/preferences/<voice>/t<n>_<sampler>_seed<k>.wav) is encoded with the Higgs codec on
CUDA, one at a time as the dataset's codecs were, so its codes look like the re-encoded training takes. Every take in
a higher tier beats every take in a lower one; takes left unranked are not used (unranked is not worse), nor are
ranked takes whose wav is missing (some old-sheet targets kept no audio; unpack preferences/audio.zip first). A target's
text comes from data/rl/preferences/texts.csv, or for t1-t3 of a test voice from data/test/test_out.csv; its reference is
the voice's entry in data/rl/speakers_processed.csv (its codec path relative to data/rl/, as there).

Output: <dir>/codes/<target>_<sampler>_seed<k>.npy and <dir>/pairs.csv:
reference|reference_transcript|speaker_id|language|text|text_language|target|better|worse|codes_better|codes_worse|
same_sampler|tiers (e.g. AC: the better take is in tier A, the worse in tier C)
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "normalization"))
from s8_codecs import Codec  # noqa: E402

RL = Path("data/rl")
ROW = re.compile(r"\|\s*([a-z]{2}-[\w-]+-t\d+)\s*\|(.*)\|(.*)\|(.*)\|")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode the user's ranked takes into A/B evaluation pairs.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    texts: dict[str, str] = {}
    for row in read(Path("data/test/test_out.csv")):
        n = sum(key.startswith(row["speaker_id"] + "-t") for key in texts) + 1
        texts[f"{row['speaker_id']}-t{n}"] = row["text"]
    for row in read(RL / "preferences" / "texts.csv"):
        texts.setdefault(row["target"], row["text"])
    references = {}
    for row in read(RL / "speakers_processed.csv"):
        voice = Path(row["name"]).stem if row["name"].startswith("clips/") else row["speaker_id"]
        references[voice] = row

    codec = Codec("cuda")
    (args.out / "codes").mkdir(parents=True, exist_ok=True)
    lines = ["reference|reference_transcript|speaker_id|language|text|text_language|target|better|worse|codes_better|"
             "codes_worse|same_sampler|tiers"]
    missing = 0
    for line in open(RL / "preferences" / "tiers.md", encoding="utf-8"):
        m = ROW.match(line)
        if not m:
            continue
        target, *tiers = m.groups()
        voice, n = target.rsplit("-t", 1)
        tiers = [[take.strip() for take in tier.split(",") if take.strip()] for tier in tiers]
        codes = {}
        for take in (take for tier in tiers for take in tier):
            sampler, seed = take.split(":")
            name, wav = f"{target}_{sampler}_seed{seed}.npy", RL / "preferences" / voice / f"t{n}_{sampler}_seed{seed}.wav"
            if not wav.exists():
                missing += 1
                continue
            if not (args.out / "codes" / name).exists():
                audio, rate = librosa.load(wav, sr=None, mono=True)
                np.save(args.out / "codes" / name, codec.encode(audio, rate))
            codes[take] = f"codes/{name}"
        tiers = [[take for take in tier if take in codes] for tier in tiers]
        ref, language = references[voice], voice[:2]
        for i, high in enumerate(tiers):
            for j, low in enumerate(tiers[i + 1:], i + 1):
                for better in high:
                    for worse in low:
                        same = better.split(":")[0] == worse.split(":")[0]
                        lines.append("|".join([ref["name"], ref["transcript"], ref["speaker_id"], language,
                                               texts[target], language, target, better, worse, codes[better],
                                               codes[worse], str(int(same)), "ABC"[i] + "ABC"[j]]))
        print(f"{target}: {len(codes)} takes", flush=True)
    (args.out / "pairs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE: {len(lines) - 1} pairs ({sum(l.split('|')[-2] == '1' for l in lines[1:])} same-sampler) in {args.out}; "
          f"{missing} ranked takes had no wav", flush=True)


if __name__ == "__main__":
    main()
