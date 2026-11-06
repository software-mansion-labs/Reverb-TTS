"""
Re-score upstream-verbalized transcripts (en Emilia) with MMS_FA, numbers as wildcards, and drop the clips that
fail the YODAS2-sidon filter's CTC bounds. (de/fr Emilia-YODAS were already filtered with these bounds.)

    .venv/bin/python scripts/normalization/ctc_rescoring.py data/train/english/Emilia [--limit 200] [--in-place]

Per clip:
    1. number words -> digits: text_to_num.alpha2digit (every number, not only those from 3 up), plus repairs of its
       English quirks (19 90 -> 1990, 19 oh 5 -> 1905, point 8 3 -> .83, 1.5 1000000 -> 1.5 million);
    2. each run of digits (with its separators and suffix: 1990, 2.5, 3., 21st) becomes one "*";
    3. echo.preprocessing.ctc_alignment scores the transcript; a clip with "missing words" or "redundant speech"
       under its BOUNDS (loss > 1.0, or > 0.7 with maxrate > 30; gap_speech > 0.06; trail_speech > 0.05) is dropped.
MMS_FA emits blanks for spoken numbers, so without the wildcard clips with numbers fail far more often (on English
Emilia 32% with number words, 56% as plain digits, against 13% for clips without numbers).

Audio is read from the portion's audio/ directory or its audio.zip, next to the CSV; clips without audio there (the
filtered CSV lists unselected clips) are not scored and are kept. Each distinct clip is scored once, and every CSV under the root (filtered, selected, ...) loses the clips that fail. --limit scores only a random
sample of that many clips (seed 0); the others are kept as they are. Like the other scripts here, the stats and
examples come first and nothing is written until the user answers Y; without --in-place each CSV is written next to
the original as <name>_ctc.csv. The transcripts themselves are not changed.
"""

import argparse
import collections
import io
import multiprocessing
import os
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator

import librosa
import numpy as np
import soundfile as sf
import torch
from text_to_num import alpha2digit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.preprocessing.ctc_alignment import BOUNDS, WILDCARD, CtcAlignment, defects  # noqa: E402
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share, short, write  # noqa: E402

SUFFIX = "_ctc"
SKIP = ("_normalized", "_without_leftovers", SUFFIX, "_readings", "_verbalized")  # outputs of the scripts in this directory
TEXT2NUM = {"en", "de", "fr", "es", "pt", "it"}
EXAMPLES = 5
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")
# torch's default of one thread per core makes its small CPU ops (resampling, loudness) 100-400x slower on a
# many-core box; the steps' main process uses THREADS, each clip-decoding worker one.
THREADS = 8
WORKERS = min(32, os.cpu_count() or 1)
AHEAD = 256  # clips decoded per batch; at most two batches wait in memory


def to_digits(text: str, lang: str) -> str:
    """Number words as digits; the repairs fix how alpha2digit splits English year pairs, decimals and scales."""
    if lang not in TEXT2NUM:
        return text
    t = alpha2digit(text, lang, threshold=0)
    if lang == "en":
        t = re.sub(r"\b(\d{1,2}) oh (\d)\b", r"\1 0\2", t)
        t = re.sub(r"\b(\d{1,2}) (0\d|[1-9]\d)\b", r"\1\2", t)
        t = re.sub(r"\bpoint (\d(?: \d)*)\b", lambda m: "." + m.group(1).replace(" ", ""), t)
        t = re.sub(r"(\d) 1000000000\b", r"\1 billion", t)
        t = re.sub(r"(\d) 1000000\b", r"\1 million", t)
    return t


def wildcarded(text: str) -> str:
    """Each run of digits, with separators and a glued suffix (2.5, 3., 21st, 1er), as one wildcard."""
    t = re.sub(r"[.,]?\d[\d.,:/-]*\w*", f" {WILDCARD} ", text)
    return re.sub(rf"\{WILDCARD}(?:\s*\{WILDCARD})+", WILDCARD, t)


class Audio:
    """Clips' audio from the audio/ directory or audio.zip next to their CSV. A portion ships audio only for its
    selected clips, so the filtered CSV lists clips that have none. Clips can also be replaced (write, commit)."""

    def __init__(self) -> None:
        self._zips: dict[Path, zipfile.ZipFile] = {}
        self._index: dict[Path, dict[str, str]] = {}
        self._pending: dict[Path, set[str]] = {}

    def index(self, portion: Path) -> dict[str, str]:
        """Clip name -> its file in the audio/ directory or the zip."""
        if portion not in self._index:
            if (portion / "audio").is_dir():
                files = [p.name for p in (portion / "audio").iterdir()]
            elif (portion / "audio.zip").exists():
                z = self._zips.setdefault(portion, zipfile.ZipFile(portion / "audio.zip"))
                files = [n.split("/", 1)[1] for n in z.namelist() if n.startswith("audio/") and not n.endswith("/")]
            else:
                files = []
            self._index[portion] = {f.rsplit(".", 1)[0]: f for f in files if f.endswith(AUDIO_EXTENSIONS)}
        return self._index[portion]

    def load(self, portion: Path, name: str):
        file = self.index(portion)[name]
        if (portion / "audio").is_dir():
            return librosa.load(portion / "audio" / file, sr=None, mono=True)
        return librosa.load(io.BytesIO(self._zips[portion].read(f"audio/{file}")), sr=None, mono=True)

    def write(self, portion: Path, name: str, audio: np.ndarray, sample_rate: int) -> None:
        """Stage a replacement for a clip, as a 16-bit wav in <portion>/.audio_pending; commit() puts it in place."""
        pending = portion / ".audio_pending"
        pending.mkdir(exist_ok=True)
        sf.write(pending / f"{name}.wav", audio, sample_rate, subtype="PCM_16")
        self._pending.setdefault(portion, set()).add(name)

    def commit(self) -> None:
        """Put the staged clips in place: into the audio/ directory (an mp3 clip becomes <name>.wav), or into a
        rebuilt audio.zip (written to audio.zip.tmp, then moved over it)."""
        for portion, names in self._pending.items():
            pending = portion / ".audio_pending"
            if (portion / "audio").is_dir():
                for name in names:
                    old = portion / "audio" / self.index(portion)[name]
                    os.replace(pending / f"{name}.wav", portion / "audio" / f"{name}.wav")
                    if old.suffix != ".wav":
                        old.unlink()
            else:
                z = self._zips.pop(portion)
                with zipfile.ZipFile(portion / "audio.zip.tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
                    for info in z.infolist():
                        if not (info.filename.startswith("audio/") and info.filename[6:].rsplit(".", 1)[0] in names):
                            out.writestr(info, z.read(info))
                    for name in sorted(names):
                        out.write(pending / f"{name}.wav", f"audio/{name}.wav")
                z.close()
                os.replace(portion / "audio.zip.tmp", portion / "audio.zip")
            shutil.rmtree(pending)
            self._index.pop(portion, None)
        self._pending.clear()

    def discard(self) -> None:
        """Drop the staged clips."""
        for portion in self._pending:
            shutil.rmtree(portion / ".audio_pending", ignore_errors=True)
        self._pending.clear()


def _start_worker() -> None:
    global _audio
    torch.set_num_threads(1)
    _audio = Audio()


def _decode(job: tuple[Path, str, Callable | None]) -> tuple[np.ndarray, int, Any] | Exception:
    portion, name, work = job
    try:
        y, sr = _audio.load(portion, name)
        return y, sr, work(y, sr) if work else None
    except Exception as error:  # returned, so the caller counts it like its own failures
        return error


def decoded(clips: list[tuple[Path, str]], work: Callable | None = None) -> Iterator[tuple[np.ndarray, int, Any] | Exception]:
    """(audio, sample rate, work(audio, sample rate)) of each (portion, name), in order, or the exception that failed
    it; decoded by WORKERS single-threaded processes. `work` must be a module-level function."""
    with multiprocessing.get_context("spawn").Pool(WORKERS, _start_worker) as pool:
        waiting = None
        for i in range(0, len(clips), AHEAD):
            batch = pool.imap(_decode, [(portion, name, work) for portion, name in clips[i:i + AHEAD]])
            if waiting:
                yield from waiting
            waiting = batch
        if waiting:
            yield from waiting


def report(rows: int, checked: dict[str, dict], examples: bool) -> None:
    """Failures among the checked clips, split by whether the transcript has numbers (wildcards)."""
    with_num = [r for r in checked.values() if "error" not in r and WILDCARD in r["scored"]]
    without = [r for r in checked.values() if "error" not in r and WILDCARD not in r["scored"]]
    errors = [r for r in checked.values() if "error" in r]
    failed = [r for r in with_num + without if r["defects"]]
    print_info("rows", f"{rows}")
    print_info("checked", f"{len(checked)} clips" + (f", {len(errors)} failed to score (kept)" if errors else ""))
    print_info("dropped", share(len(failed), len(with_num) + len(without)), Colors.FAIL if failed else Colors.OKGREEN)
    for label, group in (("with numbers", with_num), ("without numbers", without)):
        bad = [r for r in group if r["defects"]]
        print(f"    {Colors.OKCYAN}{label:16s}{Colors.ENDC} {share(len(bad), len(group))} of {len(group)} fail")
    for defect in ("missing words", "redundant speech"):
        hit = [r for r in failed if defect in r["defects"]]
        if not hit:
            continue
        print(f"    {Colors.WARNING}{defect:16s}{Colors.ENDC} {share(len(hit), len(with_num) + len(without))}")
        for r in hit[:EXAMPLES if examples else 0]:
            s = r["scores"]
            print(f"      {Colors.FAIL}- {short(r['scored'], 100)}{Colors.ENDC}\n        loss {s['loss']:.2f}  "
                  f"maxrate {s['maxrate']:.1f}  gap {s['gap_speech']:.3f}  trail {s['trail_speech']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/english/Emilia")
    parser.add_argument("--limit", type=int, help="score only this many randomly drawn clips (seed 0)")
    parser.add_argument("--in-place", action="store_true", help=f"overwrite the CSVs instead of writing *{SUFFIX}.csv")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the first available)")
    args = parser.parse_args()

    files = []
    for path in sorted(args.root.rglob("*.csv")):
        if path.stem.endswith(SKIP):
            continue
        fieldnames, rows = read(path)
        if "transcription" in fieldnames:
            files.append((path, fieldnames, rows))
    if not files:
        print(f"{Colors.WARNING}No CSV with a transcription column under {args.root}.{Colors.ENDC}")
        return
    # Each distinct clip once (the filtered and selected CSVs of a portion overlap), and only clips with audio.
    audio, clips = Audio(), {}
    for path, _, rows in files:
        for row in rows:
            clips.setdefault(row["name"], (path.parent, row["transcription"], row.get("language", "")))
    names = sorted(n for n, (portion, _, _) in clips.items() if n in audio.index(portion))
    with_audio = len(names)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"CTC rescoring with number wildcards: {args.root}")
    print_info("clips", f"{len(clips)} distinct, {with_audio} with audio (the rest are kept), scoring {len(names)}")
    print_info("bounds", f"{BOUNDS}")
    torch.set_num_threads(THREADS)
    scorer, checked, t0 = CtcAlignment(args.device), {}, time.time()
    for k, (name, result) in enumerate(zip(names, decoded([(clips[n][0], n) for n in names])), 1):
        portion, text, lang = clips[name]
        scored = wildcarded(to_digits(text, lang))
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, _ = result
            scores = scorer.evaluate(y, sr, scored)
            checked[name] = {"scored": scored.strip(), "scores": scores, "defects": defects(scores)}
        except Exception as error:  # undecodable audio or nothing alignable: kept, and counted
            checked[name] = {"scored": scored.strip(), "error": repr(error)}
        if k % 100 == 0 or k == len(names):
            rate = k / (time.time() - t0)
            print(f"  {k}/{len(names)} clips, {rate:.1f}/s", flush=True)
    print_info("device", scorer.device)

    results = []
    for path, fieldnames, rows in files:
        mine = {r["name"]: checked[r["name"]] for r in rows if r["name"] in checked}
        kept = [r for r in rows if not mine.get(r["name"], {}).get("defects")]
        results.append((path, fieldnames, kept))
        print_section(str(path.relative_to(args.root)))
        report(len(rows), mine, examples=False)
    print_section("Total (distinct clips)")
    report(len(clips), checked, examples=True)

    target = "overwrite the CSVs in place" if args.in_place else f"write *{SUFFIX}.csv next to them"
    note = " Unchecked clips are kept." if len(names) < len(clips) else ""
    answer = input(f"\n{Colors.BOLD}Drop the failing clips and {target}?{note} [Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for path, fieldnames, kept in results:
        out = path if args.in_place else path.with_name(f"{path.stem}{SUFFIX}.csv")
        write(out, fieldnames, kept)
        print(f"{Colors.OKGREEN}wrote {out}{Colors.ENDC}")


if __name__ == "__main__":
    main()
