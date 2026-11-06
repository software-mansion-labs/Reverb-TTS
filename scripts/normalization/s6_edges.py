"""Trim every clip to its transcribed speech and give it exactly EDGE seconds of silence at both ends.

    .venv/bin/python scripts/normalization/edges.py data/train/english/Emilia [--lang en] [--limit 200] [--dry] [--vad-only]

Per clip:
    1. Silero VAD (the YODAS2-sidon segmenter's settings) finds the speech regions;
    2. a CTC model of the language (ctc_check.MODELS) aligns the transcript: its first and last aligned letter;
    3. the cut runs from the start of the first to the end of the last VAD region that overlaps the aligned letters -
       the alignment says which speech the transcript covers, the VAD where that speech exactly begins and ends
       (CTC places a letter on a few frames near its middle, so cutting at the letters would clip onsets and tails);
       VAD regions wholly outside it are untranscribed (mumbling, noise, a jingle) and are cut away;
    4. KEEP_BEFORE / KEEP_AFTER seconds of the original are kept around the cut (FADE seconds faded), and digital
       silence pads each end to EDGE seconds before the speech.
A clip with more than MAX_EDGE_SPEECH seconds of VAD speech outside the cut is dropped from the CSVs: that much is
more likely transcript words missing than a stray sound. A clip that can't be aligned is cut to its VAD regions.

With --vad-only there is no alignment (step 2; about 80% of the time): the cut runs from the first to the last VAD
region, nothing is dropped, and no language is needed - untranscribed sounds at the edges stay.

The language is --lang when given, otherwise each row's language column. Only clips with audio in the portion
(audio/ or audio.zip) are processed; --limit processes a random sample of that many (seed 0). With --dry the
changes are only reported. Otherwise, the new clips are staged in <portion>/.audio_pending and, after the user
answers Y, replace the old ones in place (as 16-bit wav; an mp3 clip becomes <name>.wav; a zip portion's audio.zip
is rebuilt) and dropped clips leave every CSV in place.
The codecs no longer match the audio afterwards and must be re-encoded.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.preprocessing.verbalizer.ctc_check import MODELS, CtcChecker  # noqa: E402
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share, short, write  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import THREADS, Audio, decoded  # noqa: E402

SKIP = ("_normalized", "_without_leftovers", "_ctc", "_readings", "_verbalized")
EDGE = 0.2
KEEP_BEFORE, KEEP_AFTER, FADE = 0.05, 0.10, 0.01
MAX_EDGE_SPEECH = 2.0
VAD_RATE = 16000
_vad = None
EXAMPLES = 6


def speech_regions(audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    """Runs in the decoding workers, each loading its own VAD."""
    global _vad
    if _vad is None:
        _vad = load_silero_vad()
    y = torchaudio.functional.resample(torch.as_tensor(audio, dtype=torch.float32), sample_rate, VAD_RATE)
    found = get_speech_timestamps(y, _vad, sampling_rate=VAD_RATE, threshold=0.5, min_silence_duration_ms=100,
                                  speech_pad_ms=0, return_seconds=True)
    return [(s["start"], s["end"]) for s in found]


def cut(regions: list[tuple[float, float]], span: tuple[float, float] | None, duration: float) -> tuple[float, float]:
    """Start and end of the transcribed speech: the VAD regions overlapping the aligned letters (all regions when
    there is no alignment), never inside the letters."""
    if span is None:
        return (regions[0][0], regions[-1][1]) if regions else (0.0, duration)
    inside = [r for r in regions if r[1] > span[0] and r[0] < span[1]]
    if not inside:
        return span
    return min(inside[0][0], span[0]), max(inside[-1][1], span[1])


def outside(regions: list[tuple[float, float]], start: float, end: float) -> tuple[float, float]:
    """Seconds of VAD speech before start and after end."""
    before = sum(max(0.0, min(b, start) - a) for a, b in regions)
    after = sum(max(0.0, b - max(a, end)) for a, b in regions)
    return before, after


def rebuilt(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """audio[start - KEEP_BEFORE : end + KEEP_AFTER], faded at both ends, padded with zeros to EDGE seconds of
    silence before start and after end."""
    a = max(0, round((start - KEEP_BEFORE) * sample_rate))
    b = min(len(audio), round((end + KEEP_AFTER) * sample_rate))
    y = audio[a:b].astype(np.float32).copy()
    fade = min(round(FADE * sample_rate), len(y) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    lead = max(0, round(EDGE * sample_rate) - (round(start * sample_rate) - a))
    trail = max(0, round(EDGE * sample_rate) - (b - round(end * sample_rate)))
    return np.concatenate([np.zeros(lead, np.float32), y, np.zeros(trail, np.float32)])


def report(rows: int, done: dict[str, dict], examples: bool) -> None:
    ok = [r for r in done.values() if "error" not in r]
    dropped = [r for r in ok if r["drop"]]
    kept = [r for r in ok if not r["drop"]]
    unaligned = [r for r in ok if r["span"] is None and not r["vad_only"]]
    errors = [r for r in done.values() if "error" in r]
    med = lambda xs: f"{np.median(xs):.2f} s" if xs else "-"
    print_info("rows", f"{rows}")
    print_info("processed", f"{len(done)} clips" + (f", {len(unaligned)} cut by VAD only (no alignment)" if unaligned else "")
               + (f", {len(errors)} failed (kept as is)" if errors else ""))
    if ok:
        print(f"    {Colors.OKCYAN}{'silence before':22s}{Colors.ENDC} median {med([r['lead'] for r in ok])} -> {EDGE:.2f} s; "
              f"under 0.1 s: {share(sum(r['lead'] < 0.1 for r in ok), len(ok))}, over 0.5 s: "
              f"{share(sum(r['lead'] > 0.5 for r in ok), len(ok))}")
        print(f"    {Colors.OKCYAN}{'silence after':22s}{Colors.ENDC} median {med([r['trail'] for r in ok])} -> {EDGE:.2f} s; "
              f"under 0.1 s: {share(sum(r['trail'] < 0.1 for r in ok), len(ok))}, over 0.5 s: "
              f"{share(sum(r['trail'] > 0.5 for r in ok), len(ok))}")
        cleaned = [r for r in kept if r["edge_speech"] > 0.05]
        print(f"    {Colors.OKCYAN}{'speech cut at edges':22s}{Colors.ENDC} {share(len(cleaned), len(ok))} "
              f"(untranscribed VAD speech, at most {MAX_EDGE_SPEECH:.1f} s, removed)")
        for r in sorted(cleaned, key=lambda r: -r["edge_speech"])[:EXAMPLES if examples else 0]:
            print(f"      {Colors.WARNING}{r['edge_speech']:.2f} s{Colors.ENDC} {r['name']}: cut {r['start']:.2f}-{r['end']:.2f} "
                  f"of {r['duration']:.2f} s | {short(r['text'], 70)}")
    print_info("dropped", f"{share(len(dropped), len(ok))} (more than {MAX_EDGE_SPEECH:.1f} s of untranscribed speech at an edge)",
               Colors.FAIL if dropped else Colors.OKGREEN)
    for r in dropped[:EXAMPLES if examples else 0]:
        print(f"      {Colors.FAIL}- {r['edge_speech']:.2f} s {r['name']}: words at {r['start']:.2f}-{r['end']:.2f} of "
              f"{r['duration']:.2f} s | {short(r['text'], 70)}{Colors.ENDC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/english/Emilia")
    parser.add_argument("--lang", choices=sorted(MODELS), help="the dataset's language (default: each row's language column)")
    parser.add_argument("--limit", type=int, help="process only this many randomly drawn clips (seed 0)")
    parser.add_argument("--dry", action="store_true", help="only report the changes; write nothing")
    parser.add_argument("--vad-only", action="store_true",
                        help="cut to the first and last VAD region, without the alignment and the drop check")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the first available)")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                             else "cpu")

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
    audio, clips = Audio(), {}
    for path, _, rows in files:
        for row in rows:
            lang = args.lang or row.get("language", "")
            if lang not in MODELS and not args.vad_only:
                sys.exit(f"{path}: row {row['name']} has language {lang!r}; pass --lang (one of {', '.join(sorted(MODELS))})")
            clips.setdefault(row["name"], (path.parent, row["transcription"], lang))
    names = sorted(n for n, (portion, _, _) in clips.items() if n in audio.index(portion))
    with_audio = len(names)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"Edges: {args.root}" + (" (VAD only)" if args.vad_only else "") + (" (dry run)" if args.dry else ""))
    print_info("clips", f"{len(clips)} distinct, {with_audio} with audio (the rest are kept), processing {len(names)}")
    print_info("edges", f"cut to the transcribed speech, {EDGE:.2f} s of silence each side; drop above "
                        f"{MAX_EDGE_SPEECH:.1f} s of untranscribed edge speech")
    torch.set_num_threads(THREADS)
    checkers, done, staged, t0 = {}, {}, 0, time.time()
    for k, (name, result) in enumerate(zip(names, decoded([(clips[n][0], n) for n in names], speech_regions)), 1):
        portion, text, lang = clips[name]
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, regions = result
            if args.vad_only:
                span = None
            else:
                if lang not in checkers:
                    checkers[lang] = CtcChecker(lang, device)
                span = checkers[lang].aligned_span(y, sr, text)
            duration = len(y) / sr
            start, end = cut(regions, span, duration)
            before, after = outside(regions, start, end)
            r = done[name] = {"name": name, "text": text, "span": span, "start": start, "end": end, "duration": duration,
                              "lead": regions[0][0] if regions else start, "trail": duration - (regions[-1][1] if regions else end),
                              "edge_speech": before + after, "drop": max(before, after) > MAX_EDGE_SPEECH,
                              "vad_only": args.vad_only}
            if not r["drop"] and not args.dry:
                audio.write(portion, name, rebuilt(y, sr, start, end), sr)
                staged += 1
        except Exception as error:  # undecodable audio: kept as is, and counted
            done[name] = {"name": name, "text": text, "error": repr(error)}
        if k % 100 == 0 or k == len(names):
            print(f"  {k}/{len(names)} clips, {k / (time.time() - t0):.1f}/s", flush=True)
    print_info("device", device)

    results = []
    for path, fieldnames, rows in files:
        mine = {r["name"]: done[r["name"]] for r in rows if r["name"] in done}
        kept = [r for r in rows if not mine.get(r["name"], {}).get("drop")]
        results.append((path, fieldnames, kept))
        print_section(str(path.relative_to(args.root)))
        report(len(rows), mine, examples=False)
    print_section("Total (distinct clips)")
    report(len(clips), done, examples=True)

    if args.dry:
        print(f"\n{Colors.WARNING}Dry run: nothing written.{Colors.ENDC}")
        return
    answer = input(f"\n{Colors.BOLD}Replace {staged} clips in place and drop the flagged ones from the CSVs? "
                   f"[Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        audio.discard()
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    audio.commit()
    for path, fieldnames, kept in results:
        write(path, fieldnames, kept)
    print(f"{Colors.OKGREEN}replaced {staged} clips; CSVs updated in place{Colors.ENDC}")


if __name__ == "__main__":
    main()
