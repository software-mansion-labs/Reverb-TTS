"""Clamp every clip's loudness into the test set's range.

    .venv/bin/python scripts/normalization/loudness.py data/train/english/EARS [--limit 200] [--dry]

A clip's integrated loudness (ITU-R BS.1770, torchaudio) is moved by the smallest gain that brings it into
[MIN_LUFS, MAX_LUFS]; a quiet clip is raised only as far as its peak stays under PEAK_CEILING dBFS, so one with a loud
peak may stay below MIN_LUFS. Clips already in range are left untouched. (Unlike scripts/normalize_loudness.py, the
ceiling never lowers a clip: a peak near full scale is no reason to turn down a clip that is already in range.) Run it after edges.py: loudness is gated, so the padding silence doesn't change it.

Only clips with audio in the portion (audio/ or audio.zip) are processed; --limit processes a random sample of
that many (seed 0). With --dry the changes are only reported. Otherwise the new clips are staged in
<portion>/.audio_pending and, after the user answers Y, replace the old ones in place (as 16-bit wav; an mp3 clip
becomes <name>.wav; a zip portion's audio.zip is rebuilt). Nothing is dropped, and the CSVs are not changed. The
codecs no longer match the audio afterwards and must be re-encoded.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchaudio.functional import loudness

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import THREADS, Audio, decoded  # noqa: E402

SKIP = ("_normalized", "_without_leftovers", "_ctc", "_readings", "_verbalized")
MIN_LUFS, MAX_LUFS, PEAK_CEILING = -24.0, -16.0, -1.0
EXAMPLES = 6


def gain(audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """(loudness in LUFS, peak in dBFS, gain in dB): 0 inside the bounds; down to MAX_LUFS above them; up to MIN_LUFS
    below them, but only as far as the peak stays under PEAK_CEILING."""
    lufs = loudness(torch.as_tensor(audio, dtype=torch.float32)[None], sample_rate).item()
    peak = 20 * np.log10(max(float(np.abs(audio).max()), 1e-9))
    if lufs > MAX_LUFS:
        return lufs, peak, MAX_LUFS - lufs
    if lufs < MIN_LUFS:
        return lufs, peak, max(0.0, min(MIN_LUFS - lufs, PEAK_CEILING - peak))
    return lufs, peak, 0.0


def report(rows: int, done: dict[str, dict], examples: bool) -> None:
    ok = [r for r in done.values() if "error" not in r]
    changed = [r for r in ok if abs(r["gain"]) > 0.01]
    capped = [r for r in ok if r["lufs"] + r["gain"] < MIN_LUFS - 0.01]
    errors = [r for r in done.values() if "error" in r]
    print_info("rows", f"{rows}")
    print_info("processed", f"{len(done)} clips" + (f", {len(errors)} failed (kept as is)" if errors else ""))
    if not ok:
        return
    before, after = [r["lufs"] for r in ok], [r["lufs"] + r["gain"] for r in ok]
    for label, xs in (("before", before), ("after", after)):
        print(f"    {Colors.OKCYAN}{label:22s}{Colors.ENDC} median {np.median(xs):.1f} LUFS; below {MIN_LUFS:.0f}: "
              f"{share(sum(x < MIN_LUFS - 0.01 for x in xs), len(xs))}, above {MAX_LUFS:.0f}: "
              f"{share(sum(x > MAX_LUFS + 0.01 for x in xs), len(xs))}")
    print_info("changed", f"{share(len(changed), len(ok))} (gain median "
                          f"{np.median([r['gain'] for r in changed]) if changed else 0:+.1f} dB)", Colors.OKBLUE)
    print(f"    {Colors.WARNING}{'held by peak ceiling':22s}{Colors.ENDC} {share(len(capped), len(ok))} stay below {MIN_LUFS:.0f} "
          f"LUFS (peak at {PEAK_CEILING:.0f} dBFS)")
    for r in sorted(changed, key=lambda r: -abs(r["gain"]))[:EXAMPLES if examples else 0]:
        print(f"      {r['name']}: {r['lufs']:.1f} LUFS, peak {r['peak']:.1f} dBFS -> gain {r['gain']:+.1f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/english/EARS")
    parser.add_argument("--limit", type=int, help="process only this many randomly drawn clips (seed 0)")
    parser.add_argument("--dry", action="store_true", help="only report the changes; write nothing")
    args = parser.parse_args()

    files = []
    for path in sorted(args.root.rglob("*.csv")):
        if path.stem.endswith(SKIP):
            continue
        fieldnames, rows = read(path)
        if "transcription" in fieldnames:
            files.append((path, rows))
    if not files:
        print(f"{Colors.WARNING}No CSV with a transcription column under {args.root}.{Colors.ENDC}")
        return
    audio, clips = Audio(), {}
    for path, rows in files:
        for row in rows:
            clips.setdefault(row["name"], path.parent)
    names = sorted(n for n, portion in clips.items() if n in audio.index(portion))
    with_audio = len(names)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"Loudness: {args.root}" + (" (dry run)" if args.dry else ""))
    print_info("clips", f"{len(clips)} distinct, {with_audio} with audio, processing {len(names)}")
    print_info("bounds", f"{MIN_LUFS:.0f} to {MAX_LUFS:.0f} LUFS, peak at most {PEAK_CEILING:.0f} dBFS")
    torch.set_num_threads(THREADS)
    done, staged, t0 = {}, 0, time.time()
    for k, (name, result) in enumerate(zip(names, decoded([(clips[n], n) for n in names], gain)), 1):
        portion = clips[name]
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, (lufs, peak, g) = result
            done[name] = {"name": name, "lufs": lufs, "peak": peak, "gain": g}
            if abs(g) > 0.01 and not args.dry:
                audio.write(portion, name, y * 10 ** (g / 20), sr)
                staged += 1
        except Exception as error:  # undecodable or too short to measure: kept as is, and counted
            done[name] = {"name": name, "error": repr(error)}
        if k % 500 == 0 or k == len(names):
            print(f"  {k}/{len(names)} clips, {k / (time.time() - t0):.1f}/s", flush=True)

    for path, rows in files:
        print_section(str(path.relative_to(args.root)))
        report(len(rows), {r["name"]: done[r["name"]] for r in rows if r["name"] in done}, examples=False)
    print_section("Total (distinct clips)")
    report(len(clips), done, examples=True)

    if args.dry:
        print(f"\n{Colors.WARNING}Dry run: nothing written.{Colors.ENDC}")
        return
    answer = input(f"\n{Colors.BOLD}Replace {staged} clips in place? [Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        audio.discard()
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    audio.commit()
    print(f"{Colors.OKGREEN}replaced {staged} clips{Colors.ENDC}")


if __name__ == "__main__":
    main()
