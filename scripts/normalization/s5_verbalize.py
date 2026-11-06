"""Verbalize transcripts that keep digits and symbols (YODAS2-sidon portions): numbers become words, each read the
way the audio supports.

    .venv/bin/python scripts/normalization/verbalize.py data/train/polish/YODAS2-sidon [--lang pl] [--limit 200] [--in-place]

Per clip whose transcript the verbalizer changes:
    - with audio in the portion (audio/ or audio.zip): echo.preprocessing.verbalizer.ctc_check.CtcChecker.check -
      verbalize_spans, then every number whose reading has alternatives is scored with a number-aware CTC model and
      takes the alternative that beats the verbalizer's default by more than ctc_check.THRESHOLD per character
      ("1800" said as "eighteen hundred", "0.83" as "point eight three"). As in readings.py, alternatives within
      MIN_DIFFERENCE letters of the default are left out: the model leans to a final "-th" (two thousandth);
    - without audio (the filtered CSV also lists unselected clips): the verbalizer's default reading.
A clip whose verbalized transcript still holds a digit (a malformed source like "1,2.2.") is dropped; a transcript
the verbalizer fails on is kept unchanged and counted.

The language is --lang when given, otherwise each row's language column. Each distinct clip is processed once and
every CSV under the root gets the same transcript for it. --limit processes only a random sample of that many clips
(seed 0); the others are left unchanged. Like the other scripts here, the stats and examples come first and nothing
is written until the user answers Y; without --in-place each CSV is written next to the original as
<name>_verbalized.csv.
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.preprocessing.verbalizer import LANGUAGES, verbalize, verbalize_spans  # noqa: E402
from echo.preprocessing.verbalizer.ctc_check import CtcChecker  # noqa: E402
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share, short, write  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import THREADS, Audio, decoded  # noqa: E402
from scripts.normalization.s4_readings import distinct  # noqa: E402

SUFFIX = "_verbalized"
SKIP = ("_normalized", "_without_leftovers", "_ctc", "_readings", SUFFIX)  # outputs of the scripts in this directory
EXAMPLES = 6
DIGIT = re.compile(r"\d")


def report(rows: int, done: dict[str, dict], examples: bool) -> None:
    changed = [r for r in done.values() if r.get("new") not in (None, r["old"])]
    checked = [r for r in changed if r["checked"]]
    replaced = [r for r in checked if r["replacements"]]
    failed = [r for r in done.values() if "error" in r]
    left = [r for r in changed if DIGIT.search(r["new"])]
    print_info("rows", f"{rows}")
    print_info("verbalized", f"{share(len(changed), len(done))} of processed clips", Colors.OKBLUE)
    print(f"    {Colors.OKCYAN}{'CTC-checked':22s}{Colors.ENDC} {len(checked)} (with audio)")
    print(f"    {Colors.OKCYAN}{'reading replaced':22s}{Colors.ENDC} {share(len(replaced), len(checked))} of checked, "
          f"{sum(len(r['replacements']) for r in replaced)} numbers")
    for r in replaced[:EXAMPLES if examples else 0]:
        print(f"      {Colors.FAIL}{short(r['old'], 100)}{Colors.ENDC}")
        for original, default, reading, score in r["replacements"]:
            print(f"        {original!r}: {Colors.WARNING}{default!r}{Colors.ENDC} -> {Colors.OKGREEN}{reading!r}"
                  f"{Colors.ENDC} {score:+.2f}")
    for r in [r for r in changed if not r["replacements"]][:EXAMPLES if examples else 0]:
        print(f"      {Colors.FAIL}{short(r['old'], 100)}{Colors.ENDC}\n      {Colors.OKGREEN}{short(r['new'], 100)}{Colors.ENDC}")
    if left:
        print(f"    {Colors.FAIL}{'dropped: digits left':22s}{Colors.ENDC} {share(len(left), len(changed))}")
        for r in left[:EXAMPLES if examples else 0]:
            print(f"      {Colors.WARNING}- {short(r['new'], 100)}{Colors.ENDC}")
    if failed:
        print(f"    {Colors.FAIL}{'failed (kept as is)':22s}{Colors.ENDC} {len(failed)}")
        for r in failed[:EXAMPLES if examples else 0]:
            print(f"      {Colors.FAIL}- {short(r['old'], 80)}: {r['error'][:60]}{Colors.ENDC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/polish/YODAS2-sidon")
    parser.add_argument("--lang", choices=LANGUAGES, help="the dataset's language (default: each row's language column)")
    parser.add_argument("--limit", type=int, help="process only this many randomly drawn clips (seed 0)")
    parser.add_argument("--in-place", action="store_true", help=f"overwrite the CSVs instead of writing *{SUFFIX}.csv")
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
            if lang not in LANGUAGES:
                sys.exit(f"{path}: row {row['name']} has language {lang!r}; pass --lang (one of {', '.join(LANGUAGES)})")
            clips.setdefault(row["name"], (path.parent, row["transcription"], lang))
    names = sorted(clips)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"Verbalization: {args.root}")
    print_info("clips", f"{len(clips)} distinct, processing {len(names)}")
    torch.set_num_threads(THREADS)
    checkers, done, t0 = {}, {}, time.time()
    for name in names:
        portion, text, lang = clips[name]
        try:
            done[name] = {"old": text, "new": verbalize(text, lang), "checked": False, "replacements": []}
        except Exception as error:  # a verbalizer failure: the transcript is kept unchanged
            done[name] = {"old": text, "new": None, "checked": False, "replacements": [], "error": repr(error)}
    tested = [n for n in names if done[n]["new"] not in (None, done[n]["old"]) and n in audio.index(clips[n][0])]
    for k, (name, result) in enumerate(zip(tested, decoded([(clips[n][0], n) for n in tested])), 1):
        portion, text, lang = clips[name]
        r = done[name]
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, _ = result
            if lang not in checkers:
                checkers[lang] = CtcChecker(lang, device)
            r["new"], checks = checkers[lang].check_spans(y, sr, distinct(verbalize_spans(text, lang)))
            r["checked"] = True
            r["replacements"] = [(c.span.original, c.span.default, c.reading, c.scores[0][0]) for c in checks
                                 if c.reading != c.span.default]
        except Exception as error:  # undecodable audio: the transcript is kept unchanged
            done[name] = {"old": text, "new": None, "checked": False, "replacements": [], "error": repr(error)}
        if k % 500 == 0 or k == len(tested):
            print(f"  {k}/{len(tested)} clips to check, {k / (time.time() - t0):.1f}/s", flush=True)
    print_info("device", device)

    results = []
    for path, fieldnames, rows in files:
        mine = {r["name"]: done[r["name"]] for r in rows if r["name"] in done}
        new = [{**row, "transcription": mine[row["name"]]["new"] or row["transcription"]} if row["name"] in mine else row
               for row in rows]
        new = [row for row in new if not (row["name"] in mine and DIGIT.search(row["transcription"]))]
        results.append((path, fieldnames, new))
        print_section(str(path.relative_to(args.root)))
        report(len(rows), mine, examples=False)
    print_section("Total (distinct clips)")
    report(len(clips), done, examples=True)

    target = "overwrite the CSVs in place" if args.in_place else f"write *{SUFFIX}.csv next to them"
    note = " Unprocessed clips are left unchanged." if len(names) < len(clips) else ""
    answer = input(f"\n{Colors.BOLD}Write the verbalized transcripts (dropping those with digits left) and {target}?{note} "
                   f"[Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for path, fieldnames, rows in results:
        out = path if args.in_place else path.with_name(f"{path.stem}{SUFFIX}.csv")
        write(out, fieldnames, rows)
        print(f"{Colors.OKGREEN}wrote {out}{Colors.ENDC}")


if __name__ == "__main__":
    main()
