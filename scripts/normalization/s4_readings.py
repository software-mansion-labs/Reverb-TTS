"""Drop clips whose upstream verbalization doesn't match the audio (en Emilia, de/fr Emilia-YODAS).

    .venv/bin/python scripts/normalization/readings.py data/train/english/Emilia [--limit 200] [--in-place]

Emilia's normalizer wrote numbers out as words, sometimes not the way they were said ("one hundred and fifty" said
as "one fifty", "twenty twelve" as "two thousand twelve", "zero point five" as "point five"). Per clip:
    1. number words -> digits (ctc_rescoring.to_digits: text_to_num plus its English repairs);
    2. echo.preprocessing.verbalizer.core.realign turns the old transcript into text and spans: each number's old
       words are the reading under test, the verbalizer's readings of its digits the alternatives. Alternatives that
       differ from the old words by MIN_DIFFERENCE letters or fewer are left out: the CTC model can't tell
       sixteen/sixteenth or dritte/dritten apart, and they were most of its false flags;
    3. a number-aware CTC model (ctc_check.MODELS) scores every alternative against the old words
       (ctc_check.check_spans); a clip with any alternative beating them by more than ctc_check.THRESHOLD per
       character is dropped. The text is not rewritten: the best alternative was the right text for only 7 of 11
       English flags judged by listening, while the flag itself was right for 10 of 11.
Run it after the CTC filtering (ctc_rescoring.py for English; de/fr were filtered upstream), since a transcript that
doesn't match the audio makes fine numbers look wrong.

Only clips with audio in the portion (audio/ or audio.zip) and a language of en, de or fr are checked; the others,
and clips outside --limit's random sample (seed 0), are kept. Like the other scripts here, the stats and examples
come first and nothing is written until the user answers Y; without --in-place each CSV is written next to the
original as <name>_readings.csv.
"""
import argparse
import collections
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.preprocessing.verbalizer.core import Span, letters, realign  # noqa: E402
from echo.preprocessing.verbalizer.ctc_check import CtcChecker  # noqa: E402
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share, short, write  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import THREADS, Audio, decoded, to_digits  # noqa: E402

SUFFIX = "_readings"
SKIP = ("_normalized", "_without_leftovers", "_ctc", SUFFIX, "_verbalized")  # outputs of the scripts in this directory
LANGUAGES = ("en", "de", "fr")
MIN_DIFFERENCE = 2
EXAMPLES = 8


def distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def distinct(parts: list[str | Span]) -> list[str | Span]:
    """The spans without alternatives that differ from their default by MIN_DIFFERENCE letters or fewer."""
    return [Span(p.default, [a for a in p.alternatives if distance(letters(a), letters(p.default)) > MIN_DIFFERENCE],
                 p.original) if isinstance(p, Span) else p for p in parts]


def spans_to_test(text: str, lang: str) -> tuple[list[str | Span], int]:
    """The old transcript as text and spans whose alternatives differ enough from the old words; and the count of
    numbers that couldn't be paired with old words."""
    parts, unmapped = realign(text, to_digits(text, lang), lang)
    return distinct(parts), unmapped


def report(rows: int, checked: dict[str, dict], examples: bool) -> None:
    numbers = [r for r in checked.values() if r.get("spans")]
    flagged = [r for r in numbers if r["flags"]]
    unmapped = sum(r.get("unmapped", 0) for r in checked.values())
    errors = [r for r in checked.values() if "error" in r]
    print_info("rows", f"{rows}")
    print_info("checked", f"{len(checked)} clips, {len(numbers)} with a checkable number "
                          f"({sum(r['spans'] for r in numbers)} numbers; {unmapped} not paired with old words)"
               + (f"; {len(errors)} failed to score (kept)" if errors else ""))
    print_info("dropped", f"{share(len(flagged), len(checked))} of checked, {share(len(flagged), len(numbers))} of those "
                          f"with a number", Colors.FAIL if flagged else Colors.OKGREEN)
    for r in flagged[:EXAMPLES if examples else 0]:
        print(f"      {Colors.FAIL}- {short(r['text'], 100)}{Colors.ENDC}")
        for old, best, score in r["flags"]:
            print(f"        {Colors.WARNING}{old!r}{Colors.ENDC} -> {Colors.OKGREEN}{best!r}{Colors.ENDC} {score:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/english/Emilia")
    parser.add_argument("--limit", type=int, help="check only this many randomly drawn clips (seed 0)")
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
            clips.setdefault(row["name"], (path.parent, row["transcription"].strip(), row.get("language", "")))
    names = sorted(n for n, (portion, _, lang) in clips.items() if lang in LANGUAGES and n in audio.index(portion))
    eligible = len(names)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"Old number readings vs audio: {args.root}")
    print_info("clips", f"{len(clips)} distinct, {eligible} with audio in {'/'.join(LANGUAGES)} (the rest are kept), "
                        f"checking {len(names)}")
    torch.set_num_threads(THREADS)
    checkers, checked, parts, t0 = {}, {}, {}, time.time()
    for name in names:
        text = clips[name][1]
        parts[name], unmapped = spans_to_test(text, clips[name][2])
        n = sum(isinstance(p, Span) and bool(p.alternatives) for p in parts[name])
        checked[name] = {"text": text, "spans": n, "unmapped": unmapped, "flags": []}
    tested = [n for n in names if checked[n]["spans"]]
    for k, (name, result) in enumerate(zip(tested, decoded([(clips[n][0], n) for n in tested])), 1):
        portion, text, lang = clips[name]
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, _ = result
            if lang not in checkers:
                checkers[lang] = CtcChecker(lang, device)
            _, results = checkers[lang].check_spans(y, sr, parts[name])
            checked[name]["flags"] = [(c.span.default, c.reading, c.scores[0][0]) for c in results
                                      if c.reading != c.span.default]
        except Exception as error:  # undecodable audio or no alignable text: kept, and counted
            checked[name] = {"text": text, "spans": 0, "error": repr(error), "flags": []}
        if k % 100 == 0 or k == len(tested):
            print(f"  {k}/{len(tested)} clips with numbers, {k / (time.time() - t0):.1f}/s", flush=True)
    print_info("device", device)

    results = []
    for path, fieldnames, rows in files:
        mine = {r["name"]: checked[r["name"]] for r in rows if r["name"] in checked}
        kept = [r for r in rows if not mine.get(r["name"], {}).get("flags")]
        results.append((path, fieldnames, kept))
        print_section(str(path.relative_to(args.root)))
        report(len(rows), mine, examples=False)
    print_section("Total (distinct clips)")
    report(len(clips), checked, examples=True)

    target = "overwrite the CSVs in place" if args.in_place else f"write *{SUFFIX}.csv next to them"
    note = " Unchecked clips are kept." if len(names) < len(clips) else ""
    answer = input(f"\n{Colors.BOLD}Drop the flagged clips and {target}?{note} [Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for path, fieldnames, kept in results:
        out = path if args.in_place else path.with_name(f"{path.stem}{SUFFIX}.csv")
        write(out, fieldnames, kept)
        print(f"{Colors.OKGREEN}wrote {out}{Colors.ENDC}")


if __name__ == "__main__":
    main()
