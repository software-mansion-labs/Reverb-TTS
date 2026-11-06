"""
Drop transcripts with semi-verbalized leftovers, for datasets verbalized upstream (en Emilia, de/fr Emilia-YODAS).

    .venv/bin/python scripts/normalization/leftovers.py data/train/german/Emilia-YODAS [--in-place]

Emilia's normalizer wrote numbers out as words but left their symbols and broke some forms, so in these datasets
any of the following marks a broken transcript:
    - a symbol: $ € £ % § ° ² ³ (zehn%, $four hundred, einhundert Millionen €);
    - any digit (19htrente, 2xone, 1pm);
    - a number word glued to a short letter prefix or followed by a capital (Ltwo, xone, COzwei, CMdeux, Vdeux).
      Only rare tokens count - at most RARE times across the dataset root - since common words match the pattern
      too (Nacht = N + acht, height = h + eight). The language comes from the CSV's language column; only en, de
      and fr have number words here.
Do not run it on datasets whose transcripts keep digits by design (YODAS2-sidon): those get verbalized instead.
Works like characters.py: stats and examples first, nothing written until the user answers Y; without --in-place
each CSV is written next to the original as <name>_without_leftovers.csv.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share, short, write  # noqa: E402

SUFFIX = "_without_leftovers"
SKIP = ("_normalized", SUFFIX, "_ctc", "_readings", "_verbalized")  # outputs of the scripts in this directory
EXAMPLES = 5
RARE = 3
REASONS = ("symbol", "digit", "glued number word")
SYMBOL = re.compile(r"[$€£%§°²³]")
DIGIT = re.compile(r"\d")
NUMBER_WORDS = {
    "en": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million",
    "de": "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf zwanzig dreißig vierzig fünfzig "
          "sechzig siebzig achtzig neunzig hundert tausend",
    "fr": "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize vingt trente "
          "quarante cinquante soixante cent mille",
}
# A token that is a lowercase number word with one to three capitals or a single lowercase letter before it (Ltwo,
# COzwei, xone), or with a capital right after it (zweiGB). Lowercase only: POSIX is not PO + six.
GLUED = {lang: re.compile(rf"(?:[A-Z]{{1,3}}|[a-z])(?:{'|'.join(words.split())})|(?:{'|'.join(words.split())})[A-Z]\w*")
         for lang, words in NUMBER_WORDS.items()}


def tokens(text: str) -> list[str]:
    return [t.strip(".,;:!?\"'()") for t in text.split()]


def leftover(text: str, lang: str, counts: collections.Counter) -> tuple[str, str] | None:
    """(reason, the offending part) of a transcript with a leftover, None otherwise."""
    for reason, rx in (("symbol", SYMBOL), ("digit", DIGIT)):
        m = rx.search(text)
        if m:
            return reason, m.group(0)
    if lang in GLUED:
        for t in tokens(text):
            if counts[t] <= RARE and GLUED[lang].fullmatch(t):
                return "glued number word", t
    return None


def mark(text: str, part: str) -> str:
    """The transcript around the offending part, which is highlighted."""
    i = text.find(part)
    start = max(i - 50, 0)
    snippet = ("..." if start else "") + text[start:i] + f"{Colors.WARNING}{Colors.BOLD}{part}{Colors.ENDC}{Colors.FAIL}"
    return snippet + text[i + len(part):i + len(part) + 50] + ("..." if i + len(part) + 50 < len(text) else "")


def report(counts: collections.Counter, rows: int, examples: dict[str, list[tuple[str, str]]] | None) -> None:
    dropped = sum(counts[r] for r in REASONS)
    print_info("rows", f"{rows}")
    print_info("dropped", share(dropped, rows), Colors.FAIL if dropped else Colors.OKGREEN)
    for reason in REASONS:
        if counts[reason]:
            print(f"    {Colors.WARNING}{reason:18s}{Colors.ENDC} {share(counts[reason], rows)}")
            for text, part in (examples or {}).get(reason, [])[:EXAMPLES]:
                print(f"      {Colors.FAIL}- {mark(text, part)}{Colors.ENDC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/german/Emilia-YODAS")
    parser.add_argument("--in-place", action="store_true",
                        help=f"overwrite the CSVs instead of writing *{SUFFIX}.csv")
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
    # How often each token occurs, over the distinct transcripts of the whole root (filtered and selected CSVs overlap).
    texts = {row["name"]: row["transcription"] for _, _, rows in files for row in rows}
    counts = collections.Counter(t for text in texts.values() for t in tokens(text))

    print_test_title(f"Semi-verbalized leftovers: {args.root}")
    results, total = [], collections.Counter()
    for path, fieldnames, rows in files:
        kept, found, examples = [], collections.Counter(), collections.defaultdict(list)
        for row in rows:
            hit = leftover(row["transcription"], row.get("language", ""), counts)
            if hit:
                found[hit[0]] += 1
                examples[hit[0]].append((row["transcription"].strip(), hit[1]))
            else:
                kept.append(row)
        results.append((path, fieldnames, kept))
        total.update(found)
        total["rows"] += len(rows)
        print_section(str(path.relative_to(args.root)))
        report(found, len(rows), examples)

    print_section("Total")
    print_info("files", f"{len(results)}")
    report(total, total["rows"], None)
    target = "overwrite the CSVs in place" if args.in_place else f"write *{SUFFIX}.csv next to them"
    answer = input(f"\n{Colors.BOLD}Drop these transcripts and {target}? [Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for path, fieldnames, kept in results:
        out = path if args.in_place else path.with_name(f"{path.stem}{SUFFIX}.csv")
        write(out, fieldnames, kept)
        print(f"{Colors.OKGREEN}wrote {out}{Colors.ENDC}")


if __name__ == "__main__":
    main()
