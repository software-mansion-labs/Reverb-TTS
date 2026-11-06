"""
Normalize the characters of every transcript CSV under a dataset root, and drop transcripts that can't be kept.

    .venv/bin/python scripts/normalization/characters.py data/train/english/EARS [--in-place]

Every '|'-separated CSV with a `transcription` column under the root is processed (outputs of an earlier run,
*_normalized.csv and leftovers.py's *_without_leftovers.csv, are skipped). Characters are mapped by echo.preprocessing.characters (ellipsis, quotes,
apostrophes, dashes, spaces). A transcript is dropped when it holds:
    - foreign script: a letter outside the Latin and Greek alphabets (Chinese, Arabic, Hangul, Katakana, ...) or
      CJK punctuation (。、) - the text doesn't match the audio or can't be read by the model;
    - a non-speech annotation: any bracket or parenthesis ((Dzwonek), [[Dzwonek]]) - the sound stays in the audio;
    - a censored word: any asterisk (f**ked) - the audio has the real word.
The stats are shown first and nothing is written until the user answers Y. Without --in-place each CSV is written
next to the original as <name>_normalized.csv; audio and codec archives are left untouched.
"""
import argparse
import collections
import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.preprocessing.characters import RULES  # noqa: E402
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402

SUFFIX = "_normalized"
SKIP = (SUFFIX, "_without_leftovers", "_ctc", "_readings", "_verbalized")  # outputs of the scripts in this directory
EXAMPLES = 3
CLASSES = list(dict.fromkeys(name for name, _, _ in RULES)) + ["edge spaces"]
DROPS = ("foreign script", "annotation", "censored")
WINDOW = 35


def normalize(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """echo.preprocessing.characters.normalize_characters, also returning for each character class that changed
    the text a (before, after) window around its first change."""
    changes = {}
    for name, rx, repl in RULES + [("edge spaces", None, None)]:
        new = text.strip() if rx is None else rx.sub(repl, text)
        if new != text and name not in changes:
            changes[name] = window(text, new)
        text = new
    return text, changes


def window(old: str, new: str) -> tuple[str, str]:
    """The part of old and new around their first difference, with ... where cut."""
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    start = max(i - WINDOW, 0)
    cut = lambda t: ("..." if start else "") + t[start:i + WINDOW] + ("..." if i + WINDOW < len(t) else "")
    return cut(old), cut(new)


def drop_reason(text: str) -> str | None:
    for ch in text:
        name = unicodedata.name(ch, "")
        if ch in "ªº" or name.startswith(("LATIN", "GREEK")):
            continue
        if unicodedata.category(ch)[0] in "LM" or name.startswith(("CJK", "IDEOGRAPHIC", "FULLWIDTH", "HALFWIDTH")):
            return "foreign script"
    if any(ch in text for ch in "()[]{}"):
        return "annotation"
    if "*" in text:
        return "censored"
    return None


def read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        return reader.fieldnames or [], list(reader)


def process(rows: list[dict]) -> tuple[list[dict], collections.Counter, dict[str, list[tuple[str, str]]]]:
    """Normalized rows to keep; counts of changed rows (overall and per character class) and of dropped rows (per
    reason); and examples of each."""
    kept, counts, examples = [], collections.Counter(), collections.defaultdict(list)
    for row in rows:
        new, changes = normalize(row["transcription"])
        reason = drop_reason(new)
        if reason:
            counts[reason] += 1
            examples[reason].append((row["transcription"], ""))
            continue
        counts["changed"] += bool(changes)
        for name, pair in changes.items():
            counts[name] += 1
            examples[name].append(pair)
        kept.append({**row, "transcription": new})
    return kept, counts, examples


def write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    # Written by hand: csv's writer with QUOTE_NONE escapes every '"' in a transcript as '\"'.
    lines = ["|".join(fieldnames)] + ["|".join(row[k] for k in fieldnames) for row in rows]
    if any(line.count("|") != len(fieldnames) - 1 or "\n" in line for line in lines):
        raise ValueError(f"{path}: a field holds '|' or a newline")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def short(text: str, width: int = 110) -> str:
    text = text.strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def share(n: int, rows: int) -> str:
    return f"{n} ({n / max(rows, 1):.2%})"


def report(counts: collections.Counter, rows: int, examples: dict[str, list[tuple[str, str]]] | None) -> None:
    """Changed and dropped rows, broken down by character class and drop reason; with examples when given."""
    dropped = sum(counts[r] for r in DROPS)
    print_info("rows", f"{rows}")
    print_info("changed", share(counts["changed"], rows), Colors.OKBLUE)
    for name in CLASSES:
        if counts[name]:
            print(f"    {Colors.OKCYAN}{name:12s}{Colors.ENDC} {share(counts[name], rows)}")
            for old, new in (examples or {}).get(name, [])[:EXAMPLES]:
                print(f"      {Colors.FAIL}{old!r}{Colors.ENDC}\n      {Colors.OKGREEN}{new!r}{Colors.ENDC}")
    print_info("dropped", share(dropped, rows), Colors.FAIL if dropped else Colors.OKGREEN)
    for reason in DROPS:
        if counts[reason]:
            print(f"    {Colors.WARNING}{reason:14s}{Colors.ENDC} {share(counts[reason], rows)}")
            for old, _ in (examples or {}).get(reason, [])[:EXAMPLES]:
                print(f"      {Colors.FAIL}- {short(old)}{Colors.ENDC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/english/EARS")
    parser.add_argument("--in-place", action="store_true", help="overwrite the CSVs instead of writing *_normalized.csv")
    args = parser.parse_args()

    paths = [p for p in sorted(args.root.rglob("*.csv")) if not p.stem.endswith(SKIP)]
    print_test_title(f"Character normalization: {args.root}")
    results, total = [], collections.Counter()
    for path in paths:
        fieldnames, rows = read(path)
        if "transcription" not in fieldnames:
            continue
        kept, counts, examples = process(rows)
        results.append((path, fieldnames, kept))
        total.update(counts)
        total["rows"] += len(rows)
        print_section(str(path.relative_to(args.root)))
        report(counts, len(rows), examples)

    if not results:
        print(f"{Colors.WARNING}No CSV with a transcription column under {args.root}.{Colors.ENDC}")
        return
    print_section("Total")
    print_info("files", f"{len(results)}")
    report(total, total["rows"], None)
    target = "overwrite the CSVs in place" if args.in_place else f"write *{SUFFIX}.csv next to them"
    answer = input(f"\n{Colors.BOLD}Apply the changes and {target}? [Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for path, fieldnames, kept in results:
        out = path if args.in_place else path.with_name(f"{path.stem}{SUFFIX}.csv")
        write(out, fieldnames, kept)
        print(f"{Colors.OKGREEN}wrote {out}{Colors.ENDC}")


if __name__ == "__main__":
    main()
