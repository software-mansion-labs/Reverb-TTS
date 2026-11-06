"""Count the text tokens of every gathered sentence corpus, as the model would see them, for vocabulary pruning.

    .venv/bin/python scripts/pruning/count_tokens.py --tokenizer runs/v0.0.2/p2/depth-4-kl-checkpoint --output token_counts.tsv

Reads every <root>/<language>/<corpus>/<corpus>-sentences.txt (Leipzig format: "id<TAB>sentence"). Each sentence is
prepared like a training transcript: characters normalized (echo.preprocessing.characters), digits and symbols
verbalized (echo.preprocessing.verbalize), then tokenized inside TEXT_TEMPLATE. Sentences in a foreign script are
skipped. Tokens holding a digit or a symbol the verbalizer leaves behind are not counted: at inference they come out
of the verbalizer as words.

The output is a TSV sorted by count: id, token, decoded text, count, and the count per language.
"""
import argparse
import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.data.constants import TEXT_TEMPLATE  # noqa: E402
from echo.preprocessing import verbalize  # noqa: E402
from echo.preprocessing.characters import normalize_characters  # noqa: E402

LANGUAGES = {"german": "de", "english": "en", "spanish": "es", "french": "fr", "italian": "it", "polish": "pl", "portuguese": "pt"}
PUNCTUATION = set(".,;:!?\"'()-")   # What the normalized transcripts keep besides letters and spaces
BATCH = 10000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/corpora"), help="Holds one directory per language")
    parser.add_argument("--tokenizer", required=True, help="The model's tokenizer, e.g. its checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="TSV the counts are written to")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    counts = {code: Counter() for code in LANGUAGES.values()}

    for language, code in LANGUAGES.items():
        for path in sorted((args.root / language).glob("*/*-sentences.txt")):
            sentences, skipped = _sentences(path, code)
            for start in range(0, len(sentences), BATCH):
                counts[code].update(_tokens(tokenizer, sentences[start:start + BATCH]))
            print(f"{code}  {path.parent.name}: {len(sentences):,} sentences, {skipped:,} skipped", flush=True)

    special = set(tokenizer.added_tokens_decoder)
    counts = {code: Counter({i: n for i, n in c.items() if i in special or _spoken(tokenizer.decode([i]))}) for code, c in counts.items()}
    total = sum(counts.values(), Counter())
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "token", "text", "count"] + list(counts))
        for token_id, count in total.most_common():
            text = tokenizer.decode([token_id])
            writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id), repr(text), count] + [c[token_id] for c in counts.values()])

    print(f"{sum(total.values()):,} tokens, {len(total):,} distinct of {len(tokenizer):,}, written to {args.output}")
    for code, counter in counts.items():
        if counter:
            print(f"  {code}: {sum(counter.values()):,} tokens, {len(counter):,} distinct")


def _sentences(path: Path, language: str) -> tuple[List[str], int]:
    """The corpus' sentences prepared as transcripts, and how many were skipped."""
    sentences, skipped = [], 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            text = normalize_characters(line.rstrip("\n").split("\t", 1)[-1])
            if _foreign(text):
                skipped += 1
                continue
            try:
                sentences.append(verbalize(text, language))
            except Exception:
                skipped += 1

    return sentences, skipped


def _tokens(tokenizer: PreTrainedTokenizerBase, sentences: List[str]) -> Counter:
    """Tokens of a batch of sentences, as the training data wraps them."""
    counts = Counter()
    for ids in tokenizer([TEXT_TEMPLATE.format(text=text) for text in sentences]).input_ids:
        counts.update(ids)

    return counts


def _spoken(text: str) -> bool:
    """Whether a token is letters, spaces and transcript punctuation only."""
    return all(ch.isspace() or ch in PUNCTUATION or unicodedata.category(ch)[0] in "LM" for ch in text)


def _foreign(text: str) -> bool:
    """Whether the text holds a letter outside the Latin and Greek alphabets."""
    for ch in text:
        if unicodedata.category(ch)[0] in "LM" and not unicodedata.name(ch, "").startswith(("LATIN", "GREEK")) and ch not in "ªº":
            return True
    return False


if __name__ == "__main__":
    main()
