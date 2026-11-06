import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

STYLE_TEMPLATE = "<|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>"
TOKENIZE_BATCH = 2000


def codec_path(name: str) -> str:
    """'<dirs>/audio/<file>.wav' -> '<dirs>/codecs/<file>.npy'."""
    audio_dir, _, filename = name.rpartition("/")
    parent, _, leaf = audio_dir.rpartition("/")
    if leaf != "audio":
        raise ValueError(f"expected an 'audio' directory in {name!r}")
    return str(Path(parent) / "codecs" / f"{Path(filename).stem}.npy")


def count_frames(path: Path) -> int:
    # mmap reads only the header; never touches the payload.
    return int(np.load(path, mmap_mode="r").shape[1])


def count_prefix_tokens(
    rows: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase
) -> List[int]:
    """Style + text tokens preceding the audio, one count per row."""
    style_lengths = {
        language: len(tokenizer(STYLE_TEMPLATE.format(language=language)).input_ids)
        for language in {row["language"] for row in rows}
    }
    texts = [f"<|text_start|>{row['transcript']}<|text_end|>" for row in rows]

    prefixes: List[int] = []
    for start in range(0, len(texts), TOKENIZE_BATCH):
        encoded = tokenizer(texts[start : start + TOKENIZE_BATCH]).input_ids
        for offset, ids in enumerate(encoded):
            prefixes.append(len(ids) + style_lengths[rows[start + offset]["language"]])
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dataset .csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Root the names are relative to (default: the input's directory)")
    parser.add_argument("--tokenizer", default="k2-fsa/OmniVoice")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root is not None else args.input.parent

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefixes = count_prefix_tokens(rows, tokenizer)

    fields = list(rows[0]) + ["tokens"]
    missing = 0
    with open(args.output, "w") as f:
        f.write("|".join(fields) + "\n")
        for row, prefix in zip(rows, prefixes):
            row["name"] = codec_path(row["name"])
            path = data_root / row["name"]
            if not path.exists():
                missing += 1
                continue
            row["tokens"] = str(count_frames(path) + prefix)
            f.write("|".join(row[field] for field in fields) + "\n")

    print(f"wrote {len(rows) - missing} rows to {args.output}")
    if missing:
        print(f"skipped {missing} row(s) with no codec file")


if __name__ == "__main__":
    main()
