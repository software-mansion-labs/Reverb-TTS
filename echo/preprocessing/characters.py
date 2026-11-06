"""Character normalization: one ASCII form per character class, for training transcripts and inference input alike.

    ellipsis      …                 -> ...
    quotes        « » „ “ ” ‟ ˝     -> "
    apostrophes   ‘ ’ ‚ ‛ ´ ` ʼ     -> '
    dashes        — – ‒ ― --        -> - inside a word or number (Szczurek-Żelazko, 1990-2000), " - " between words;
                                       a run of dashes (– —, ----) becomes one
    spaces        non-breaking and other spaces -> " "; runs of spaces -> one; none before , . ! ? ; : (an ellipsis keeps its space)
"""
import re

# (character class, pattern, replacement), applied in order; a final strip removes leading and trailing spaces.
RULES = [
    ("ellipsis", re.compile("…"), "..."),
    ("quotes", re.compile("[«»„“”‟˝]"), '"'),
    ("apostrophes", re.compile("[‘’‚‛´`ʼ]"), "'"),
    ("dashes", re.compile(r"(?<=\w)(?:-{2,}|[—–‒―])(?=\w)"), "-"),
    ("dashes", re.compile(r"\s*(?:-{2,}|[—–‒―])(?:\s*(?:-+|[—–‒―]))*\s*"), " - "),  # a run of dashes is one: – —
    ("spaces", re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000\t]"), " "),
    ("spaces", re.compile(r"[\u200b\u200c\u200d\ufeff]"), ""),
    ("spaces", re.compile(r" {2,}"), " "),
    ("spaces", re.compile(r" +(?=[,!?;:]|\.(?!\.\.))"), ""),  # but " ..." keeps its space
]


def normalize_characters(text: str) -> str:
    for _, rx, repl in RULES:
        text = rx.sub(repl, text)
    return text.strip()
