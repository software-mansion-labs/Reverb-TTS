"""
Text preparation and target length. Verbalis reads numbers, dates, money, units, symbols and abbreviations out as
words, and writes quotes, apostrophes, dashes and ellipses in the forms the training transcripts use.
"""

import functools
import re
import unicodedata
from typing import Dict, List

from verbalis import Verbalizer

LANGUAGES: Dict[str, str] = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pl": "Polish",
    "pt": "Portuguese",
}

BRACKETS = re.compile(r"[()\[\]{}]")  # never in the training transcripts
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def check_language(language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language {language!r}, expected one of: {', '.join(LANGUAGES)}")
    return language


@functools.lru_cache(maxsize=None)
def _verbalizer(language: str) -> Verbalizer:
    return Verbalizer(language)


def prepare_text(text: str, language: str) -> str:
    """
    Text as the model was trained to read it: verbalized, with brackets dropped and single spaces.
    """

    text = _verbalizer(check_language(language)).process(text)
    return " ".join(BRACKETS.sub("", text).split())


def estimate_frames(text: str, reference_text: str, reference_frames: int) -> int:
    """
    The target's length in codec frames: the text's spoken weight, at the rate the reference speaks its transcript.
    """

    def weight(string: str) -> float:
        kinds = (unicodedata.category(char)[0] for char in string)
        return sum({"Z": 0.2, "M": 0.0, "P": 0.5, "S": 0.5, "N": 3.5}.get(kind, 1.0) for kind in kinds)

    frames = weight(text) / (weight(reference_text) / reference_frames)
    if frames < 50:  # short texts get relatively more time
        frames = 50 * (frames / 50) ** (1 / 3)
    return max(1, int(frames))


def split_text(text: str, reference_text: str, reference_frames: int, longest: int, piece: int) -> List[str]:
    """
    The text whole when it is estimated at most `longest` frames, else cut at sentence ends into pieces of about
    `piece` frames. A sentence longer than a piece stays whole.
    """

    if estimate_frames(text, reference_text, reference_frames) <= longest:
        return [text]
    pieces: List[str] = []
    for sentence in SENTENCE_END.split(text):
        joined = f"{pieces[-1]} {sentence}" if pieces else sentence
        if pieces and estimate_frames(joined, reference_text, reference_frames) <= piece:
            pieces[-1] = joined
        else:
            pieces.append(sentence)
    return pieces
