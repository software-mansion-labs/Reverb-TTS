"""Rule-based verbalizer: rewrites digits and symbols in transcripts as spoken words (en, de, fr, es, pl, pt)."""
from echo.preprocessing.verbalizer.core import LANGUAGES, Span, verbalize, verbalize_spans

__all__ = [
    "LANGUAGES",
    "Span",
    "verbalize",
    "verbalize_spans",
]
