"""Rule-based verbalizer: rewrites digits and symbols in a transcript as the words a speaker says.

``verbalize(text, lang)`` runs an ordered list of regex passes. Each pass finds one kind of expression
(a date, a time, an amount of money, a plain number, ...) and asks the language module how to read it.
The passes are shared; a language module only supplies word tables and readings:

Constants
    GROUP, DECIMAL          characters used as group and decimal separators (1,000.5 or 1.000,5)
    POINT, DOT              words joining the parts of a version (3.8.10) and an IP address (10.0.0.51)
    RANGE, BY               words for "-" between numbers and for "x" in dimensions; a string, or a function
                            of (match, *numbers) when the choice depends on context
    SCORE, MINUS            words for ":" in a score (2:1) and for a minus sign
    TIME_SEPS, TIME_SUFFIX  separators between hours and minutes, and a regex of words after a time (Uhr, pm)
    DATE, ORDINAL, DECADE   regexes of numeric dates, ordinals (21st, 3., 1er) and decades (1990s, 90er), or None
    PHONES                  regexes of phone numbers
    CURRENCIES, SCALES      written currencies ($, zł) and scale words (million, Mio., tys.) after amounts
    UNITS                   written units (km, °C); single capital letters (W, V) only count after a space
    PREFIXES, SUFFIXES      abbreviations before a number (Nr., p.) and after one (r.), with their words
    FRACTIONS               (numerator, denominator) -> words; the fraction pass is off when empty
    SYMBOLS                 symbol -> word, or a function of the match for context-dependent ones

Readings, each returning the words, or None to leave the match for a later pass
    read_date(m), read_phone(m), read_fraction(m), read_time(m), read_ordinal(m), read_decade(m)
    read_money(number, scale, currency, m), read_scale(number, scale, m)
    read_percent(number, suffix, m), read_unit(number, unit, m)
    read_compound(m)                 a number joined to a word by a hyphen (21-year-old)
    read_code_digits(digits)         a digit run inside a token with letters (MP3, A380)
    read_number(number, m, zero)     any other number; zero=False drops the "zero" of ".5"
    plural(words), optional          the plural of a reading, for "F-16s"

Alternatives, optional
    GRID                    forced decisions (dicts) for verbalize_spans; while one is in FORCE, the module's
                            readings follow it instead of the context
"""
import difflib
import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from types import ModuleType

LANGUAGES = ("de", "en", "es", "fr", "it", "pl", "pt")
PUNCT = ".,;:!?\"'()[]«»„“”…"

_LETTER = "A-Za-zÀ-ÖØ-öø-ɏ"
_ANNOTATIONS = (
    "Dzwonek|Oklaski|Wesołość na sali|Gwar na sali|Poruszenie na sali|Music|Applause|Laughter|Laughs|inaudible|"
    "Musik|Applaus|Lachen|musique|applaudissements|rires|Música|Aplausos|Risas"
)
_CLEANUP = [
    (re.compile(rf"\s*(?:\[\[[^\]]*\]\]|[(\[](?:{_ANNOTATIONS})[)\]])", re.I), ""),  # non-speech annotations
    (re.compile(r"[()\[\]]"), ""),
    (re.compile("˝"), '"'),
    (re.compile("´"), "'"),
    (re.compile("�"), ""),
]
_DOMAINS = "com|org|net|edu|gov|io|pl|de|fr|es|uk|eu|at|ch|info|tv|co"
FORCE: dict = {}


@dataclass
class Span:
    default: str
    alternatives: list[str]
    original: str


def verbalize(text: str, lang: str) -> str:
    """Rewrites the digits and symbols of a transcript in `lang` (one of LANGUAGES) as spoken words."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}, expected one of {LANGUAGES}")
    for rx, repl in _CLEANUP:
        text = rx.sub(repl, text)
    for rx, read in _passes(lang):
        text = rx.sub(lambda m: _sentence_case(m, read(m)), text)
    return text


def verbalize_spans(text: str, lang: str) -> list[str | Span]:
    """verbalize() as untouched text and replaced spans; a span's alternatives are its readings under the
    module's GRID decisions. A span rewritten by a later pass merges into it (alternatives of the outer one)."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}, expected one of {LANGUAGES}")
    for rx, repl in _CLEANUP:
        text = rx.sub(repl, text)
    grid = importlib.import_module(f"echo.preprocessing.verbalizer.{lang}").__dict__.get("GRID", [])
    spans: list[tuple[int, int, Span]] = []
    for rx, read in _passes(lang):
        edits = []
        for m in rx.finditer(text):
            out = _sentence_case(m, read(m))
            if out == m.group(0):
                continue
            alternatives = []
            for decision in grid:
                FORCE.update(decision)
                try:
                    alt = _sentence_case(m, read(m))
                finally:
                    FORCE.clear()
                if alt not in (out, m.group(0)) and alt not in alternatives:
                    alternatives.append(alt)
            edits.append((m.start(), m.end(), out, alternatives))
        if not edits:
            continue
        shift = lambda p: p + sum(len(o) - (e - s) for s, e, o, _ in edits if e <= p)
        new = text
        for s, e, out, _ in reversed(edits):
            new = new[:s] + out + new[e:]
        kept = []
        for a, b, span in spans:
            if not any(s < b and a < e for s, e, _, _ in edits):
                kept.append((shift(a), shift(b), span))
        for s, e, out, alternatives in edits:
            inner = [(a, b, span) for a, b, span in spans if s < b and a < e]
            a, b = min([s] + [x[0] for x in inner]), max([e] + [x[1] for x in inner])
            original = text[a:b]
            for x, y, span in sorted(inner, reverse=True):
                original = original[: x - a] + span.original + original[y - a:]
            na, nb, ns, ne = shift(a), shift(b), shift(s), shift(s) + len(out)
            alts = [new[na:ns] + alt + new[ne:nb] for alt in alternatives]
            kept.append((na, nb, Span(new[na:nb], alts, original)))
        text, spans = new, sorted(kept, key=lambda x: x[0])
    pieces, pos = [], 0
    for a, b, span in spans:
        pieces += [text[pos:a], span]
        pos = b
    return [p for p in pieces + [text[pos:]] if p != ""]


def realign(old: str, digits: str, lang: str) -> tuple[list[str | Span], int]:
    """An already verbalized transcript as text and spans to test against its audio (see ctc_check.check_spans).

    `digits` is `old` with its number words written as digits (e.g. by text_to_num). Each number of it becomes a
    span whose default is the old words for it, whose alternatives are the verbalizer's readings of it (other than
    the old one) and whose original is the digits. Old and digit text are paired word by word; a number whose old
    words can't be told apart (the pairing is ambiguous, or another number shares them) stays plain text and is
    counted in the second return value.
    """
    parts = verbalize_spans(digits, lang)
    text, ranges = "", []  # the digit text as verbalize_spans read it, and each span's character range in it
    for p in parts:
        piece = p.original if isinstance(p, Span) else p
        if isinstance(p, Span):
            ranges.append((len(text), len(text) + len(piece), p))
        text += piece
    old_words = [m.span() for m in re.finditer(r"\S+", old)]
    new_words = [m.span() for m in re.finditer(r"\S+", text)]
    key = lambda s: s.strip(PUNCT).lower()
    to_old, replaced = {}, set()  # digit-text word -> old word range; digit-text words in changed blocks
    matcher = difflib.SequenceMatcher(None, [key(old[a:b]) for a, b in old_words],
                                      [key(text[a:b]) for a, b in new_words], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for k in range(j1, j2):
            if tag == "equal" or tag == "replace" and i2 - i1 == j2 - j1:  # same word count: paired one to one
                to_old[k] = (i1 + k - j1, i1 + k - j1 + 1)
            elif tag == "replace":
                to_old[k] = (i1, i2)
            if tag == "replace":
                replaced.add(k)
    found, unmapped = [], 0
    for a, b, span in ranges:
        words = [k for k, (s, e) in enumerate(new_words) if s < b and a < e]
        if not span.alternatives or not words:
            continue
        if not replaced & set(words) or any(k not in to_old for k in words):
            unmapped += 1
            continue
        found.append((min(to_old[k][0] for k in words), max(to_old[k][1] for k in words), span))
    clean = [f for f in found if sum(g[0] < f[1] and f[0] < g[1] for g in found) == 1]
    unmapped += len(found) - len(clean)
    out, pos = [], 0
    for i1, i2, span in sorted(clean, key=lambda f: f[0]):
        s, e = old_words[i1][0], old_words[i2 - 1][1]
        s += len(old[s:e]) - len(old[s:e].lstrip(PUNCT))
        e -= len(old[s:e]) - len(old[s:e].rstrip(PUNCT))
        reading = old[s:e]
        alternatives = [x for x in dict.fromkeys([span.default, *span.alternatives]) if letters(x) != letters(reading)]
        out += [old[pos:s], Span(reading, alternatives, span.original)]
        pos = e
    return [p for p in out + [old[pos:]] if p != ""], unmapped


def letters(text: str) -> str:
    """Only the letters of a text, lowercased: for comparing readings regardless of spacing and punctuation."""
    return re.sub(r"[\W\d_]", "", text.lower())


# Helpers for language modules


def before(m: re.Match, k: int = 1) -> str:
    """The k-th word before the match, lowercased and without punctuation."""
    words = m.string[: m.start()].split()
    return words[-k].strip(PUNCT).lower() if len(words) >= k else ""


def after(m: re.Match, k: int = 1) -> str:
    """The k-th word after the match, case kept; '' when punctuation directly follows the match."""
    rest = m.string[m.end():]
    if rest[:1] and rest[:1] in PUNCT:
        return ""
    words = rest.split()
    return words[k - 1].strip(PUNCT) if len(words) >= k else ""


def parse(s: str, group: str, decimal: str) -> tuple[str, str | None, bool]:
    """Splits a written number into (integer digits, fraction digits or None, had group separators)."""
    m = re.fullmatch(rf"(\d{{1,3}}(?:[{_chars(group)}]\d{{3}})+|\d*)(?:[{_chars(decimal)}](\d+))?", s)
    i = re.sub(r"\D", "", m.group(1))
    return i, m.group(2), i != m.group(1)


def spoken_digits(i: str, grouped: bool, longest: int) -> bool:
    """Whether an integer is read digit by digit: a leading zero, or a long code that is not a round amount."""
    return len(i) > 1 and i[0] == "0" or len(i) >= longest and not grouped and not i.endswith("000")


def sentence_dot(m: re.Match, capital: bool = False) -> str:
    """The dot of an abbreviation (Min., r.) when it also ends the text, or the sentence before a capital letter."""
    end = r"\s*(?:$|[A-ZĄĆĘŁŃÓŚŹŻ])" if capital else r"\s*$"
    return "." if m.group(0).endswith(".") and re.match(end, m.string[m.end():]) else ""


# Shared readings


def _sentence_case(m: re.Match, out: str | None) -> str:
    """Keeps the match when its reading declines; capitalizes a reading that starts a sentence."""
    if out is None:
        return m.group(0)
    if out[:1].islower() and not m.group(0)[:1].isalpha() and re.search(r"(?:^|[.!?…]\s+)[¿¡\"«„]?$", m.string[: m.start()]):
        return out[0].upper() + out[1:]
    return out


def _joiner(word: str | Callable, m: re.Match, *numbers: str) -> str:
    return word(m, *numbers) if callable(word) else word


def _ranged(L: ModuleType, m: re.Match, read: Callable[[str], str | None]) -> str | None:
    """Reads "10-15 km" as both numbers with the unit once: the start as a plain number, the end through `read`."""
    head, _, tail = m.group(1).rpartition("-")
    out = read(tail)
    return f"{L.read_number(head, m)}{_joiner(L.RANGE, m, head, tail)}{out}" if head and out is not None else out


def _read_version(L: ModuleType, m: re.Match) -> str | None:
    """3.8.10 or 10.0.0.51; a chain of thousands groups (1.000.000) is left for the number pass."""
    if re.fullmatch(rf"\d{{1,3}}(?:[{_chars(L.GROUP)}]\d{{3}})+", m.group(0)):
        return None
    parts = m.group(0).split(".")
    return f" {L.DOT if len(parts) == 4 else L.POINT} ".join(L.read_number(p, m) for p in parts)


def _read_unit(L: ModuleType, m: re.Match) -> str:
    unit = m.group(2) or m.group(3)
    return _ranged(L, m, lambda x: L.read_unit(re.sub(r"^0+(?=\d)", "", x), unit, m)) + sentence_dot(m)


def _read_code(L: ModuleType, m: re.Match) -> str:
    """A token mixing letters and digits (MP3, COVID-19, L'A380): letters stay, digit runs are read as numbers."""
    out, glue, prev = [], False, ""
    for part in re.findall(r"\d+(?:\.\d+)?|\.\d+|[^\d'’.-]+|['’.-]", m.group(0)):
        if part in "'’" and out:
            out[-1], glue = out[-1] + part, True  # L'A380, TLU2's
            continue
        if part in "-.":
            word = None
        elif part.isdigit():
            word = L.read_code_digits(part)
        elif part[0] == "." and out:
            word = L.read_code_digits(part[1:])  # H.264
        elif part[-1].isdigit():
            word = L.read_number("0" + part if part[0] == "." else part, m, zero=part[0] != ".")
        elif part == "s" and prev.isdigit() and hasattr(L, "plural"):
            out[-1], word = L.plural(out[-1]), None  # F-16s
        else:
            word = part
        if word is not None:
            out[-1:] = [out[-1] + word] if glue else out[-1:] + [word]
        glue, prev = False, part
    return " ".join(out)


def _read_symbols(L: ModuleType, m: re.Match) -> str:
    """Symbols read as words (&, +, =, §); a symbol read as nothing leaves a single space between words."""
    words = [L.SYMBOLS[s] for s in re.findall(_alternation(L.SYMBOLS), m.group(1))]
    words = " ".join(w for w in (w(m) if callable(w) else w for w in words) if w)
    if not words:
        return " " if 0 < m.start() and m.end() < len(m.string) else ""
    right = m.end() < len(m.string) and m.string[m.end()] not in PUNCT
    return (" " if m.start() > 0 else "") + words + (" " if right else "")


def _read_suffix(L: ModuleType, m: re.Match) -> str:
    """An abbreviation after a number (2020 r.); its dot is kept when it also ends the sentence."""
    return f" {L.SUFFIXES[m.group(1)]}{sentence_dot(m, capital=True)}"


def _chars(s: str) -> str:
    return "".join(re.escape(c) for c in s)


def _alternation(keys) -> str:
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


@cache
def _passes(lang: str) -> list[tuple[re.Pattern, Callable[[re.Match], str | None]]]:
    L = importlib.import_module(f"echo.preprocessing.verbalizer.{lang}")
    g, d = _chars(L.GROUP), _chars(L.DECIMAL)
    num = rf"(?:\d{{1,3}}(?:[{g}]\d{{3}})+(?![{g}]?\d)|\d+)(?:[{d}]\d+)?"
    rng = rf"(?:{num}-)?"  # an optional range start: 10-15 km
    start = r"(?<![\w.,])"
    amount = r"(?<![\w.,-])"  # not inside a code like i9-14900K
    hsep = r"|\s?(?P<hsep>h)\s?" if "h" in L.TIME_SEPS else ""
    units = _alternation(k for k in L.UNITS if not re.fullmatch("[A-Z]", k))
    letter_units = _alternation(k for k in L.UNITS if re.fullmatch("[A-Z]", k)) or "(?!)"
    scales, currencies = _alternation(L.SCALES), _alternation(L.CURRENCIES)
    currencies_first = _alternation(k for k in L.CURRENCIES if not k.isalpha())  # $5, €15; but 15 zł

    rules = [
        # Whole expressions whose parts are numbers
        (L.DATE, lambda m: L.read_date(m)),
        *[(p, L.read_phone) for p in L.PHONES],
        (rf"{start}\d+(?:\.\d+){{2,}}(?![\w]|\.\d)", lambda m: _read_version(L, m)),
        (rf"(?<=\w)\.({_DOMAINS})\b", lambda m: f" {L.DOT} {m.group(1)}"),
        # Symbols between numbers
        (r"(?<=\d) - (?=\d)", lambda m: f" {L.MINUS} "),
        (r"(?<=\d)(?:-|\s?[–—]\s?)(?=\d)", lambda m: _joiner(L.RANGE, m)),
        (rf"(?<=\d)(?<!\b0)(\s?(?:{units})?)\s?[x×]\s?(?=\d)", lambda m: f"{m.group(1)} {_joiner(L.BY, m)} "),
        (r"(?<![\w./])(\d+)/(\d+)(?![\w/]|[.,]\d)" if L.FRACTIONS else None, lambda m: L.read_fraction(m)),
        (r"(?<![\w.,])[-−](?=\d)", lambda m: L.MINUS + " "),
        # Times, then any other ":" between numbers is a score
        (rf"{start}(?P<h>\d{{1,2}})(?:(?P<sep>[{_chars(L.TIME_SEPS.replace('h', ''))}]){hsep})(?P<mm>\d{{2}})"
         rf"(?:\s?(?P<suf>{L.TIME_SUFFIX}))?(?![\w]|[.,:]?\d|\s?(?:{units}|min)(?![\w]))", L.read_time),
        (rf"{start}(?P<h>\d{{1,2}})(?P<mm>)\s?(?P<suf>{L.TIME_SUFFIX})(?![\w])", L.read_time),
        (r"(?<=\d):(?=\d)", lambda m: L.SCORE),
        # Abbreviations around numbers
        (rf"(?<![\w.])({_alternation(L.PREFIXES)})\s?(?=\d)", lambda m: L.PREFIXES[m.group(1)] + " "),
        (rf"(?<=\d)\s?({_alternation(L.SUFFIXES)})(?=\s|$|[,;:!?])" if L.SUFFIXES else None, lambda m: _read_suffix(L, m)),
        # Numbers with what they count
        (rf"(?<![\w])({currencies_first})\s?({num})(?:\s?({scales})(?![\w]))?",
         lambda m: L.read_money(m.group(2), m.group(3), m.group(1), m)),
        (rf"{amount}({rng}{num})(?:\s?({scales}))?\s?({currencies})(?![\w])",
         lambda m: _ranged(L, m, lambda x: L.read_money(x, m.group(2), m.group(3), m))),
        (rf"{amount}({rng}{num})\s?({scales})(?![\w])", lambda m: _ranged(L, m, lambda x: L.read_scale(x, m.group(2), m))),
        (rf"{amount}({rng}{num})\s?%(\w*)", lambda m: _ranged(L, m, lambda x: L.read_percent(x, m.group(2), m))),
        (rf"{amount}({rng}{num})(?:\s?({units})|\s({letter_units}))(?![\w'’])", lambda m: _read_unit(L, m)),
        # Numbers marked by a suffix or joined to letters
        (L.ORDINAL, L.read_ordinal),
        (L.DECADE, lambda m: L.read_decade(m)),
        (rf"(?<![\w.,-])({num})-([{_LETTER}]+)", L.read_compound),
        (rf"(?<![\w'’.-])\.?(?=[\w'’.-]*\d)(?=[\w'’.-]*[{_LETTER}])\w(?:[\w'’-]|\.(?=\d))*", lambda m: _read_code(L, m)),
        # Remaining symbols, then remaining numbers
        (rf"\s*((?:{_alternation(L.SYMBOLS)})+)\s*", lambda m: _read_symbols(L, m)),
        (rf"{start}({num})(?![\w])", lambda m: L.read_number(m.group(1), m)),
        (r"(?<![\w.,])\.(\d+)(?![\w])" if "." in L.DECIMAL else None, lambda m: L.read_number("0." + m.group(1), m, zero=False)),
    ]
    return [(re.compile(p), read) for p, read in rules if p]
