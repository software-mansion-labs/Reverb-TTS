"""German readings for the verbalizer (see core.py for what a language module provides)."""
import re

from echo.preprocessing.verbalizer.core import FORCE, PUNCT, after, before, parse, spoken_digits

GROUP, DECIMAL = ".   ", ",."
POINT, DOT, RANGE, BY, SCORE, MINUS = "Komma", "Punkt", " bis ", "mal", " zu ", "minus"
TIME_SEPS = ":."
TIME_SUFFIX = "Uhr"
DATE = r"(?<![\w.,])(\d{1,2})\.(\s?)(\d{1,2})\.(\d{4}|\d{2})?(?![\w]|[.,]?\d)"
PHONES = [r"(?<![\w.,])0\d{2,4}[ /-]?\d{3,}(?:[ -]\d+)*(?![\w.,]?\d)"]
ORDINAL = r"(?<![\w.,])(\d+)\.(?=\s+\S)"
DECADE = r"(?<![\w.,-])(\d+)er(n|s)?\b"
CURRENCIES = {"€": "Euro", "Euro": "Euro", "$": "Dollar", "Dollar": "Dollar", "£": "Pfund", "CHF": "Franken"}
SCALES = {"Mio.": ("Million", "Millionen"), "Million": ("Million", "Millionen"), "Millionen": ("Million", "Millionen"),
          "Mrd.": ("Milliarde", "Milliarden"), "Milliarde": ("Milliarde", "Milliarden"),
          "Milliarden": ("Milliarde", "Milliarden"), "Tsd.": ("tausend", "tausend"), "k": ("tausend", "tausend")}
UNITS = {
    "km": ("Kilometer", "Kilometer"), "m": ("Meter", "Meter"), "cm": ("Zentimeter", "Zentimeter"),
    "mm": ("Millimeter", "Millimeter"), "nm": ("Nanometer", "Nanometer"), "m²": ("Quadratmeter", "Quadratmeter"),
    "km²": ("Quadratkilometer", "Quadratkilometer"), "m³": ("Kubikmeter", "Kubikmeter"), "ha": ("Hektar", "Hektar"),
    "kg": ("Kilogramm", "Kilogramm"), "g": ("Gramm", "Gramm"), "mg": ("Milligramm", "Milligramm"),
    "t": ("Tonne", "Tonnen"), "l": ("Liter", "Liter"), "ml": ("Milliliter", "Milliliter"),
    "TB": ("Terabyte", "Terabyte"), "GB": ("Gigabyte", "Gigabyte"), "MB": ("Megabyte", "Megabyte"),
    "Hz": ("Hertz", "Hertz"), "kHz": ("Kilohertz", "Kilohertz"), "MHz": ("Megahertz", "Megahertz"),
    "GHz": ("Gigahertz", "Gigahertz"), "W": ("Watt", "Watt"), "kW": ("Kilowatt", "Kilowatt"),
    "kWh": ("Kilowattstunde", "Kilowattstunden"), "V": ("Volt", "Volt"),
    "km/h": ("Kilometer pro Stunde", "Kilometer pro Stunde"), "°C": ("Grad Celsius", "Grad Celsius"),
    "°": ("Grad", "Grad"), "h": ("Stunde", "Stunden"), "Std.": ("Stunde", "Stunden"), "min": ("Minute", "Minuten"),
    "Min.": ("Minute", "Minuten"), "s": ("Sekunde", "Sekunden"), "Sek.": ("Sekunde", "Sekunden"),
    "ct": ("Cent", "Cent"), "PS": ("PS", "PS"),
}
PREFIXES = {"Abs.": "Absatz", "Nr.": "Nummer", "S.": "Seite", "Art.": "Artikel", "Kap.": "Kapitel", "ca.": "circa",
            "Bd.": "Band"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "ein halb", (1, 3): "ein Drittel", (2, 3): "zwei Drittel", (1, 4): "ein Viertel",
             (3, 4): "drei Viertel"}
SYMBOLS = {"&": "und", "€": "Euro", "$": "Dollar", "£": "Pfund", "+": "plus", "=": "gleich", "@": "at", "%": "Prozent",
           "°": "Grad", "#": "Hashtag", "²": "Quadrat", "³": "hoch drei", "¹": "hoch eins", "π": "Pi", "×": "mal",
           "½": "ein halb", "§§": "Paragrafen",
           "§": lambda m: "Paragrafen" if before(m) in ("im", "vom", "zum", "beim", "dem", "des", "den") else "Paragraf"}

ONES = ("null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn "
        "siebzehn achtzehn neunzehn").split()
TENS = "_ _ zwanzig dreißig vierzig fünfzig sechzig siebzig achtzig neunzig".split()
MONTHS = "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split()
_ORDINAL_STEMS = {"eins": "ers", "drei": "drit", "sieben": "sieb", "acht": "ach"}
# Adjective endings of an ordinal: weak after these determiners, mixed after ein-words, strong otherwise.
_WEAK = set("den dem des am im vom zum zur beim ins diesem diesen dieses jedem jeden einen einem einer eines keinen "
            "keinem keiner keines meinen meinem meiner seinen seinem seiner ihren ihrem ihrer unseren unserem unserer".split())
_WEAK_NOM = set("der die das diese dieser jede jeder jedes eine keine meine seine ihre unsere".split())
_MIXED = set("ein kein mein dein sein ihr unser euer".split())
_DATIVE = set("mit nach bei seit von zu aus gegenüber in an auf unter über vor hinter neben zwischen ab".split())
_ACCUSATIVE = set("für durch gegen ohne um bis".split())
_GENITIVE = set("wegen während trotz statt innerhalb außerhalb".split())
_STRONG = {"nom": {"m": "er", "f": "e", "n": "es"}, "acc": {"m": "en", "f": "e", "n": "es"},
           "dat": {"m": "em", "f": "er", "n": "em"}, "gen": {"m": "en", "f": "er", "n": "en"}}
_ONE = {"nom": {"m": "ein", "f": "eine", "n": "ein"}, "acc": {"m": "einen", "f": "eine", "n": "ein"},
        "dat": {"m": "einem", "f": "einer", "n": "einem"}, "gen": {"m": "eines", "f": "einer", "n": "eines"}}
GRID = ([{"style": s} for s in ("cardinal", "year", "digits")]
        + [{"one": w} for w in ("eins", "ein", "eine", "einen", "einem", "einer", "eines")]
        + [{"ending": e} for e in ("e", "en", "er", "es", "em")] + [{"frac": f} for f in ("digits", "number")])
_INFLECTING = _WEAK | _WEAK_NOM | _MIXED | _DATIVE | _ACCUSATIVE | _GENITIVE | {"und", "oder", "bis"}
_DATE_CUES = {"am", "vom", "zum", "bis", "ab", "seit", "den", "der"}
_YEAR_CUES = {"jahr", "jahre", "jahren", "im", "seit", "bis", "von", "ab", "anfang", "ende", "mitte", "sommer", "winter",
              "frühjahr", "herbst", "frühling", "anno", *(m.lower() for m in MONTHS)}
# Nouns that stay singular after a number: 1500 Stück, like plural nouns they make a 4-digit number a count.
_MEASURE_WORDS = {"Stück", "Mal", "Prozent", "Grad", "Euro", "Dollar", "Pfund", "Kilo", "Gramm", "Paar", "Cent", "Franken"}
# Noun gender by ending, with the exceptions that matter after numbers (matched as compound endings).
_FEMININE = ("e", "ung", "heit", "keit", "schaft", "ion", "tät", "ik", "ur", "enz", "anz", "ei", "in", "ie")
_NEUTER = ("chen", "lein", "ment", "um", "tum", "ma", "o")
_GENDER = {"person": "f", "mal": "n", "jahr": "n", "kind": "n", "prozent": "n", "kilo": "n", "gramm": "n", "stück": "n",
           "ei": "n", "tor": "n", "glas": "n", "haus": "n", "spiel": "n", "land": "n", "buch": "n", "kapitel": "n",
           "jahrhundert": "n", "jahrtausend": "n", "mitglied": "n", "bild": "n", "quartal": "n", "semester": "n",
           "rennen": "n", "ende": "n", "leben": "n", "zimmer": "n", "pfund": "n", "euro": "m", "name": "m", "käse": "m",
           "kunde": "m", "junge": "m", "grad": "m", "meter": "m", "liter": "m", "monat": "m"}


# Number words


def cardinal(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return (f"{stem(u)}und" if u else "") + TENS[t]
    if n < 1000:
        return f"{stem(n // 100)}hundert" + (cardinal(n % 100) if n % 100 else "")
    if n < 10**6:
        return f"{stem(n // 1000)}tausend" + (cardinal(n % 1000) if n % 1000 else "")
    for value, one, many in ((10**9, "Milliarde", "Milliarden"), (10**6, "Million", "Millionen")):
        if n >= value:
            q, r = divmod(n, value)
            return (f"eine {one}" if q == 1 else f"{cardinal(q)} {many}") + (f" {cardinal(r)}" if r else "")


def stem(n: int) -> str:
    """Cardinal as the first part of a compound: ein, einundzwanzig, einhundertein."""
    return re.sub("eins$", "ein", cardinal(n))


def year(n: int) -> str:
    """1100-1999 in hundreds style (neunzehnhundertneunzig); later years read like cardinals."""
    return f"{cardinal(n // 100)}hundert" + (cardinal(n % 100) if n % 100 else "") if 1100 <= n <= 1999 else cardinal(n)


def ordinal(n: int, ending: str) -> str:
    words, r = cardinal(n), n % 100
    if 0 < r < 20:
        return words[: len(words) - len(ONES[r])] + _ORDINAL_STEMS.get(ONES[r], ONES[r]) + "t" + ending
    return words + "st" + ending


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


# Context


def _gender(noun: str) -> str:
    w = noun.lower()
    for key in sorted(_GENDER, key=len, reverse=True):
        if w.endswith(key):
            return _GENDER[key]
    return "f" if w.endswith(_FEMININE) else "n" if w.endswith(_NEUTER) else "m"


def _case(word: str) -> str:
    return "dat" if word in _DATIVE else "acc" if word in _ACCUSATIVE else "gen" if word in _GENITIVE else "nom"


def _one(m: re.Match, noun: str) -> str:
    """"1" before a noun, inflected like the indefinite article: eine Stunde, mit einer Stimme."""
    if "one" in FORCE:
        return FORCE["one"]
    return _ONE[_case(before(m))][_gender(noun)]


def _labelled(m: re.Match) -> bool:
    """A number right after a noun is a label: Seite 1, Platz 1, Grafik 1."""
    head = m.string[: m.start()].split()
    return bool(head) and head[-1][:1].isupper() and not (len(head) == 1 or head[-2][-1:] in ".!?")


def _ending(m: re.Match, noun: str) -> str:
    """The adjective ending of an ordinal from the words before it: am dritten, der dritte, ein zweiter."""
    if "ending" in FORCE:
        return FORCE["ending"]
    prev = [w.strip(PUNCT).lower() for w in m.string[: m.start()].split()]
    while len(prev) >= 2 and prev[-1] in ("und", "oder", "bis", "sowie") and prev[-2].isdigit():
        prev = prev[:-2]  # "in 2. und 3. Lesung", "vom 1. 5. bis 3. 5.": both take the ending of the first
        while prev and prev[-1].isdigit():
            prev = prev[:-1]
    w1, w2 = (["", ""] + prev)[-1], (["", ""] + prev)[-2]
    if w1 in _WEAK or w1 == "der" and w2 in _DATIVE | _GENITIVE:
        return "en"
    if w1 in _WEAK_NOM:
        return "e"
    g = _gender(noun)
    if w1 in _MIXED:
        return "es" if g == "n" else "er"
    return _STRONG[_case(w1)][g]


def _integer(i: str, grouped: bool, m: re.Match) -> str:
    n, nxt = int(i), after(m)
    if "style" in FORCE:
        return {"cardinal": cardinal, "year": year, "digits": lambda _: digits(i)}[FORCE["style"]](n)
    if n == 1 and "one" in FORCE:
        return FORCE["one"]
    if spoken_digits(i, grouped, 5):
        return digits(i)
    if n == 1:
        if nxt in ("bis", "oder") and after(m, 2).isdigit():  # "1 bis 2 Werktage": ein, gendered by the plural noun
            nxt = re.sub(r"(?<=e)n$|e$", "", after(m, 3))
        if nxt[:1].isupper() and nxt != "Quadrat" and (nxt != after(m) or not _labelled(m)):
            return _one(m, nxt)
    counted = nxt[:1].isupper() and (nxt.endswith(("e", "n", "er", "s")) or nxt in _MEASURE_WORDS) and nxt not in MONTHS
    if len(i) == 4 and not grouped and "mal" not in (nxt, before(m)) and (before(m) in _YEAR_CUES or not counted):
        return year(n)  # 1998 Abi gemacht, im Jahr 1945; but 1945 Einwohner, 1920 mal 1080
    return cardinal(n)


# Readings


def read_date(m: re.Match) -> str | None:
    d, mo = int(m.group(1)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12) or m.group(2) and before(m) not in _DATE_CUES:
        return None  # "am 12. 3." is a date, "Platz 1. 2 Tore" is not
    words = f"{ordinal(d, _ending(m, 'Monat'))} {MONTHS[mo - 1]}"
    return words + (f" {year(int(m.group(4)))}" if m.group(4) else "")


def read_phone(m: re.Match) -> str:
    return digits(re.sub(r"\D", "", m.group(0)))


def read_fraction(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    return FRACTIONS.get((a, b), f"{cardinal(a)} durch {cardinal(b)}")


def read_time(m: re.Match) -> str | None:
    h, mm = int(m["h"]), m["mm"]
    if mm and (h > 24 or int(mm) > 59 or m.groupdict().get("sep") == "." and not m["suf"]):
        return None
    return f"{stem(h)} Uhr" + (f" {cardinal(int(mm))}" if mm and int(mm) else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    noun = CURRENCIES[currency]
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if scale:
        return f"{read_scale(s, scale, m)} {noun}"
    if f is not None and len(f) == 2:  # a price: drei Euro fünfzig, neunundneunzig Cent
        if int(i or 0) == 0:
            return f"{cardinal(int(f))} Cent"
        return f"{_one(m, noun) if i == '1' else _integer(i, grouped, m)} {noun} {cardinal(int(f))}"
    return f"{_one(m, noun) if s == '1' else read_number(s, m)} {noun}"


def read_scale(s: str, scale: str, m: re.Match) -> str:
    one, many = SCALES[scale]
    if many == "tausend":
        return read_number(s, m) + "tausend"
    return f"eine {one}" if s == "1" else f"{read_number(s, m)} {many}"


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    words = read_number(s, m)
    return f"{words}prozent{suffix}" if suffix else f"{words} Prozent"  # 8%ige: achtprozentige


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    return f"{_one(m, one)} {one}" if s == "1" else f"{read_number(s, m)} {many}"


def read_ordinal(m: re.Match) -> str | None:
    words = m.string[m.end():].split()
    noun = next((w.strip(".,;:!?") for w in words if w[:1].isupper()), "")
    if words[0][:1].isdigit() or words[0][:1].isupper() and words[0] not in MONTHS and before(m) not in _INFLECTING:
        return None  # "Platz 3. Danach", "Platz 1. 2 Tore": a sentence end, not an ordinal
    return ordinal(int(m.group(1)), _ending(m, noun))


def read_decade(m: re.Match) -> str:
    i = m.group(1)
    words = ("null" + cardinal(int(i)) if i[0] == "0" and len(i) > 1 else year(int(i))) + "er" + (m.group(2) or "")
    return words if after(m) in ("Jahre", "Jahren", "Jahrgang") else words[0].upper() + words[1:]


def read_compound(m: re.Match) -> str:
    """30-jährige: dreißigjährige; 4-Zimmer-Wohnung: Vier-Zimmer-Wohnung."""
    words, suffix = stem(int(m.group(1))) if m.group(1).isdigit() else read_number(m.group(1), m), m.group(2)
    return f"{words[0].upper()}{words[1:]}-{suffix}" if suffix[0].isupper() else words + suffix


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) > 4 else year(int(s))


def read_number(s: str, m: re.Match, zero: bool = True) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m)
    frac = digits(f)
    if FORCE.get("frac") == "number":  # Komma fünfundzwanzig
        frac = " ".join(["null"] * (len(f) - len(f.lstrip("0"))) + ([cardinal(int(f))] if f.strip("0") else []))
    return (f"{cardinal(int(i))} " if zero else "") + f"Komma {frac}"
