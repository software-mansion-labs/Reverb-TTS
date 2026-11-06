"""Italian readings for the verbalizer (see core.py for what a language module provides).

Numbers are one word up to the millions (duemilaventidue, centottanta, ventitré); "&" is "and", as it only joins
English names (AT&T, R&B).
"""
import re

from echo.preprocessing.verbalizer.core import FORCE, after, before, parse, spoken_digits

GROUP, DECIMAL = ".", ",."
POINT, DOT, BY, SCORE, MINUS, RANGE = "punto", "punto", "per", " a ", "meno", "-"
TIME_SEPS = ":."
TIME_SUFFIX = r"(?!)"
# A numeric date: 24/03/2022.
DATE = r"(?<![\w.,/-])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w]|[.,/]?\d)"
PHONES = []
ORDINAL = r"(?<![\w.,])(\d+)(?:\s?([ºª])|(esim[oaie]))(?![\w])"
DECADE = r"(?<![\w])['’]([1-9]0)(?![\w])"  # anni '80
CURRENCIES = {"€": ("euro", "euro"), "$": ("dollaro", "dollari"), "£": ("sterlina", "sterline")}
SCALES = {"mila": "mila", "k": "mila", "K": "mila", "milione": "milione", "milioni": "milioni", "miliardo": "miliardo",
          "miliardi": "miliardi", "mln": ("milione", "milioni"), "mld": ("miliardo", "miliardi")}
UNITS = {
    "km": ("chilometro", "chilometri"), "m": ("metro", "metri"), "cm": ("centimetro", "centimetri"),
    "mm": ("millimetro", "millimetri"), "m²": ("metro quadro", "metri quadri"), "m2": ("metro quadro", "metri quadri"),
    "km²": ("chilometro quadrato", "chilometri quadrati"), "m³": ("metro cubo", "metri cubi"),
    "kg": ("chilo", "chili"), "g": ("grammo", "grammi"), "mg": ("milligrammo", "milligrammi"),
    "l": ("litro", "litri"), "ml": ("millilitro", "millilitri"), "TB": ("terabyte", "terabyte"), "GB": ("giga", "giga"),
    "MB": ("mega", "mega"), "Hz": ("hertz", "hertz"), "kHz": ("kilohertz", "kilohertz"), "MHz": ("megahertz", "megahertz"),
    "GHz": ("gigahertz", "gigahertz"), "W": ("watt", "watt"), "kW": ("kilowatt", "kilowatt"),
    "kWh": ("kilowattora", "kilowattora"), "V": ("volt", "volt"), "km/h": ("chilometro orario", "chilometri orari"),
    "°C": ("grado", "gradi"), "°": ("grado", "gradi"),
}
PREFIXES = {"n.": "numero", "N.": "numero", "nr.": "numero", "n°": "numero", "N°": "numero", "art.": "articolo",
            "Art.": "articolo", "pag.": "pagina", "cap.": "capitolo", "§": "paragrafo"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "un mezzo", (1, 3): "un terzo", (2, 3): "due terzi", (1, 4): "un quarto", (3, 4): "tre quarti"}
SYMBOLS = {"&": "and", "+": "più", "=": "uguale a", "@": "chiocciola", "%": "per cento", "°": "gradi", "#": "hashtag",
           "²": "al quadrato", "³": "al cubo", "π": "pi greco", "×": "per", "€": "euro", "$": "dollari", "£": "sterline",
           "§": "paragrafo"}
GRID = [{"frac": "digits"}, {"sep": "virgola"}, {"sep": "punto"}, {"time": "plain"}]

ONES = ("zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici quattordici quindici sedici "
        "diciassette diciotto diciannove").split()
TENS = {2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta", 6: "sessanta", 7: "settanta", 8: "ottanta", 9: "novanta"}
_BIG = [(10**9, "un miliardo", "miliardi"), (10**6, "un milione", "milioni")]
MONTHS = "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split()
_ORDINALS = "_ primo secondo terzo quarto quinto sesto settimo ottavo nono decimo".split()
_TIME_CUES = {"alle", "ore", "dalle", "le", "verso", "entro", "per"}


# Number words


def _below100(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, unit = TENS[n // 10], n % 10
    return (tens[:-1] if unit in (1, 8) else tens) + (ONES[unit] if unit else "")  # ventuno, ventotto


def _below1000(n: int) -> str:
    h, r = divmod(n, 100)
    head = "" if h == 0 else "cento" if h == 1 else ONES[h] + "cento"
    if head and (r == 8 or 80 <= r <= 89):
        head = head[:-1]  # centotto, novecentottanta
    return head + (_below100(r) if r or not head else "")


def _accent(words: str) -> str:
    """A compound ending in tre takes the accent: ventitré, centotré, milletré (but tre on its own)."""
    return " ".join(w[:-3] + "tré" if w.endswith("tre") and len(w) > 3 else w for w in words.split(" "))


def cardinal(n: int) -> str:
    """Italian cardinal: one word below a million (duemilaventidue), then un milione duecentomila."""
    words = []
    for value, one, many in _BIG:
        q, n = divmod(n, value)
        if q:
            words.append(one if q == 1 else f"{cardinal(q)} {many}")
    q, n = divmod(n, 1000)
    head = "" if q == 0 else "mille" if q == 1 else _below1000(q) + "mila"
    if head or n or not words:
        words.append(head + (_below1000(n) if n or not head else ""))
    return _accent(" ".join(words))


def ordinal(n: int, end: str = "o") -> str:
    """primo … decimo, then the cardinal with -esimo: undicesimo, ventunesimo, ventitreesimo, millesimo."""
    if n <= 10:
        return _ORDINALS[n][:-1] + end
    w = cardinal(n).replace("tré", "tre")
    if w.endswith(("tre", "sei")):
        return w + "esim" + end
    if w.endswith("mila"):
        return w[:-4] + "millesim" + end
    return w[:-1] + "esim" + end


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


# Readings


def _integer(i: str, grouped: bool, m: re.Match) -> str:
    n = int(i)
    if spoken_digits(i, grouped, 7):
        return digits(i)
    if n == 1 and after(m).lower() in MONTHS:
        return "primo"  # 1 maggio
    return cardinal(n)


def read_date(m: re.Match) -> str | None:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{'primo' if d == 1 else cardinal(d)} {MONTHS[mo - 1]} {cardinal(int(y))}"


def read_phone(m: re.Match) -> str | None:
    return None


def read_fraction(m: re.Match) -> str | None:
    return FRACTIONS.get((int(m.group(1)), int(m.group(2))))


def read_time(m: re.Match) -> str | None:
    """alle 21.30: ventuno e trenta. With a dot it is a time only after alle, ore, dalle... or from 13 h on;
    otherwise a decimal (1.13, versione 2.79)."""
    h, mm = int(m["h"]), m["mm"]
    if not mm or h > 24 or int(mm) > 59:
        return None
    if m["sep"] == "." and before(m) not in _TIME_CUES and not 13 <= h <= 23:
        return None
    hour = "una" if h == 1 else cardinal(h)
    if int(mm) == 0:
        return hour
    return f"{hour} {cardinal(int(mm))}" if FORCE.get("time") == "plain" else f"{hour} e {cardinal(int(mm))}"


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    one, many = CURRENCIES[currency]
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if scale:
        return f"{_scaled(s, scale, m)} di {many}" if SCALES[scale] != "mila" else f"{_scaled(s, scale, m)} {many}"
    if f is not None and len(f) == 2 and s[-3] == ",":  # 10,50 €: dieci euro e cinquanta
        whole = _integer(i or "0", grouped, m)
        return f"{whole} {one if i == '1' else many}" + (f" e {cardinal(int(f))}" if int(f) else "")
    return f"{read_number(s, m)} {one if s == '1' else many}"


def _scaled(s: str, scale: str, m: re.Match) -> str:
    word = SCALES[scale]
    i, f, _ = parse(s, GROUP, DECIMAL)
    if word == "mila" and f is None:
        return cardinal(int(i) * 1000)  # 74 mila: settantaquattromila
    if isinstance(word, tuple):
        word = word[0] if int(i or 0) < 2 and f is None else word[1]
    return f"{read_number(s, m)} {word}"


def read_scale(s: str, scale: str, m: re.Match) -> str | None:
    if scale in "kK" and int(parse(s, GROUP, DECIMAL)[0] or 0) < 10:
        return None  # 4K: a resolution
    return _scaled(s, scale, m)


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return f"{read_number(s, m)} per cento" + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    return f"{read_number(s, m)} {one if s == '1' else many}"


def read_ordinal(m: re.Match) -> str:
    n = int(m.group(1))
    if m.group(3):
        return ordinal(n, m.group(3)[-1])
    return ordinal(n, "a" if m.group(2) == "ª" else "o")


def read_decade(m: re.Match) -> str:
    return cardinal(int(m.group(1)))


def read_compound(m: re.Match) -> str:
    return f"{read_number(m.group(1), m)}-{m.group(2)}"


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) >= 7 else cardinal(int(s))


def read_number(s: str, m: re.Match, zero: bool = True) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m)
    word = FORCE.get("sep") or ("virgola" if s[-len(f) - 1] == "," else "punto")  # the separator as written
    zeros = ["zero"] * (len(f) - len(f.lstrip("0")))
    frac = digits(f) if FORCE.get("frac") == "digits" else " ".join(zeros + ([cardinal(int(f))] if f.strip("0") else []))
    return (f"{cardinal(int(i or 0))} " if zero else "") + f"{word} {frac}"
