"""English readings for the verbalizer (see core.py for what a language module provides)."""
import re

from echo.preprocessing.verbalizer.core import FORCE, after, before, parse, spoken_digits

GROUP, DECIMAL = ",", "."
POINT, DOT, BY, SCORE, MINUS = "point", "dot", "by", " to ", "minus"
TIME_SEPS = ":."
TIME_SUFFIX = r"a\.m\.|p\.m\.|[ap]m\b\.?|[AP]M\b"
DATE = None
PHONES = [r"(?<![\w-])(?:1-)?\d{3}-\d{3}-\d{4}(?![\w-])", r"(?<![\w-])\d{3}-\d{4}(?![\w-])"]
ORDINAL = r"(?<![\w.,])(\d+)(?:st|nd|rd|th)\b"
DECADE = r"(?<![\w.,-])['’]?(\d+)['’]?s\b"
CURRENCIES = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds"), "¥": ("yen", "yen")}
SCALES = {"thousand": "thousand", "million": "million", "billion": "billion", "trillion": "trillion",
          "k": "thousand", "K": "thousand", "M": "million", "bn": "billion"}
UNITS = {
    "km": ("kilometer", "kilometers"), "m": ("meter", "meters"), "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"), "nm": ("nanometer", "nanometers"), "m²": ("square meter", "square meters"),
    "km²": ("square kilometer", "square kilometers"), "m³": ("cubic meter", "cubic meters"),
    "kg": ("kilogram", "kilograms"), "g": ("gram", "grams"), "mg": ("milligram", "milligrams"),
    "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"), "oz": ("ounce", "ounces"), "ft": ("foot", "feet"),
    "l": ("liter", "liters"), "L": ("liter", "liters"), "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"), "TB": ("terabyte", "terabytes"), "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"), "KB": ("kilobyte", "kilobytes"), "kB": ("kilobyte", "kilobytes"),
    "Hz": ("hertz", "hertz"), "kHz": ("kilohertz", "kilohertz"), "MHz": ("megahertz", "megahertz"),
    "GHz": ("gigahertz", "gigahertz"), "W": ("watt", "watts"), "kW": ("kilowatt", "kilowatts"),
    "kWh": ("kilowatt-hour", "kilowatt-hours"), "V": ("volt", "volts"), "mph": ("mile per hour", "miles per hour"),
    "km/h": ("kilometer per hour", "kilometers per hour"), "°C": ("degree Celsius", "degrees Celsius"),
    "°F": ("degree Fahrenheit", "degrees Fahrenheit"), "°": ("degree", "degrees"),
}
PREFIXES = {"No.": "number", "no.": "number", "Nos.": "numbers", "vol.": "volume", "Vol.": "volume", "p.": "page",
            "pp.": "pages", "ch.": "chapter", "Ch.": "chapter", "fig.": "figure", "Fig.": "figure"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "one half", (1, 3): "one third", (2, 3): "two thirds", (1, 4): "one quarter",
             (3, 4): "three quarters"}
SYMBOLS = {"&": "and", "+": "plus", "=": "equals", "@": "at", "%": "percent", "°": "degrees", "#": "number",
           "²": "squared", "³": "cubed", "π": "pi", "×": "times", "±": "plus or minus", "½": "one half"}
GRID = ([{"style": s} for s in ("cardinal", "year", "hundreds", "digits", "ordinal")]
        + [{"zero": z} for z in ("zero", "oh", "")]
        + [{"and": True}] + [{"style": s, "and": True} for s in ("cardinal", "hundreds")])

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_BIG = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
              "twelve": "twelfth"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_YEAR_CUES = {"in", "of", "since", "from", "by", "until", "till", "circa", "year", "the", "during", "before", "after",
              "early", "late", "mid", *_MONTHS}
_IRREGULAR_PLURALS = {"people", "children", "men", "women", "feet", "persons"}
_TIME_WORDS = {"at", "from", "until", "till", "to", "by", "around", "about", "before", "after", "since", "between",
               "and", "is", "it's", "or"}
# Words after an amount that do not make it an adjective ("$5 a month" vs "a $5 bill", "a 75mm gun").
_STOP = set("""a an the and or but nor in on at to for of per with from by into than more less each every is are was
were be been will would can could should may might must plus minus times anyway now then so that this these those it
if when which who what where why how you i we they he she as up down off out over under about around after before
back away ago versus vs too also only just even still yet already not no there here all both while because since
until though although unless has have had do does did get got instead worth today tomorrow tonight outside inside apart
ahead behind above below long wide high tall deep old later earlier overall""".split())


# Number words


def cardinal(n: int) -> str:
    """With "and" in FORCE, the British form: one hundred and twenty, two thousand and twenty-one."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else "")
    joiner = " and " if FORCE.get("and") else " "
    if n < 1000:
        return f"{ONES[n // 100]} hundred" + (f"{joiner}{cardinal(n % 100)}" if n % 100 else "")
    for value, word in _BIG:
        if n >= value:
            r = n % value
            return f"{cardinal(n // value)} {word}" + (f"{joiner if r < 100 else ' '}{cardinal(r)}" if r else "")


def ordinal(n: int) -> str:
    head, last = re.fullmatch(r"(.*?)([a-z]+)", cardinal(n)).groups()
    return head + _IRREGULAR.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")


def pairs(n: int) -> str:
    """Year-style reading, also used for 3-4 digit codes: nineteen oh five, two thousand seven, nine fifty-six."""
    hi, lo = divmod(n, 100)
    if n >= 1000 and n % 1000 < 10:
        return cardinal(n)
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    return f"{cardinal(hi)} " + (f"oh {ONES[lo]}" if lo < 10 else cardinal(lo))


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


def plural(words: str) -> str:
    if words.endswith("y"):
        return words[:-1] + "ies"
    return words + ("es" if words.endswith("x") else "s")


# Context


def _adjectival(m: re.Match) -> bool:
    """An amount before a noun is an adjective and keeps its unit singular: a $5 bill, a 75mm gun."""
    w = after(m)
    return w[:1].isalpha() and w.lower() not in _STOP


def _counted(m: re.Match) -> bool:
    """A 4-digit number before a plural noun is a count (1999 people), unless a year cue precedes (in 1999)."""
    w = after(m).lower()
    plural_noun = w in _IRREGULAR_PLURALS or len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w not in _STOP
    return plural_noun and before(m) not in _YEAR_CUES


def _forced(i: str, style: str) -> str:
    n = int(i)
    if style == "digits":
        return digits(i)
    if style == "ordinal":
        return ordinal(n)
    if style == "year" and 100 <= n <= 9999:
        return pairs(n)
    if style == "hundreds" and 1000 <= n <= 9999 and n % 1000 >= 100:  # eighteen hundred five
        joiner = " and " if FORCE.get("and") else " "
        return f"{cardinal(n // 100)} hundred" + (f"{joiner}{cardinal(n % 100)}" if n % 100 else "")
    return cardinal(n)


def _integer(i: str, grouped: bool, m: re.Match) -> str:
    n = int(i)
    if "style" in FORCE:
        return _forced(i, FORCE["style"])
    if spoken_digits(i, grouped, 5):
        return digits(i)
    if before(m) in _MONTHS and 1 <= n <= 31:
        return ordinal(n)
    if len(i) == 4 and not grouped:
        return cardinal(n) if _counted(m) else pairs(n)
    return cardinal(n)


def _ampm(s: str) -> str:
    return {"am": "a.m.", "pm": "p.m."}.get(s.rstrip(".").lower(), s) if s.islower() else s


def RANGE(m: re.Match, a: str = "", b: str = "") -> str:
    """10-15 is "ten to fifteen"; scores, codes and abbreviated seasons (51-44, A350-900, 2016-17, 21-22) keep "-"."""
    head, tail = m.string[: m.start()], m.string[m.end():]
    if not a:
        if re.search(r"[A-Za-z]\d*$", head) or re.match(r"\d*[A-Za-z]", tail):
            return "-"
        a, b = re.search(r"\d+$", head).group(0), re.match(r"\d+", tail).group(0)
    season = len(a) == 2 and int(b) == int(a) + 1
    return " to " if len(a) == len(b) and int(b) > int(a) and not season else "-"


# Readings


def read_phone(m: re.Match) -> str:
    groups = m.group(0).split("-")
    if groups[0] == "1":
        groups = [groups[0] + groups[1], *groups[2:]]
    return ", ".join(digits(g) for g in groups)


def read_fraction(m: re.Match) -> str | None:
    return FRACTIONS.get((int(m.group(1)), int(m.group(2))))


def read_time(m: re.Match) -> str | None:
    h, mm, suf = int(m["h"]), m["mm"], m["suf"]
    if not mm:
        return f"{cardinal(h)} {_ampm(suf)}" if h <= 12 and suf[0] in "apAP" else None
    if h > 24 or int(mm) > 59 or m.groupdict().get("sep") == "." and not suf and before(m) not in _TIME_WORDS:
        return None
    words = cardinal(h) + ("" if mm == "00" else f" oh {ONES[int(mm)]}" if mm[0] == "0" else f" {cardinal(int(mm))}")
    return f"{words} {_ampm(suf)}" if suf else words + (" o'clock" if mm == "00" else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    one, many = CURRENCIES[currency]
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is not None and len(f) == 2 and not scale:  # a price: four ninety-nine, ninety-nine cents
        cents = f"oh {ONES[int(f)]}" if f[0] == "0" else cardinal(int(f))
        return f"{cardinal(int(f))} cents" if int(i or 0) == 0 else f"{_integer(i, grouped, m)} {cents}"
    words = read_number(s, m) + (f" {SCALES[scale]}" if scale else "")
    return f"{words} {one if s == '1' and not scale or _adjectival(m) else many}"


def read_scale(s: str, scale: str, m: re.Match) -> str | None:
    return None if scale == "K" else f"{read_number(s, m)} {SCALES[scale]}"  # 4K, 2K: a letter


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return f"{read_number(s, m)} percent" + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    i, f, _ = parse(s, GROUP, DECIMAL)
    return f"{read_number(s, m)} {one if f is None and int(i) == 1 or _adjectival(m) else many}"


def read_ordinal(m: re.Match) -> str:
    return ordinal(int(m.group(1)))


def read_decade(m: re.Match) -> str:
    return plural(read_number(m.group(1), m))


def read_compound(m: re.Match) -> str:
    return f"{read_number(m.group(1), m)}-{m.group(2)}"


def read_code_digits(s: str) -> str:
    if s[0] == "0" and len(s) > 1:
        return " ".join("oh" if c == "0" else ONES[int(c)] for c in s)
    return cardinal(int(s)) if len(s) <= 2 else pairs(int(s)) if len(s) <= 4 else digits(s)


def read_number(s: str, m: re.Match, zero: bool = True) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m)
    head = f"{cardinal(int(i))} " if zero else ""
    if "zero" in FORCE and not int(i or 0):
        head = f"{FORCE['zero']} " if FORCE["zero"] else ""
    return head + ("point oh" if f == "0" else f"point {digits(f)}")
