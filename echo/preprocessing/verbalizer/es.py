"""Spanish readings for the verbalizer (see core.py for what a language module provides)."""
import re

from echo.preprocessing.verbalizer.core import FORCE, after, before, parse, spoken_digits

GROUP, DECIMAL = ".,", ".,"
POINT, DOT, RANGE, BY, SCORE, MINUS = "punto", "punto", "-", "por", " a ", "menos"
TIME_SEPS = ":."
TIME_SUFFIX = r"h\b|hrs\b"
_LAWS = ("ley|decreto|legislativo|sentencia|resolución|declaración|orden|acuerdo|circular|reglamento|directiva|"
         "constitucionalidad|expediente|norma|instrucción")
# A law number with its year (Ley 2/2018), or a numeric date (14/07/2023).
DATE = (rf"(?i:\b({_LAWS}))\s+(\d{{1,3}})[./-]((?:19|20)\d\d)(?![\w]|[.,/]?\d)"
        r"|(?<![\w.,/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w]|[.,/]?\d)")
PHONES = [r"(?<![\w.,-])\d{3}-\d{3}-\d{3}(?![\w-]|[.,]?\d)"]
ORDINAL = r"(?<![\w.,])(\d+)(?:\s?([ºª])|(er|ero|era|ro|ra|do|da|to|ta|vo|va|mo|ma|no|na)\b)"
DECADE = r"(?<![\w.,-])(\d+)'?s\b"
CURRENCIES = {"$": ("dólar", "dólares"), "US$": ("dólar", "dólares"), "€": ("euro", "euros"), "£": ("libra", "libras")}
SCALES = {"millones": "millones", "millón": "millón", "mil": "mil", "k": "mil", "K": "mil"}
UNITS = {
    "km": ("kilómetro", "kilómetros"), "m": ("metro", "metros"), "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"), "nm": ("nanómetro", "nanómetros"), "m²": ("metro cuadrado", "metros cuadrados"),
    "km²": ("kilómetro cuadrado", "kilómetros cuadrados"), "m³": ("metro cúbico", "metros cúbicos"),
    "ha": ("hectárea", "hectáreas"), "kg": ("kilogramo", "kilogramos"), "g": ("gramo", "gramos"),
    "mg": ("miligramo", "miligramos"), "t": ("tonelada", "toneladas"), "l": ("litro", "litros"), "L": ("litro", "litros"),
    "ml": ("mililitro", "mililitros"), "TB": ("terabyte", "terabytes"), "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"), "KB": ("kilobyte", "kilobytes"), "Hz": ("hercio", "hercios"),
    "kHz": ("kilohercio", "kilohercios"), "MHz": ("megahercio", "megahercios"), "GHz": ("gigahercio", "gigahercios"),
    "W": ("vatio", "vatios"), "kW": ("kilovatio", "kilovatios"), "V": ("voltio", "voltios"),
    "km/h": ("kilómetro por hora", "kilómetros por hora"), "MPH": ("milla por hora", "millas por hora"),
    "mph": ("milla por hora", "millas por hora"), "°C": ("grado Celsius", "grados Celsius"), "°": ("grado", "grados"),
}
PREFIXES = {"Nº": "número", "nº": "número", "N°": "número", "núm.": "número", "pág.": "página", "art.": "artículo"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "un medio", (1, 3): "un tercio", (2, 3): "dos tercios", (1, 4): "un cuarto", (3, 4): "tres cuartos"}
SYMBOLS = {"&": "y", "=": "igual a", "@": "arroba", "%": "por ciento", "°": "grados", "#": "numeral", "²": "al cuadrado",
           "³": "al cubo", "π": "pi", "×": "por", "€": "euros", "$": "dólares", "£": "libras",
           "+": lambda m: "plus" if re.search(r"[^\W\d_]{2,}$", m.string[: m.start()]) else "más"}  # Google+, but C++
GRID = ([{"g": g} for g in ("m", "a", "f")] + [{"g": g, "mode": "ord"} for g in ("m", "f")]
        + [{"currency": c} for c in (("dólar", "dólares"), ("peso", "pesos"))])

UNITS_WORDS = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis "
               "diecisiete dieciocho diecinueve veinte veintiuno veintidós veintitrés veinticuatro veinticinco "
               "veintiséis veintisiete veintiocho veintinueve").split()
TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
HUNDREDS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos", 6: "seiscientos",
            7: "setecientos", 8: "ochocientos", 9: "novecientos"}
MONTHS = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
_ORD_UNITS = "_ primer segund tercer cuart quint sext séptim octav noven".split()
_ORD_TENS = "_ décim vigésim trigésim cuadragésim quincuagésim sexagésim septuagésim octogésim nonagésim".split()
_ORD_HUNDREDS = ("_ centésim ducentésim tricentésim cuadringentésim quingentésim sexcentésim septingentésim "
                 "octingentésim noningentésim").split()
# Nouns in -a/-ión/-dad/... are feminine unless listed; other feminine nouns are listed.
_MASCULINE_A = {"día", "mapa", "problema", "sistema", "programa", "tema", "idioma", "clima", "planeta", "poema",
                "dilema", "drama", "esquema", "sofá", "papá", "cometa", "fantasma", "panorama", "diagrama", "teorema"}
_FEMININE = {"vez", "veces", "mano", "foto", "moto", "radio", "mujer", "flor", "noche", "clase", "parte", "gente",
             "leche", "muerte", "suerte", "calle", "carne", "red", "voz", "luz", "paz", "ley", "piel", "fase", "base",
             "frase", "nube", "llave", "fuente", "mente", "torre", "tarde", "serie", "especie", "superficie", "imagen",
             "orden", "edición"}
_FEMININE_ENDINGS = ("a", "ión", "dad", "tad", "tud", "umbre", "sis", "itis")
# Words after a number that are not the noun it counts.
_STOP = set("""de del en con por para sin que y e o u a al los las el la lo un una unos unas es son era sea pero como más menos
hasta desde entre sobre contra hacia tras según durante mediante cada toda todas todo todos esta estas este estos esa esas
ese esos otra otras otro otros nada ya ahora cuando donde mientras muy casi también así se le les me te nos""".split())


# Number words


def _below100(n: int, g: str) -> str:
    """g: "m" standalone (uno), "a" before a masculine noun (un, veintiún), "f" feminine (una)."""
    w = UNITS_WORDS[n] if n < 30 else TENS[n // 10] + (f" y {UNITS_WORDS[n % 10]}" if n % 10 else "")
    if w.endswith("uno"):
        stem = w[:-3]
        w = stem + {"f": "una", "a": "ún" if stem == "veinti" else "un"}.get(g, "uno")
    return w


def _below1000(n: int, g: str) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below100(r, g)
    if n == 100:
        return "cien"
    head = HUNDREDS[h][:-2] + "as" if g == "f" and h > 1 else HUNDREDS[h]
    return head + (f" {_below100(r, g)}" if r else "")


def cardinal(n: int, g: str = "m") -> str:
    """Spanish cardinal in gender form `g` (see _below100): doscientas mil personas, veintiún años."""
    if n < 1000:
        return _below1000(n, g)
    if n < 10**6:
        t, r = divmod(n, 1000)
        return ("mil" if t == 1 else f"{_below1000(t, 'f' if g == 'f' else 'a')} mil") + (f" {_below1000(r, g)}" if r else "")
    if n < 10**12:
        q, r = divmod(n, 10**6)
        return ("un millón" if q == 1 else f"{cardinal(q, 'a')} millones") + (f" {cardinal(r, g)}" if r else "")
    q, r = divmod(n, 10**12)
    return ("un billón" if q == 1 else f"{cardinal(q, 'a')} billones") + (f" {cardinal(r, g)}" if r else "")


def ordinal(n: int, fem: bool) -> str:
    """centésima trigésima séptima"""
    return " ".join(words[n // value % 10] + ("a" if fem else "o")
                    for value, words in ((100, _ORD_HUNDREDS), (10, _ORD_TENS), (1, _ORD_UNITS)) if n // value % 10)


def digits(s: str) -> str:
    return " ".join(UNITS_WORDS[int(c)] for c in s)


# Context


def _feminine(noun: str) -> bool:
    w = noun.lower()
    sing = w[:-5] + "ión" if w.endswith("iones") else w[:-1] if w.endswith(("as", "es")) else w
    return w in _FEMININE or sing in _FEMININE or sing not in _MASCULINE_A and sing.endswith(_FEMININE_ENDINGS)


def _gender(m: re.Match) -> str:
    """The gender form of a number from the word after it."""
    w = after(m)
    if before(m) in ("la", "las"):  # a la 1, las 21: the hour is feminine
        return "f"
    if not w[:1].isalpha() or w.lower() in _STOP:
        return "m"
    return "f" if _feminine(w) else "a"


def _counted(m: re.Match) -> bool:
    w = after(m)
    return len(w) > 2 and w.isalpha() and w.islower() and w not in _STOP


def _integer(i: str, grouped: bool, m: re.Match, g: str | None = None) -> str:
    if "g" in FORCE:
        return ordinal(int(i), FORCE["g"] == "f") if FORCE.get("mode") == "ord" else cardinal(int(i), FORCE["g"])
    if spoken_digits(i, grouped, 7):
        return digits(i)
    return cardinal(int(i), g or _gender(m))


# Readings


def read_date(m: re.Match) -> str | None:
    if m.group(1):  # a law number: Ley 2/2018
        return f"{m.group(1)} {cardinal(int(m.group(2)))} de {cardinal(int(m.group(3)))}"
    d, mo = int(m.group(4)), int(m.group(5))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{'primero' if d == 1 else cardinal(d)} de {MONTHS[mo - 1]} de {cardinal(int(m.group(6)))}"


def read_phone(m: re.Match) -> str:
    return digits(re.sub(r"\D", "", m.group(0)))


def read_fraction(m: re.Match) -> str | None:
    return FRACTIONS.get((int(m.group(1)), int(m.group(2))))


def read_time(m: re.Match) -> str | None:
    h, mm = int(m["h"]), m["mm"]
    if not mm or h > 24 or int(mm) > 59 or m.groupdict().get("sep") == "." and before(m) not in ("las", "la", "a"):
        return None
    return cardinal(h, "f") + (f" {cardinal(int(mm))}" if int(mm) else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    one, many = FORCE["currency"] if "currency" in FORCE and currency in ("$", "US$") else CURRENCIES[currency]
    g = "f" if one == "libra" else "a"
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is not None and len(f) == 2 and not scale:
        return f"{_integer(i, grouped, m, g)} {one if i == '1' else many} con {cardinal(int(f))}"
    words = _scaled(s, scale, m) if scale else read_number(s, m, g=g)
    if after(m).lower() in (one, many, "pesos"):  # "$5,000 dólares": the currency word already follows
        return words
    return f"{words} {'de ' if scale and SCALES[scale] != 'mil' else ''}{one if s == '1' and not scale else many}"


def _scaled(s: str, scale: str, m: re.Match) -> str:
    word = "mil" if SCALES[scale] == "mil" else "millón" if s == "1" else "millones"
    return f"{read_number(s, m, g='a')} {word}"


def read_scale(s: str, scale: str, m: re.Match) -> str | None:
    if scale in "kK" and not _counted(m):
        return None  # 4K, 2K: a letter, read by the code pass
    return _scaled(s, scale, m)


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return f"{read_number(s, m, g='m')} por ciento" + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    return f"{read_number(s, m, g='f' if _feminine(one.split()[0]) else 'a')} {one if s == '1' else many}"


def read_ordinal(m: re.Match) -> str:
    n, suffix = int(m.group(1)), m.group(2) or m.group(3)
    if suffix == "er":
        return {1: "primer", 3: "tercer"}.get(n, ordinal(n, False))
    return ordinal(n, suffix in ("ª", "era", "ra", "da", "ta", "va", "ma", "na"))


def read_decade(m: re.Match) -> str:
    return read_number(m.group(1), m, g="m") + "s"  # los 80s: los ochentas


def read_compound(m: re.Match) -> str:
    return f"{read_number(m.group(1), m, g='m')}-{m.group(2)}"


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) >= 7 else cardinal(int(s))


def read_number(s: str, m: re.Match, zero: bool = True, g: str | None = None) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m, g)
    g = FORCE.get("g") or g or _gender(m)
    word = "coma" if s[-len(f) - 1] == "," else "punto"  # read the separator as written
    zeros = ["cero"] * (len(f) - len(f.lstrip("0")))
    frac = digits(f) if len(f) > 2 else " ".join(zeros + ([cardinal(int(f), g)] if f.strip("0") else []))
    return (f"{cardinal(int(i or 0), 'm' if g == 'a' else g)} " if zero else "") + f"{word} {frac}"
