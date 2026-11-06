"""French readings for the verbalizer (see core.py for what a language module provides)."""
import re

from echo.preprocessing.verbalizer.core import FORCE, after, before, parse, spoken_digits

GROUP, DECIMAL = "   .", ",."
POINT, DOT, RANGE, SCORE, MINUS = "virgule", "point", " à ", " à ", "moins"
TIME_SEPS = ":.h"
TIME_SUFFIX = "h"
DATE = r"(?<![\w.,/])(\d{1,2})([./])(\d{1,2})(?:\2(\d{4}|\d{2}))?(?![\w]|[.,/]?\d)"
PHONES = [r"(?<![\w.,])0\d(?:[ .]?\d\d){4}(?![\w]|[.,]?\d)"]
ORDINAL = r"(?<![\w.,])(\d+)(ères|èmes|ère|ème|eme|ers|er|res|re|es|ndes|nde|nds|nd|e|è)\b|\b([IVXLC]{2,}|I)(er|re|ère|e|ème)\b"
DECADE = None
CURRENCIES = {"€": ("euro", "euros"), "$": ("dollar", "dollars"), "£": ("livre", "livres"),
              "CHF": ("franc suisse", "francs suisses"), "M€": ("million d'euros", "millions d'euros")}
SCALES = {"millions": "millions", "million": "million", "milliards": "milliards", "milliard": "milliard",
          "k": "mille", "K": "mille"}
UNITS = {
    "km": ("kilomètre", "kilomètres"), "m": ("mètre", "mètres"), "cm": ("centimètre", "centimètres"),
    "mm": ("millimètre", "millimètres"), "nm": ("nanomètre", "nanomètres"), "m²": ("mètre carré", "mètres carrés"),
    "km²": ("kilomètre carré", "kilomètres carrés"), "m³": ("mètre cube", "mètres cubes"), "ha": ("hectare", "hectares"),
    "kg": ("kilogramme", "kilogrammes"), "g": ("gramme", "grammes"), "mg": ("milligramme", "milligrammes"),
    "t": ("tonne", "tonnes"), "l": ("litre", "litres"), "L": ("litre", "litres"), "cl": ("centilitre", "centilitres"),
    "ml": ("millilitre", "millilitres"), "mL": ("millilitre", "millilitres"),
    "km/h": ("kilomètre par heure", "kilomètres par heure"), "°C": ("degré Celsius", "degrés Celsius"),
    "°": ("degré", "degrés"), "To": ("téraoctet", "téraoctets"), "Go": ("gigaoctet", "gigaoctets"),
    "Mo": ("mégaoctet", "mégaoctets"), "Ko": ("kilooctet", "kilooctets"), "GHz": ("gigahertz", "gigahertz"),
    "MHz": ("mégahertz", "mégahertz"), "Hz": ("hertz", "hertz"), "kW": ("kilowatt", "kilowatts"),
    "kWh": ("kilowattheure", "kilowattheures"), "W": ("watt", "watts"), "V": ("volt", "volts"),
    "Mbit/s": ("mégabit par seconde", "mégabits par seconde"), "min": ("minute", "minutes"), "s": ("seconde", "secondes"),
}
PREFIXES = {"n°": "numéro", "N°": "numéro", "nº": "numéro", "p.": "page", "art.": "article", "Art.": "article"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "un demi", (1, 3): "un tiers", (2, 3): "deux tiers", (1, 4): "un quart", (3, 4): "trois quarts"}
SYMBOLS = {"&": "et", "+": "plus", "=": "égale", "@": "arobase", "%": "pour cent", "°": "degrés", "§": "paragraphe",
           "#": "hashtag", "²": "carré", "³": "cube", "π": "pi", "×": "fois", "½": "un demi", "€": "euros",
           "$": "dollars", "£": "livres"}

ONES = ("zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize dix-sept "
        "dix-huit dix-neuf").split()
TENS = "_ dix vingt trente quarante cinquante soixante _ quatre-vingt".split()
MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
_WEEKDAYS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
# Nouns in -e are feminine unless listed or ending like a masculine noun; other feminine nouns are listed.
_FEMININE = {"fois", "voix", "nuit", "part", "dent", "main", "fleur", "mort", "photo", "vidéo", "radio", "moto", "eau",
             "loi", "foi", "fin", "faim", "peur", "mer", "cour", "tour", "souris", "noix", "paix", "croix", "forêt",
             "image", "page", "plage", "cage", "nage", "rage"}
_MASCULINE_E = {"monde", "livre", "groupe", "modèle", "problème", "système", "programme", "exemple", "nombre", "membre",
                "homme", "père", "frère", "rôle", "style", "site", "espace", "service", "commerce", "couple", "magazine",
                "musée", "lycée", "verre", "chiffre", "véhicule", "stade", "poste", "signe", "rythme", "volume",
                "domaine", "vote", "texte", "téléphone", "jeune", "collège", "kilowattheure", "gramme", "kilogramme",
                "milligramme", "pourcentage", "ordre", "type", "genre", "acte", "arbre", "autre", "génome", "crime",
                "trimestre", "semestre", "siècle", "masque", "risque", "casque", "disque", "manque", "parc", "vice", "film"}
_MASCULINE_ENDINGS = ("age", "ège", "isme", "ème", "aire", "tre", "cle", "ste", "ome")
_STOP = set("avec dans pour mais donc puis sur sous par chez vers entre sans contre comme selon alors aussi déjà "
            "encore très plus moins tout tous toute toutes".split())
_NUMBER_WORD = ("zéro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|"
                "vingts?|trente|quarante|cinquante|soixante|cents?|mille")
GRID = ([{"fem": f} for f in (False, True)] + [{"style": s} for s in ("cardinal", "hundreds", "digits")]
        + [{"frac": f} for f in ("digits", "number")])
_TIME_WORDS = {"à", "vers", "dès", "de", "jusqu'à", "avant", "après", "entre", "et"}


# Number words


def _below100(n: int, fem: bool, final: bool) -> str:
    if n < 20:
        return "une" if fem and n == 1 else ONES[n]
    t, u = divmod(n, 10)
    if t in (7, 9):  # soixante-dix, quatre-vingt-dix
        t, u = t - 1, u + 10
    if u == 0:
        return TENS[t] + ("s" if t == 8 and final else "")
    unit = "une" if fem and u == 1 else ONES[u]
    return f"{TENS[t]} et {unit}" if u in (1, 11) and t < 8 else f"{TENS[t]}-{unit}"


def _below1000(n: int, fem: bool, final: bool) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below100(r, fem, final)
    head = "cent" if h == 1 else f"{ONES[h]} cent" + ("s" if r == 0 and final else "")
    return head + (f" {_below100(r, fem, final)}" if r else "")


def cardinal(n: int, fem: bool = False, final: bool = True) -> str:
    """French cardinal; `final` is False when "mille" follows (deux cent mille, quatre-vingt mille)."""
    if n < 1000:
        return _below1000(n, fem, final)
    if n < 10**6:
        t, r = divmod(n, 1000)
        head = "mille" if t == 1 else f"{_below1000(t, False, False)} mille"
        return head + (f" {_below1000(r, fem, final)}" if r else "")
    for value, word in ((10**9, "milliard"), (10**6, "million")):
        if n >= value:
            q, r = divmod(n, value)
            return f"{cardinal(q)} {word}{'s' if q > 1 else ''}" + (f" {cardinal(r, fem, final)}" if r else "")


def ordinal(n: int, fem: bool = False) -> str:
    if n == 1:
        return "première" if fem else "premier"
    words = re.sub(r"(vingt|cent)s$", r"\1", cardinal(n))
    return re.sub(r"e$", "", re.sub(r"cinq$", "cinqu", re.sub(r"neuf$", "neuv", words))) + "ième"


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


def _roman(s: str) -> int | None:
    """The value of a Roman numeral in canonical form (XIX), None otherwise."""
    n = sum(-_ROMAN[a] if _ROMAN[a] < _ROMAN[b] else _ROMAN[a] for a, b in zip(s, s[1:])) + _ROMAN[s[-1]]
    canonical, rest = "", n
    for value, sym in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while rest >= value:
            canonical, rest = canonical + sym, rest - value
    return n if canonical == s else None


# Context


def _feminine(noun: str) -> bool:
    w = noun.lower().split("-")[-1]
    w = w if w in _FEMININE else w.removesuffix("s")
    return w in _FEMININE or w.endswith("e") and w not in _MASCULINE_E and w not in MONTHS and w not in _WEEKDAYS \
        and not w.endswith(_MASCULINE_ENDINGS)


def _number_form(value: str, one: str, many: str) -> str:
    """A noun after a number is singular below 2: un virgule cinq kilomètre, deux kilomètres."""
    i, _, _ = parse(value, GROUP, DECIMAL)
    return one if int(i or 0) < 2 else many


def _counted(m: re.Match) -> bool:
    """A lowercase noun follows: 10k abonnés."""
    w = after(m)
    return len(w) > 2 and w.isalpha() and w.islower() and w not in _STOP and not re.fullmatch(_NUMBER_WORD, w)


def _forced(i: str, style: str) -> str:
    n = int(i)
    if style == "digits":
        return digits(i)
    if style == "hundreds" and 1100 <= n <= 1999:  # dix-neuf cent quatre-vingt-dix
        return f"{cardinal(n // 100)} cent" + (f" {cardinal(n % 100)}" if n % 100 else "")
    return cardinal(n)


def _integer(i: str, grouped: bool, m: re.Match, fem: bool | None = None) -> str:
    if "style" in FORCE:
        return _forced(i, FORCE["style"])
    if "fem" in FORCE:
        fem = FORCE["fem"]
    if spoken_digits(i, grouped, 5):
        return digits(i)
    if fem is None:
        fem = _feminine(after(m))
    return cardinal(int(i), fem, final=before(m) not in ("années", "année"))


def BY(m: re.Match) -> str:
    return "fois" if "=" in m.string[m.end():] else "par"  # 12 x 4 = 48, but 1920x1080


# Readings


def read_date(m: re.Match) -> str | None:
    d, mo, y = int(m.group(1)), int(m.group(3)), m.group(4)
    if not (1 <= d <= 31 and 1 <= mo <= 12) or not y and not (m.group(1)[0] == "0" or m.group(3)[0] == "0"):
        return None  # without a year only a zero-padded day or month is a date: 05/11, but not 3/4
    return f"{ordinal(1) if d == 1 else cardinal(d)} {MONTHS[mo - 1]}" + (f" {cardinal(int(y))}" if y else "")


def read_phone(m: re.Match) -> str:
    """Read in pairs: zéro six douze trente-quatre ..."""
    return " ".join(f"zéro {ONES[int(p[1])]}" if p[0] == "0" else cardinal(int(p)) for p in re.findall(r"\d\d", m.group(0)))


def read_fraction(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    return FRACTIONS.get((a, b), f"{cardinal(a)} sur {cardinal(b)}")


def read_time(m: re.Match) -> str | None:
    h, mm = int(m["h"]), m["mm"]
    if mm and (h > 24 or int(mm) > 59 or m.groupdict().get("sep") == "." and before(m) not in _TIME_WORDS):
        return None
    words = f"{cardinal(h, fem=True)} {'heure' if h < 2 else 'heures'}"
    return words + (f" {cardinal(int(mm), fem=True)}" if mm and int(mm) else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    one, many = CURRENCIES[currency]
    if currency == "M€":
        scale, one, many = "millions", "euro", "euros"
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if scale:
        words = _scaled(s, scale, m)
        return f"{words} {many}" if SCALES[scale] == "mille" else re.sub(r"\bde (?=[aeiouyéh])", "d'", f"{words} de {many}")
    fem = one == "livre"
    if f is not None and len(f) == 2:  # a price: trois euros cinquante, quatre-vingt-dix-neuf centimes
        if int(i or 0) == 0:
            return f"{cardinal(int(f))} centimes"
        return f"{_integer(i, grouped, m, fem)} {one if i == '1' else many} {cardinal(int(f))}"
    return f"{read_number(s, m, fem=fem)} {_number_form(s, one, many)}"


def _scaled(s: str, scale: str, m: re.Match) -> str:
    word = SCALES[scale]
    if word == "mille":
        return f"{read_number(s, m, fem=False)} mille"
    return f"{read_number(s, m, fem=False)} {_number_form(s, word.removesuffix('s'), word.removesuffix('s') + 's')}"


def read_scale(s: str, scale: str, m: re.Match) -> str | None:
    if SCALES[scale] == "mille" and not _counted(m):
        return None  # 4K, 40K, 3k de Q: a letter, read by the code pass
    return _scaled(s, scale, m)


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return f"{read_number(s, m, fem=False)} pour cent" + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    return f"{read_number(s, m, fem=_feminine(one.split()[0]))} {_number_form(s, one, many)}"


def read_ordinal(m: re.Match) -> str | None:
    n = int(m.group(1)) if m.group(1) else _roman(m.group(3))
    suffix = m.group(2) or m.group(4)
    if n is None:
        return None
    if suffix.startswith("nd"):
        words = "seconde" if suffix.startswith("nde") else "second"
    else:
        words = ordinal(n, fem=suffix in ("re", "ère", "res", "ères"))
    return words + ("s" if suffix.endswith("s") else "")


def read_compound(m: re.Match) -> str:
    return f"{read_number(m.group(1), m)}-{m.group(2)}"


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) > 4 else cardinal(int(s))


def read_number(s: str, m: re.Match, zero: bool = True, fem: bool | None = None) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m, fem)
    zeros = ["zéro"] * (len(f) - len(f.lstrip("0")))
    frac = digits(f) if FORCE.get("frac", "digits" if len(f) > 3 else "number") == "digits" else " ".join(zeros + ([cardinal(int(f))] if f.strip("0") else []))
    return (f"{cardinal(int(i), bool(FORCE.get('fem', fem)))} " if zero else "") + f"virgule {frac}"
