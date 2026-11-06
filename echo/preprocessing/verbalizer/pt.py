"""Portuguese readings for the verbalizer (see core.py for what a language module provides).

Brazilian forms (dezesseis, bilhão); European teens (dezasseis) are a GRID alternative.
"""
import re

from echo.preprocessing.verbalizer.core import FORCE, after, before, parse, spoken_digits

GROUP, DECIMAL = ".", ",."
POINT, DOT, BY, SCORE, MINUS = "ponto", "ponto", "por", " a ", "menos"
TIME_SEPS = ":h"
TIME_SUFFIX = r"h\b"
# A numeric date: 24/03/2022, 07-11-2018.
DATE = r"(?<![\w.,/-])(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})(?![\w]|[.,/-]?\d)"
PHONES = [r"(?<![\w.,-])(?:0800|\d{4,5})-\d{4,6}(?![\w-]|[.,]?\d)"]
ORDINAL = r"(?<![\w.,])(\d+)\s?([ºª])"
DECADE = None
CURRENCIES = {"R$": ("real", "reais"), "US$": ("dólar", "dólares"), "$": ("dólar", "dólares"), "€": ("euro", "euros"),
              "£": ("libra", "libras")}
# Full words are kept as written; abbreviations take the singular below 2.
SCALES = {"mil": "mil", "k": "mil", "K": "mil", "milhão": "milhão", "milhões": "milhões", "bilhão": "bilhão",
          "bilhões": "bilhões", "trilhão": "trilhão", "trilhões": "trilhões", "mi": ("milhão", "milhões"),
          "bi": ("bilhão", "bilhões"), "tri": ("trilhão", "trilhões")}
UNITS = {
    "km": ("quilômetro", "quilômetros"), "m": ("metro", "metros"), "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"), "nm": ("nanômetro", "nanômetros"), "m²": ("metro quadrado", "metros quadrados"),
    "km²": ("quilômetro quadrado", "quilômetros quadrados"), "m³": ("metro cúbico", "metros cúbicos"),
    "ha": ("hectare", "hectares"), "kg": ("quilo", "quilos"), "g": ("grama", "gramas"), "mg": ("miligrama", "miligramas"),
    "t": ("tonelada", "toneladas"), "l": ("litro", "litros"), "L": ("litro", "litros"), "ml": ("mililitro", "mililitros"),
    "mL": ("mililitro", "mililitros"), "TB": ("terabyte", "terabytes"), "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"), "KB": ("kilobyte", "kilobytes"), "Hz": ("hertz", "hertz"),
    "kHz": ("quilohertz", "quilohertz"), "MHz": ("megahertz", "megahertz"), "GHz": ("gigahertz", "gigahertz"),
    "W": ("watt", "watts"), "kW": ("quilowatt", "quilowatts"), "kWh": ("quilowatt-hora", "quilowatts-hora"),
    "V": ("volt", "volts"), "km/h": ("quilômetro por hora", "quilômetros por hora"), "mph": ("milha por hora", "milhas por hora"),
    "°C": ("grau Celsius", "graus Celsius"), "°": ("grau", "graus"),
}
PREFIXES = {"nº": "número", "Nº": "número", "n.º": "número", "N.º": "número", "núm.": "número", "art.": "artigo",
            "Art.": "artigo", "pág.": "página", "§": "parágrafo"}
SUFFIXES = {}
FRACTIONS = {(1, 2): "meio", (1, 3): "um terço", (2, 3): "dois terços", (1, 4): "um quarto", (3, 4): "três quartos"}
SYMBOLS = {"&": "e", "+": "mais", "=": "igual a", "@": "arroba", "%": "por cento", "°": "graus", "#": "hashtag",
           "²": "ao quadrado", "³": "ao cubo", "π": "pi", "×": "vezes", "€": "euros", "$": "dólares", "£": "libras",
           "§": "parágrafo"}
GRID = ([{"g": g} for g in ("m", "f")] + [{"g": g, "mode": "ord"} for g in ("m", "f")]
        + [{"frac": f} for f in ("digits", "number")] + [{"money": s} for s in ("short", "noun")]
        + [{"time": t} for t in ("hours", "meia")] + [{"six": True}, {"european": True}])

ONES = ("zero um dois três quatro cinco seis sete oito nove dez onze doze treze catorze quinze dezesseis dezessete "
        "dezoito dezenove").split()
_EUROPEAN = {"dezesseis": "dezasseis", "dezessete": "dezassete", "dezenove": "dezanove"}
TENS = {2: "vinte", 3: "trinta", 4: "quarenta", 5: "cinquenta", 6: "sessenta", 7: "setenta", 8: "oitenta", 9: "noventa"}
HUNDREDS = {1: "cento", 2: "duzentos", 3: "trezentos", 4: "quatrocentos", 5: "quinhentos", 6: "seiscentos",
            7: "setecentos", 8: "oitocentos", 9: "novecentos"}
_BIG = [(10**12, "trilhão", "trilhões"), (10**9, "bilhão", "bilhões"), (10**6, "milhão", "milhões")]
MONTHS = "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro".split()
_ORD_UNITS = "_ primeir segund terceir quart quint sext sétim oitav non".split()
_ORD_TENS = "_ décim vigésim trigésim quadragésim quinquagésim sexagésim septuagésim octogésim nonagésim".split()
_ORD_HUNDREDS = ("_ centésim ducentésim trecentésim quadringentésim quingentésim sexcentésim septingentésim "
                 "octingentésim noningentésim").split()
# Nouns in -a are feminine unless listed; other feminine nouns are listed or end in a feminine ending.
_MASCULINE_A = {"dia", "mapa", "problema", "sistema", "programa", "tema", "idioma", "clima", "planeta", "poema",
                "cometa", "drama", "esquema", "sofá", "diagrama", "teorema", "telefonema", "cinema", "prisma", "pijama",
                "aroma", "dilema", "fantasma", "panorama", "enigma", "dogma", "carisma", "atleta", "guia", "pila"}
_FEMININE = {"vez", "mão", "foto", "moto", "mulher", "flor", "noite", "tarde", "parte", "gente", "lei", "luz", "voz",
             "paz", "rede", "sede", "fase", "base", "frase", "classe", "chave", "fonte", "mente", "morte", "sorte",
             "ponte", "árvore", "nuvem", "ordem", "série", "espécie", "superfície", "cor", "dor", "mãe", "pele", "neve",
             "febre", "carne", "torre", "hora", "semana", "polegada", "lição"}
_FEMININE_ENDINGS = ("a", "ção", "são", "dade", "tude", "gem", "ise", "ite", "ez")
_STOP = set("""de do da dos das em no na nos nas com por para sem que e ou a o os as um uma uns umas é são era foi mais
menos até desde entre sobre contra após cada todo toda todos todas este esta estes estas esse essa esses essas isso isto
já agora quando onde muito também se lhe me te ao à aos às pelo pela pelos pelas não sim meu minha seu sua""".split())
_HOUR_CUES = {"às", "à", "das", "pelas"}  # às duas, das dez


# Legal headings read as cardinals from 10 on, even when written with º: artigo 80º, artigo oitenta.
_HEADINGS = {"artigo", "art", "parágrafo", "capítulo", "inciso"}


# Number words


def _unit(n: int, g: str) -> str:
    w = ONES[n]
    if g == "f" and n in (1, 2):
        return "uma" if n == 1 else "duas"
    return _EUROPEAN.get(w, w) if FORCE.get("european") else w


def _below100(n: int, g: str) -> str:
    if n < 20:
        return _unit(n, g)
    return TENS[n // 10] + (f" e {_unit(n % 10, g)}" if n % 10 else "")


def _below1000(n: int, g: str) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below100(r, g)
    if n == 100:
        return "cem"
    head = HUNDREDS[h][:-2] + "as" if g == "f" and h > 1 else HUNDREDS[h]
    return head + (f" e {_below100(r, g)}" if r else "")


def cardinal(n: int, g: str = "m") -> str:
    """Portuguese cardinal; "e" joins the last group when it is below 100 or round hundreds: dois mil e vinte e dois,
    mil quinhentos e vinte, dois milhões e trezentos mil. Thousands agree with the noun: duas mil pessoas."""
    groups = []  # (count, words)
    for value, one, many in _BIG:
        q, n = divmod(n, value)
        if q:
            groups.append((q, f"{cardinal(q)} {one if q == 1 else many}"))
    q, n = divmod(n, 1000)
    if q:
        groups.append((q, "mil" if q == 1 else f"{_below1000(q, g)} mil"))
    if n or not groups:
        groups.append((n, _below1000(n, g)))
    if len(groups) == 1:
        return groups[0][1]
    last, words = groups[-1][0], [w for _, w in groups]
    joiner = " e " if last < 100 or last % 100 == 0 else " "
    return " ".join(words[:-1]) + joiner + words[-1]


def ordinal(n: int, fem: bool) -> str:
    """décimo terceiro, vigésima primeira, milésimo."""
    end = "a" if fem else "o"
    words = []
    t, n = divmod(n, 1000)
    if t:
        words.append(("" if t == 1 else cardinal(t) + " ") + "milésim" + end)
    words += [stems[n // value % 10] + end for value, stems in ((100, _ORD_HUNDREDS), (10, _ORD_TENS), (1, _ORD_UNITS))
              if n // value % 10]
    return " ".join(words)


def digits(s: str) -> str:
    """Digit by digit; with "six" in FORCE a 6 is "meia", as in Brazilian phone numbers."""
    return " ".join("meia" if c == "6" and FORCE.get("six") else _unit(int(c), "m") for c in s)


# Context


def _feminine(noun: str) -> bool:
    w = noun.lower()
    sing = (w[:-3] + "ão" if w.endswith(("ões", "ães")) else w[:-2] if w.endswith(("res", "zes", "les")) else
            w[:-1] if w.endswith("s") else w)
    if sing in _MASCULINE_A or w in _MASCULINE_A:
        return False
    return sing in _FEMININE or w in _FEMININE or sing.endswith(_FEMININE_ENDINGS)


def _gender(m: re.Match) -> str:
    """The gender of a number from the noun after it; an hour after "às" is feminine (às duas)."""
    w = after(m)
    if w in ("da", "de") and after(m, 2).lower() in ("manhã", "tarde", "noite", "madrugada"):
        return "f"  # duas da tarde
    if not w[:1].isalpha() or w.lower() in _STOP:
        return "f" if before(m) in _HOUR_CUES else "m"
    return "f" if _feminine(w) else "m"


def _counted(m: re.Match) -> bool:
    w = after(m)
    return len(w) > 2 and w.isalpha() and w.islower() and w not in _STOP


def _integer(i: str, grouped: bool, m: re.Match, g: str | None = None) -> str:
    n = int(i)
    if "g" in FORCE:
        return ordinal(n, FORCE["g"] == "f") if FORCE.get("mode") == "ord" else cardinal(n, FORCE["g"])
    if spoken_digits(i, grouped, 7):
        return digits(i)
    if n == 1 and g is None and after(m).lower() == "de" and after(m, 2).lower() in MONTHS:
        return "primeiro"  # 1 de maio
    return cardinal(n, g or _gender(m))


def RANGE(m: re.Match, a: str = "", b: str = "") -> str:
    """2016-2022 and 10-15 read "a"; scores, formations and codes (4-3-3, 60-40, A-320) keep "-"."""
    head, tail = m.string[: m.start()], m.string[m.end():]
    if not a:
        if re.search(r"[A-Za-z]\d*$", head) or re.match(r"\d*[A-Za-z]", tail):
            return "-"
        a, b = re.search(r"\d+$", head).group(0), re.match(r"\d+", tail).group(0)
    if not (a.isdigit() and b.isdigit()):
        return "-"
    return " a " if len(a) == len(b) and int(b) > int(a) else "-"


# Readings


def read_date(m: re.Match) -> str | None:
    d, mo, y = int(m.group(1)), int(m.group(3)), m.group(4)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{'primeiro' if d == 1 else cardinal(d)} de {MONTHS[mo - 1]} de {cardinal(int(y))}"


def read_phone(m: re.Match) -> str | None:
    groups = m.group(0).split("-")
    if all(len(g) == 4 and 1900 <= int(g) <= 2100 for g in groups):
        return None  # 2021-2022: a range of years
    return ", ".join("zero oitocentos" if g == "0800" else digits(g) for g in groups)


def read_fraction(m: re.Match) -> str | None:
    return FRACTIONS.get((int(m.group(1)), int(m.group(2))))


def read_time(m: re.Match) -> str | None:
    """14h30: catorze e trinta; 21h: vinte e uma horas. The hour is feminine (horas)."""
    h, mm = int(m["h"]), m["mm"]
    if mm and (h > 24 or int(mm) > 59):
        return None
    hour = cardinal(h, "f")
    hours = f"{hour} {'hora' if h == 1 else 'horas'}"
    if not mm or int(mm) == 0:
        return hours
    style = FORCE.get("time")
    if style == "hours":
        return f"{hours} e {cardinal(int(mm))} {'minuto' if int(mm) == 1 else 'minutos'}"
    if style == "meia" and mm == "30":
        return f"{hour} e meia"
    return f"{hour} e {cardinal(int(mm))}"


def _with_de(words: str) -> str:
    """um milhão de reais, but dez mil reais."""
    return words + (" de" if words.endswith(("milhão", "milhões", "bilhão", "bilhões", "trilhão", "trilhões")) else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    one, many = CURRENCIES[currency]
    g = "f" if one == "libra" else "m"
    if re.fullmatch(r"\d{1,3},\d{3}", s):
        s = s.replace(",", "")  # R$ 37,517: prices have two decimals, so three digits are a thousands group
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if after(m).lower() in (one, many):  # "R$ 50 reais": the currency word already follows
        return _scaled(s, scale, m, g) if scale else read_number(s, m, g=g)
    if scale:
        return f"{_with_de(_scaled(s, scale, m, g))} {many}"
    if f is not None and len(f) == 2 and s[-3] == ",":  # a price: vinte e nove reais e noventa centavos
        cents = f"{cardinal(int(f))} {'centavo' if int(f) == 1 else 'centavos'}"
        if int(i or 0) == 0:
            return cents
        whole, unit = _integer(i, grouped, m, g), one if i == "1" else many
        if int(f) == 0:
            return f"{_with_de(whole)} {unit}"
        style = FORCE.get("money")
        if style == "short":
            return f"{whole} e {cardinal(int(f))}"
        return f"{_with_de(whole)} {unit} e " + (cardinal(int(f)) if style == "noun" else cents)
    return f"{_with_de(read_number(s, m, g=g))} {one if s == '1' else many}"


def _scaled(s: str, scale: str, m: re.Match, g: str | None = None) -> str:
    word = SCALES[scale]
    if word == "mil":
        return f"{read_number(s, m, g=g or _gender(m))} mil"  # vinte e duas mil pessoas
    if isinstance(word, tuple):
        word = word[0] if int(parse(s, GROUP, DECIMAL)[0] or 0) < 2 else word[1]
    return f"{read_number(s, m, g='m')} {word}"


def read_scale(s: str, scale: str, m: re.Match) -> str | None:
    if scale in "kK" and int(parse(s, GROUP, DECIMAL)[0] or 0) < 10 and not _counted(m):
        return None  # 4K, 2K: a resolution, read by the code pass; but 500k is quinhentos mil
    return _scaled(s, scale, m)


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return f"{read_number(s, m, g='m')} por cento" + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    one, many = UNITS[unit]
    g = "f" if _feminine(one.split()[0]) else "m"
    return f"{read_number(s, m, g=g)} {one if int(parse(s, GROUP, DECIMAL)[0] or 0) < 2 else many}"


def read_ordinal(m: re.Match) -> str:
    n, fem = int(m.group(1)), m.group(2) == "ª"
    if "g" in FORCE and FORCE.get("mode") != "ord":
        return cardinal(n, FORCE["g"])
    if "g" not in FORCE and n >= 10 and before(m) in _HEADINGS:
        return cardinal(n)
    return ordinal(n, FORCE["g"] == "f" if "g" in FORCE else fem)


def read_compound(m: re.Match) -> str:
    return f"{read_number(m.group(1), m, g='m')}-{m.group(2)}"


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) >= 7 else cardinal(int(s))


def read_number(s: str, m: re.Match, zero: bool = True, g: str | None = None) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m, g)
    g = FORCE.get("g") or g or "m"
    word = "vírgula" if s[-len(f) - 1] == "," else "ponto"  # read the separator as written
    zeros = ["zero"] * (len(f) - len(f.lstrip("0")))
    if FORCE.get("frac", "digits" if len(f) > 2 else "number") == "digits":
        frac = digits(f)
    else:
        frac = " ".join(zeros + ([cardinal(int(f))] if f.strip("0") else []))
    return (f"{cardinal(int(i or 0), g)} " if zero else "") + f"{word} {frac}"
