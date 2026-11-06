"""Polish readings for the verbalizer (see core.py for what a language module provides).

Polish numbers inflect for case and gender, so most readings first find the case a number takes: from the
preposition before it (od pięciu), the noun after it (z trzema kolegami), a governing noun or verb (od kwoty
trzystu tysięcy, potrzebowali stu gramów), or the year and date patterns that make it an ordinal
(w tysiąc dziewięćset osiemdziesiątym piątym roku, pierwszego stycznia).
"""
import re

from echo.preprocessing.verbalizer.core import FORCE, PUNCT, after, before, parse, spoken_digits

GROUP, DECIMAL = "   .", ",."
POINT, DOT, RANGE, SCORE, MINUS = "kropka", "kropka", "-", " do ", "minus"
TIME_SEPS = ":."
TIME_SUFFIX = r"(?!)"
DATE = r"(?<![\w.,])(\d{1,2})\.(\d{1,2})\.(\d{4})(\s?r\.)?(?![\w]|[.,]?\d)"
PHONES = [r"(?<=numer )\d{3} \d{3} \d{3}\b|(?<=numerem )\d{3} \d{3} \d{3}\b|(?<=tel\. )\d{3} \d{3} \d{3}\b"]
ORDINAL = r"(?<![\w.,])(\d+)\.(?=\s+\S|,|$)"
DECADE = None

CASES = ("nom", "gen", "dat", "acc", "ins", "loc")
GRID = ([{"mode": "card", "case": c, "g": g} for c in CASES for g in ("m", "f", "n")]
        + [{"mode": "ord", "case": c, "g": g} for c in CASES for g in ("m", "f", "n", "pl")])


def _noun(forms: str) -> dict[str, str]:
    """A noun's twelve forms, singular then plural, each in CASES order; multi-word forms are split by "|"."""
    words = forms.split("|") if "|" in forms else forms.split()
    return {f"{c}_{num}": w for num, part in (("sg", words[:6]), ("pl", words[6:])) for c, w in zip(CASES, part)}


def _masculine(stem: str, loc: str) -> str:
    return f"{stem} {stem}a {stem}owi {stem} {stem}em {loc} {stem}y {stem}ów {stem}om {stem}y {stem}ami {stem}ach"


_TYSIAC = _noun("tysiąc tysiąca tysiącowi tysiąc tysiącem tysiącu tysiące tysięcy tysiącom tysiące tysiącami tysiącach")
_MILION = _noun(_masculine("milion", "milionie"))
_MILIARD = _noun(_masculine("miliard", "miliardzie"))
_ZLOTY = _noun("złoty złotego złotemu złoty złotym złotym złote złotych złotym złote złotymi złotych")
_GROSZ = _noun("grosz grosza groszowi grosz groszem groszu grosze groszy groszom grosze groszami groszach")
_PROCENT = _noun("procent procent procentowi procent procentem procencie procent procent procentom procent procentami "
                 "procentach")
_EURO = _noun(" ".join(["euro"] * 12))
_DOLAR = _noun(_masculine("dolar", "dolarze"))
_STOPIEN = "stopień stopnia stopniowi stopień stopniem stopniu stopnie stopni stopniom stopnie stopniami stopniach"

CURRENCIES = {"zł": _ZLOTY, "PLN": _ZLOTY, "€": _EURO, "EUR": _EURO, "$": _DOLAR, "USD": _DOLAR,
              "£": _noun(_masculine("funt", "funcie")), "gr": _GROSZ}
SCALES = {"tys.": _TYSIAC, "mln": _MILION, "mld": _MILIARD}
UNITS = {unit: _noun(forms) for unit, forms in {
    "km": _masculine("kilometr", "kilometrze"), "m": _masculine("metr", "metrze"),
    "cm": _masculine("centymetr", "centymetrze"), "mm": _masculine("milimetr", "milimetrze"),
    "nm": _masculine("nanometr", "nanometrze"), "kg": _masculine("kilogram", "kilogramie"),
    "g": _masculine("gram", "gramie"), "mg": _masculine("miligram", "miligramie"), "l": _masculine("litr", "litrze"),
    "ml": _masculine("mililitr", "mililitrze"), "ha": _masculine("hektar", "hektarze"), "W": _masculine("wat", "wacie"),
    "kW": _masculine("kilowat", "kilowacie"), "MW": _masculine("megawat", "megawacie"), "V": _masculine("wolt", "wolcie"),
    "GB": _masculine("gigabajt", "gigabajcie"), "MB": _masculine("megabajt", "megabajcie"),
    "TB": _masculine("terabajt", "terabajcie"),
    "kWh": "kilowatogodzina kilowatogodziny kilowatogodzinie kilowatogodzinę kilowatogodziną kilowatogodzinie "
           "kilowatogodziny kilowatogodzin kilowatogodzinom kilowatogodziny kilowatogodzinami kilowatogodzinach",
    "Hz": "herc herca hercowi herc hercem hercu herce herców hercom herce hercami hercach",
    "°C": "|".join(f"{w} Celsjusza" for w in _STOPIEN.split()), "°": _STOPIEN,
    "m²": "metr kwadratowy|metra kwadratowego|metrowi kwadratowemu|metr kwadratowy|metrem kwadratowym|metrze kwadratowym|"
          "metry kwadratowe|metrów kwadratowych|metrom kwadratowym|metry kwadratowe|metrami kwadratowymi|metrach kwadratowych",
    "t": "tona tony tonie tonę toną tonie tony ton tonom tony tonami tonach",
    "h": "godzina godziny godzinie godzinę godziną godzinie godziny godzin godzinom godziny godzinami godzinach",
    "min": "minuta minuty minucie minutę minutą minucie minuty minut minutom minuty minutami minutach",
    "s": "sekunda sekundy sekundzie sekundę sekundą sekundzie sekundy sekund sekundom sekundy sekundami sekundach",
}.items()}
PREFIXES = {"nr": "numer", "Nr": "numer", "nr.": "numer", "ust.": "ustęp", "pkt": "punkt", "art.": "artykuł",
            "str.": "strona", "poz.": "pozycja", "ok.": "około"}
SUFFIXES = {"r.": "roku"}
FRACTIONS = {}
SYMBOLS = {"&": "and", "+": "plus", "=": "równa się", "%": "procent", "°": "stopni", "#": "", "/": "", "²": "do kwadratu",
           "³": "do sześcianu", "π": "pi", "×": "razy", "€": "euro", "$": "dolarów", "@": "małpa"}

# Cardinal forms: nominative/accusative, genitive/locative, dative, instrumental.
_CARD = {n: forms.split() for n, forms in {
    0: "zero zera zeru zerem", 2: "dwa dwóch dwóm dwoma", 3: "trzy trzech trzem trzema",
    4: "cztery czterech czterem czterema", 5: "pięć pięciu pięciu pięcioma", 6: "sześć sześciu sześciu sześcioma",
    7: "siedem siedmiu siedmiu siedmioma", 8: "osiem ośmiu ośmiu ośmioma", 9: "dziewięć dziewięciu dziewięciu dziewięcioma",
    10: "dziesięć dziesięciu dziesięciu dziesięcioma", 11: "jedenaście jedenastu jedenastu jedenastoma",
    12: "dwanaście dwunastu dwunastu dwunastoma", 13: "trzynaście trzynastu trzynastu trzynastoma",
    14: "czternaście czternastu czternastu czternastoma", 15: "piętnaście piętnastu piętnastu piętnastoma",
    16: "szesnaście szesnastu szesnastu szesnastoma", 17: "siedemnaście siedemnastu siedemnastu siedemnastoma",
    18: "osiemnaście osiemnastu osiemnastu osiemnastoma", 19: "dziewiętnaście dziewiętnastu dziewiętnastu dziewiętnastoma",
    20: "dwadzieścia dwudziestu dwudziestu dwudziestoma", 30: "trzydzieści trzydziestu trzydziestu trzydziestoma",
    40: "czterdzieści czterdziestu czterdziestu czterdziestoma",
    50: "pięćdziesiąt pięćdziesięciu pięćdziesięciu pięćdziesięcioma",
    60: "sześćdziesiąt sześćdziesięciu sześćdziesięciu sześćdziesięcioma",
    70: "siedemdziesiąt siedemdziesięciu siedemdziesięciu siedemdziesięcioma",
    80: "osiemdziesiąt osiemdziesięciu osiemdziesięciu osiemdziesięcioma",
    90: "dziewięćdziesiąt dziewięćdziesięciu dziewięćdziesięciu dziewięćdziesięcioma",
    100: "sto stu stu stoma", 200: "dwieście dwustu dwustu dwustoma", 300: "trzysta trzystu trzystu trzystoma",
    400: "czterysta czterystu czterystu czterystoma", 500: "pięćset pięciuset pięciuset pięciuset",
    600: "sześćset sześciuset sześciuset sześciuset", 700: "siedemset siedmiuset siedmiuset siedmiuset",
    800: "osiemset ośmiuset ośmiuset ośmiuset", 900: "dziewięćset dziewięciuset dziewięciuset dziewięciuset",
}.items()}
_CARD_COLUMN = {"nom": 0, "acc": 0, "gen": 1, "loc": 1, "dat": 2, "ins": 3}
_ONE = {g: dict(zip(CASES, forms.split())) for g, forms in {
    "m": "jeden jednego jednemu jeden jednym jednym", "f": "jedna jednej jednej jedną jedną jednej",
    "n": "jedno jednego jednemu jedno jednym jednym"}.items()}
_ORD = {1: "pierwsz", 2: "drug", 3: "trzec", 4: "czwart", 5: "piąt", 6: "szóst", 7: "siódm", 8: "ósm", 9: "dziewiąt",
        10: "dziesiąt", 11: "jedenast", 12: "dwunast", 13: "trzynast", 14: "czternast", 15: "piętnast", 16: "szesnast",
        17: "siedemnast", 18: "osiemnast", 19: "dziewiętnast", 20: "dwudziest", 30: "trzydziest", 40: "czterdziest",
        50: "pięćdziesiąt", 60: "sześćdziesiąt", 70: "siedemdziesiąt", 80: "osiemdziesiąt", 90: "dziewięćdziesiąt",
        100: "setn", 200: "dwusetn", 300: "trzechsetn", 400: "czterechsetn", 500: "pięćsetn", 600: "sześćsetn",
        700: "siedemsetn", 800: "osiemsetn", 900: "dziewięćsetn"}
_ADJ = {g: dict(zip(CASES, endings.split())) for g, endings in {
    "m": "y ego emu y ym ym", "f": "a ej ej ą ą ej", "n": "e ego emu e ym ym", "pl": "e ych ym e ymi ych"}.items()}
# Combining forms for compounds: 45-metrowe -> czterdziestopięciometrowe.
_COMBINING = {1: "jedno", 2: "dwu", 3: "trzy", 4: "cztero", 5: "pięcio", 6: "sześcio", 7: "siedmio", 8: "ośmio",
              9: "dziewięcio", 10: "dziesięcio", 11: "jedenasto", 12: "dwunasto", 13: "trzynasto", 14: "czternasto",
              15: "piętnasto", 16: "szesnasto", 17: "siedemnasto", 18: "osiemnasto", 19: "dziewiętnasto",
              20: "dwudziesto", 30: "trzydziesto", 40: "czterdziesto", 50: "pięćdziesięcio", 60: "sześćdziesięcio",
              70: "siedemdziesięcio", 80: "osiemdziesięcio", 90: "dziewięćdziesięcio", 100: "stu", 200: "dwustu",
              300: "trzystu", 400: "czterystu", 500: "pięćset", 600: "sześćset", 700: "siedemset", 800: "osiemset",
              900: "dziewięćset"}
# Hyphenated endings that make a number an ordinal (lat 70-tych), with the adjective ending they stand for.
_ORDINAL_SUFFIXES = {"tych": "ych", "ych": "ych", "ty": "y", "y": "y", "szy": "y", "gi": "y", "ci": "y", "go": "ego",
                     "ego": "ego", "tego": "ego", "ej": "ej", "tej": "ej", "ym": "ym", "tym": "ym", "ta": "a", "a": "a",
                     "te": "e", "e": "e", "mu": "emu", "temu": "emu"}

_PREP_CASE = {
    **dict.fromkeys("od do bez dla u około koło obok podczas według wśród spośród sprzed spod zza znad oprócz prócz "
                    "zamiast wokół dookoła naprzeciwko blisko niedaleko powyżej poniżej".split(), "gen"),
    **dict.fromkeys("dzięki przeciw przeciwko ku wbrew".split(), "dat"),
    **dict.fromkeys("przez ponad na o za po w we".split(), "acc"),
    **dict.fromkeys("z ze nad pod przed między pomiędzy".split(), "ins"),
    "przy": "loc",
}
_LOCATIVE_PREPS = {"w", "we", "o", "po", "przy", "na"}
# Prepositional phrases that govern the genitive, with the preposition they need: w ciągu dwóch tygodni.
_COMPLEX_PREPS = {"ciągu": "w", "trakcie": "w", "czasie": "w", "wyniku": "w", "przypadku": "w", "zakresie": "w",
                  "ramach": "w", "celu": "w", "postaci": "w", "powodu": "z"}
# A number after an amount noun is its genitive attribute: od kwoty trzystu tysięcy, na stan dwudziestu jeden radnych.
_AMOUNT_NOUNS = ("rzęd", "wysokoś", "gruboś", "średnic", "długoś", "szerokoś", "głębokoś", "wiek", "kwot", "kwoc",
                 "poziom", "stan", "strat", "strac", "wartoś", "liczb", "cen", "koszt", "opłat", "kar", "moc", "wag",
                 "wadz", "pojemnoś", "powierzchn", "prędkoś", "temperatur", "objętoś", "odległoś", "armi", "grup",
                 "okres", "limit", "próg", "prog", "budżet", "deficyt", "nadwyżk", "dochod", "dochód", "przychod",
                 "przychód", "wydatk", "wzrost", "spad", "zysk", "dotacj", "wynagrodzen", "pensj")
_NOUN_ENDINGS = ("", "a", "ą", "ę", "y", "i", "u", "e", "ie", "em", "iem", "om", "ów", "ami", "ach", "ć", "ci", "cią",
                 "ze", "ia", "ię", "ii", "ek", "ku")
# Verbs whose object is in the genitive: potrzebowali stu czterdziestu gramów.
_GENITIVE_VERBS = ("potrzeb", "używ", "użyj", "użyć", "wymag", "szuk", "poszuk", "unik", "brak", "zabrak", "życz",
                   "dotycz", "uczy", "nauczy", "słucha", "pilnu")
# Words skipped when looking for the verb that governs a number: brakuje nam trzech osób.
_CLITICS = {"mi", "ci", "mu", "jej", "nam", "wam", "im", "się", "sobie", "już", "jeszcze", "teraz", "tu", "tam", "aż",
            "nawet"}
_COPULA = {"jest", "są", "był", "była", "było", "były", "będzie", "będą", "to"}
_MONTHS_GEN = "stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia".split()
_MONTHS = set(_MONTHS_GEN) | set("styczeń luty marzec kwiecień maj czerwiec lipiec sierpień wrzesień październik "
                                 "listopad grudzień".split())
# Prepositions before a bare year (od 2020 do 2022) and the case of the year ordinal after them.
_YEAR_PREPS = {"od": "gen", "do": "gen", "z": "gen", "ze": "gen", "sprzed": "gen", "około": "gen", "przed": "ins",
               "po": "loc", "na": "acc"}
_COUNT_NOUNS = {"osób", "lat", "dni", "godzin", "minut", "sekund", "razy", "sztuk", "ludzi", "dzieci", "tygodni",
                "miesięcy"}
_GENITIVE_PLURALS = {forms["gen_pl"] for forms in [*UNITS.values(), *CURRENCIES.values(), *SCALES.values()]}
_CURRENCY_WORDS = {"zł", "złoty", "złote", "złotych"}
_FEMININE_PLURAL = ("osob", "osób", "godzin", "minut", "sekund", "złotówk", "kobiet", "stron", "sztuk", "rzecz", "częś",
                    "noc", "dob", "klas", "wersj", "sesj", "edycj", "zapał", "tablic", "butelk", "szklank", "łyżk")
_TIME_WORDS = {"o", "godzinie", "godz", "od", "do", "około", "koło", "po", "przed", "na", "między", "za"}
_HEADINGS = ("podpunkt", "punkt", "punkc", "rozdział", "paragraf", "artykuł", "pkt")
_ORDINAL_HEADINGS = ("podpunkt", "punkt", "punkc", "paragraf", "pkt")  # read as ordinals even without a dot


# Number words


def cardinal(n: int, case: str = "nom", g: str = "m") -> str:
    if n == 1:
        return _ONE[g][case]
    parts = []
    for value, forms in ((10**9, _MILIARD), (10**6, _MILION), (1000, _TYSIAC)):
        q, n = divmod(n, value)
        if q:
            parts.append(_form(forms, q, case) if q == 1 else f"{cardinal(q, case)} {_form(forms, q, case)}")
    if n or not parts:
        parts.append(_below1000(n, case, g))
    return " ".join(parts)


def _below1000(n: int, case: str, g: str) -> str:
    col = _CARD_COLUMN[case]
    if n == 0:
        return _CARD[0][col]
    words = []
    for part in (n - n % 100, n % 100 if n % 100 < 20 else n % 100 - n % 10, n % 10 if n % 100 >= 20 else 0):
        if part == 1:
            words.append("jeden")  # inside a compound number "jeden" does not inflect
        elif part == 2 and g == "f" and case in ("nom", "acc", "ins"):
            words.append("dwiema" if case == "ins" else "dwie")
        elif part:
            words.append(_CARD[part][col])
    return " ".join(words)


def _form(forms: dict[str, str], n: int, case: str, fraction: bool = False) -> str:
    """The noun form that goes with a number: 1 / 2-4 / 5+ in the nominative, the case's plural otherwise."""
    if fraction:
        return forms["gen_sg"]
    if n == 1:
        return forms[f"{case}_sg"]
    if case in ("nom", "acc"):
        return forms["nom_pl" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "gen_pl"]
    return forms[f"{case}_pl"]


def _adj(stem: str, ending: str) -> str:
    if stem.endswith(("g", "k")):  # drugi, drugiego
        ending = "i" + ending[1:] if ending[0] == "y" else "i" + ending if ending[0] == "e" else ending
    elif stem.endswith("c"):  # trzeci, trzecia
        ending = "i" + ending[1:] if ending[0] == "y" else "i" + ending
    return stem + ending


def ordinal(n: int, case: str = "nom", g: str = "m") -> str:
    """Only the last two parts inflect: dwa tysiące dwudziestym drugim, tysiąc osiemsetnym."""
    e = _ADJ[g][case]
    r = n % 100
    if r:
        words = [_adj(_ORD[r], e)] if r < 20 or r % 10 == 0 else [_adj(_ORD[r - r % 10], e), _adj(_ORD[r % 10], e)]
        return " ".join(([cardinal(n - r)] if n - r else []) + words)
    h = n % 1000
    if h:
        return " ".join(([cardinal(n - h)] if n - h else []) + [_adj(_ORD[h], e)])
    t = n // 1000
    return _adj(("" if t == 1 else combining(t)) + "tysięczn", e)


def combining(n: int) -> str:
    """The form of a number inside a compound: czterdziestopięcio(metrowe), stu(letni)."""
    h, r = n - n % 100, n % 100
    parts = [_COMBINING[h]] if h else []
    if r:
        parts += [_COMBINING[r]] if r < 20 or r % 10 == 0 else [_COMBINING[r - r % 10], _COMBINING[r % 10]]
    return "".join(parts)


def digits(s: str) -> str:
    return " ".join(_CARD[int(c)][0] if c != "1" else "jeden" for c in s)


# Context


def _words_around(m: re.Match) -> tuple[list[str], list[str]]:
    """Words before and after a number within its clause, skipping a range or coordination it belongs to
    (20-30, 3 a 368) and the złote of a grosze amount (39 532 złotych 20 groszy)."""
    head = re.split(r"[,;:.!?](?=\s)", m.string[: m.start()])[-1]
    prev = [w.strip(PUNCT).lower() for w in head.split()]
    while prev and re.fullmatch(r"\d[\d\s,.]*-", prev[-1]):
        prev = prev[:-1]
    while len(prev) >= 2 and prev[-1] in ("a", "i", "lub", "albo", "czy", "oraz") and re.fullmatch(r"[\d,.]+", prev[-2]):
        prev = prev[:-2]
    if prev and prev[-1] in _CURRENCY_WORDS:
        prev = prev[:-1]
        while prev and prev[-1].isdigit():
            prev = prev[:-1]
    nxt = [w.strip(PUNCT) for w in m.string[m.end():].split()]
    if m.string[m.end():m.end() + 1] in PUNCT and m.string[m.end():m.end() + 1]:
        nxt = []
    while nxt and re.fullmatch(r"-[\d,.]+", nxt[0]):
        nxt = nxt[1:]
    return prev, nxt


def _enumerated(m: re.Match) -> re.Match | None:
    """The number this one continues after a comma (140 g mąki, 130 g mąki): both share a case."""
    head = m.string[: m.start()]
    comma = re.search(r",\s*$", head)
    numbers = list(re.finditer(r"(?<![\w.,])\d+", head[: comma.start()])) if comma else []
    if numbers and len(head[numbers[-1].end(): comma.start()].split()) <= 4:
        return numbers[-1]
    return None


def _counted(word: str) -> bool:
    """A noun that a number counts: osób, metrów, złotych (as opposed to a verb or nothing after a year)."""
    return word in _COUNT_NOUNS or word in _GENITIVE_PLURALS or word.endswith(("ów", "ych", "ich"))


def _governed(prev: list[str]) -> str | None:
    """The genitive of an amount noun, a genitive verb or a negated verb right before the number."""
    w1 = prev[-1] if prev else ""
    if any(w1.startswith(stem) and w1[len(stem):] in _NOUN_ENDINGS for stem in _AMOUNT_NOUNS):
        return "gen"  # od kwoty trzystu tysięcy
    verbs = [w for w in prev[-3:] if w not in _CLITICS]
    if verbs and verbs[-1].startswith(_GENITIVE_VERBS):
        return "gen"  # potrzebowali stu czterdziestu gramów, brakuje nam trzech osób
    window = prev[-3:]
    if "nie" in window and not set(window[window.index("nie") + 1:]) & (_COPULA | {"niż", "tylko"}):
        return "gen"  # nie powinno przekraczać pięciu gramów
    return None


def _noun_case(w: str, prep: str) -> tuple[str, str]:
    """Case and gender of a noun from its ending, as seen after an ordinal: sesji, sesję, rocznik, roku."""
    loc = prep in _LOCATIVE_PREPS
    instrumental = prep in ("z", "ze", "przed", "nad", "pod")
    for ending, case, g in (("ami", "ins", "pl"), ("ach", "loc", "pl"), ("ych", "gen", "pl"), ("ów", "gen", "pl"),
                            ("iem", "ins", "m"), ("em", "ins", "m"), ("ą", "ins" if instrumental else "acc", "f"),
                            ("ę", "acc", "f"), ("a", "nom", "f"), ("i", "loc" if loc else "gen", "f"),
                            ("y", "loc" if loc else "gen", "f"), ("u", "loc" if loc else "gen", "m"), ("ie", "loc", "m"),
                            ("e", "nom", "n"), ("o", "nom", "n")):
        if w.endswith(ending):
            return case, g
    return "nom", "m"


def _context(m: re.Match, n: int, decimal: bool = False) -> tuple[str, str, str]:
    """(mode, case, gender) of a number: mode is "ord" for years, days of the month and headings, "card" otherwise."""
    if FORCE:
        return FORCE["mode"], FORCE["case"], FORCE["g"]
    prev, nxt = _words_around(m)
    w1, w2 = (["", ""] + prev)[-1], (["", ""] + prev)[-2]
    noun = nxt[0].lower() if nxt else ""
    if noun in ("roku", "rok", "rokiem"):
        return "ord", {"roku": "loc" if w1 in _LOCATIVE_PREPS else "gen", "rok": "nom", "rokiem": "ins"}[noun], "m"
    if w1 == "dnia":  # z dnia 1.
        return "ord", "gen", "m"
    if w1 in ("roku", "rok"):
        return "ord", "nom" if w1 == "rok" else "loc" if w2 in _LOCATIVE_PREPS else "gen", "m"
    if noun in _MONTHS_GEN and 1 <= n <= 31:
        return "ord", "nom" if w1 in _COPULA else "gen", "m"  # od pierwszego kwietnia; to jest drugi maja
    if w1 in _MONTHS and 1000 <= n <= 2100:
        return "ord", "gen", "m"
    if w1 in ("w", "we") and 1000 <= n <= 2100 and noun not in _GENITIVE_PLURALS:
        return "ord", "loc", "m"
    if w1 in _YEAR_PREPS and 1900 <= n <= 2100 and not _counted(noun):
        return "ord", _YEAR_PREPS[w1], "m"  # od dwa tysiące dwudziestego do dwa tysiące dwudziestego drugiego
    if w1.startswith(_ORDINAL_HEADINGS) and n < 100:  # w punkcie drugim, z paragrafem czternastym
        return "ord", _noun_case(w1, w2)[0], "m"
    enumerated = _enumerated(m)
    case = (_PREP_CASE.get(w1) or _COMPLEX_PREPS.get(w1) == w2 and "gen" or not decimal and _governed(prev)
            or enumerated and _context(enumerated, 0)[1] or "nom")
    if noun.endswith(("ami", "mi")) and len(noun) > 4:
        case = "ins"
    elif noun.endswith("ach") and len(noun) > 4:
        case = "loc"
    elif case in ("nom", "acc") and n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) and _counted(noun):
        case = "gen"  # trzech osób: the noun is genitive, so the number is too
    g = "f" if noun.endswith("a") and n == 1 or noun.startswith(_FEMININE_PLURAL) else "n" if noun == "euro" else "m"
    return "card", case, g


def _integer(i: str, grouped: bool, m: re.Match, case: str | None = None, g: str | None = None) -> str:
    n = int(i)
    if spoken_digits(i, grouped, 7):
        return digits(i)
    mode, c, gender = _context(m, n)
    if mode == "ord" and case is None:
        return ordinal(n, c, gender)
    return cardinal(n, case or c, g or gender)


def _amount(s: str, forms: dict[str, str], m: re.Match) -> str:
    """A number with its noun in the right case: 524 km -> pięćset dwadzieścia cztery kilometry."""
    i, f, _ = parse(s, GROUP, DECIMAL)
    case = _context(m, 0, decimal=f is not None)[1]
    g = "f" if forms["nom_sg"].endswith("a") else "n" if forms["nom_sg"] == "euro" else "m"
    return f"{read_number(s, m, case=case, g=g)} {_form(forms, int(i or 0), case, f is not None)}"


def BY(m: re.Match) -> str:
    """"x" in a document number (numerem 3x38) stands for a slash: przez; otherwise na (1920x1080)."""
    return "przez" if re.search(r"\b(?:numer\w*|nr)\s+\d+$", m.string[: m.start()]) else "na"


# Readings


def read_date(m: re.Match) -> str | None:
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    words = f"{ordinal(d, 'nom' if before(m) in _COPULA else 'gen')} {_MONTHS_GEN[mo - 1]} {ordinal(y, 'gen')}"
    if not m.group(4):
        return words
    return words + " roku" + ("." if re.match(r"\s*$|\s+[A-ZĄĆĘŁŃÓŚŹŻ]", m.string[m.end():]) else "")


def read_phone(m: re.Match) -> str:
    return " ".join(cardinal(int(g)) for g in m.group(0).split())


def read_time(m: re.Match) -> str | None:
    """o osiemnastej trzydzieści: the hour is a feminine ordinal (godzina) in the case of the word before."""
    h, mm, w1 = int(m["h"]), m["mm"], before(m)
    if not mm or h > 24 or int(mm) > 59 or m["sep"] == "." and mm not in ("00", "15", "30", "45") and w1 not in _TIME_WORDS:
        return None
    case = ("loc" if w1 in ("o", "godzinie", "po") else "gen" if w1 in ("do", "od", "około", "koło") else
            "ins" if w1 in ("przed", "między") else "acc" if w1 in ("na", "za") else "nom")
    return ordinal(h, case, "f") + (f" {digits(mm) if mm[0] == '0' else cardinal(int(mm))}" if int(mm) else "")


def read_money(s: str, scale: str | None, currency: str, m: re.Match) -> str:
    forms = CURRENCIES[currency]
    i, f, _ = parse(s, GROUP, DECIMAL)
    if scale:
        return f"{_amount(s, SCALES[scale], m)} {forms['gen_pl']}"
    if f is not None and len(f) == 2 and forms is _ZLOTY:  # dwanaście złotych dziewięćdziesiąt dziewięć groszy
        case = _context(m, 0)[1]
        return f"{_amount(i, forms, m)} {cardinal(int(f), case)} {_form(_GROSZ, int(f), case)}"
    return _amount(s, forms, m)


def read_scale(s: str, scale: str, m: re.Match) -> str:
    return _amount(s, SCALES[scale], m)


def read_percent(s: str, suffix: str, m: re.Match) -> str:
    return _amount(s, _PROCENT, m) + (f" {suffix}" if suffix else "")


def read_unit(s: str, unit: str, m: re.Match) -> str:
    return _amount(s, UNITS[unit], m)


def read_ordinal(m: re.Match) -> str | None:
    """"29. sesji": an ordinal agreeing with the noun after it, the decade of "lat 70.", or a heading (Punkt 3.)."""
    n, w1 = int(m.group(1)), before(m)
    head = m.string[: m.start()].split()
    if w1 in ("i", "a", "lub", "oraz", "albo") and len(head) > 1 and re.fullmatch(r"\d+\.", head[-2]):
        w1 = before(m, 3)  # lat 50. i 60.
    nxt = (m.string[m.end():].split() or [""])[0].strip(PUNCT)
    end = "." if re.match(r"\s*$|\s+[A-ZĄĆĘŁŃÓŚŹŻ]", m.string[m.end():]) else ""  # the dot also ends the sentence
    if w1 in ("lata", "lat", "latach", "latami", "roczników", "rocznikach"):
        return ordinal(n, {"lata": "nom", "latach": "loc", "latami": "ins", "rocznikach": "loc"}.get(w1, "gen"), "pl") + end
    if w1.startswith(_HEADINGS) and n < 100:  # Punkt 3., w punkcie 5.; but rozdział 614. is a code
        return ordinal(n, _noun_case(w1, before(m, 2))[0]) + end
    if not nxt[:1].islower():
        return None
    case, g = _noun_case(nxt.lower(), w1)
    return ordinal(n, case, g)


def read_compound(m: re.Match) -> str:
    """lat 70-tych: siedemdziesiątych; 45-metrowe: czterdziestopięciometrowe."""
    i, f, _ = parse(m.group(1), GROUP, DECIMAL)
    suffix = m.group(2)
    if suffix in _ORDINAL_SUFFIXES and f is None:
        e = _ORDINAL_SUFFIXES[suffix]
        g, case = next(((g, c) for g in ("m", "f", "n", "pl") for c in CASES if _ADJ[g][c] == e), ("m", "nom"))
        return ordinal(int(i), case, g)
    if f is None:
        return combining(int(i)) + suffix
    return f"{read_number(m.group(1), m)}-{suffix}"


def read_code_digits(s: str) -> str:
    return digits(s) if s[0] == "0" and len(s) > 1 or len(s) >= 7 else cardinal(int(s))


def read_number(s: str, m: re.Match, zero: bool = True, case: str | None = None, g: str | None = None) -> str:
    i, f, grouped = parse(s, GROUP, DECIMAL)
    if f is None:
        return _integer(i, grouped, m, case, g)
    if s[-len(f) - 1] == "." and (len(f) == 2 or f == "0"):  # a dotted code, not a decimal: 801.04, 4.0
        return f"{cardinal(int(i or 0))} {digits(f) if f[0] == '0' else cardinal(int(f))}"
    case = case or _context(m, 0, decimal=True)[1]
    if i == "1" and f == "5" and zero:
        return "półtora"
    head = cardinal(int(i or 0), case, "m") + " " if zero else ""
    if f == "5":
        return head + "i pół"
    zeros = ["zero"] * (len(f) - len(f.lstrip("0")))
    return head + "przecinek " + " ".join(zeros + ([cardinal(int(f), case)] if f.strip("0") else []))
