"""Tests for the rule-based transcript verbalizer."""

import pytest

from echo.preprocessing import Span, verbalize, verbalize_spans
from echo.preprocessing.characters import normalize_characters

CASES = [
    ("en", "The meeting is at 10:30, about 20-25 minutes.", "The meeting is at ten thirty, about twenty to twenty-five minutes."),
    ("en", "He was born on March 3, 1990, and paid $1.5 million.", "He was born on March third, nineteen ninety, and paid one point five million dollars."),
    ("en", "About 1999 people, a $5 bill and SCP-606's.", "About one thousand nine hundred ninety-nine people, a five dollar bill and SCP six oh six's."),
    ("de", "Am 1. Januar 2020 um 18:30 Uhr kostete es 12,99 €.", "Am ersten Januar zweitausendzwanzig um achtzehn Uhr dreißig kostete es zwölf Euro neunundneunzig."),
    ("de", "Er wird in 2. und 3. Lesung beraten.", "Er wird in zweiter und dritter Lesung beraten."),
    ("de", "Nur 1 Person kam, 1945 Einwohner, im Jahr 1945.", "Nur eine Person kam, eintausendneunhundertfünfundvierzig Einwohner, im Jahr neunzehnhundertfünfundvierzig."),
    ("fr", "Le 1er janvier 2020 à 18h30, 21 personnes, 3,50 €.", "Le premier janvier deux mille vingt à dix-huit heures trente, vingt et une personnes, trois euros cinquante."),
    ("fr", "Au XXe siècle, au 06 12 34 56 78.", "Au vingtième siècle, au zéro six douze trente-quatre cinquante-six soixante-dix-huit."),
    ("es", "Había 21 personas y 200 casas a la 1 de la tarde.", "Había veintiuna personas y doscientas casas a la una de la tarde."),
    ("es", "Más de 500.000 personas, el 51% y $5,000 dólares.", "Más de quinientas mil personas, el cincuenta y uno por ciento y cinco mil dólares."),
    ("pl", "Urodził się w 1985 roku, 1 stycznia o 18:30.", "Urodził się w tysiąc dziewięćset osiemdziesiątym piątym roku, pierwszego stycznia o osiemnastej trzydzieści."),
    ("pl", "Otwieram 55. sesję, około 20-30 minut, z 3 kolegami.", "Otwieram pięćdziesiątą piątą sesję, około dwudziestu-trzydziestu minut, z trzema kolegami."),
    ("pl", "To 524 km, 11 342 zł i 45-metrowe mieszkania z lat 70-tych.", "To pięćset dwadzieścia cztery kilometry, jedenaście tysięcy trzysta czterdzieści dwa złote i czterdziestopięciometrowe mieszkania z lat siedemdziesiątych."),
    ("pl", "Od kwoty 300 tysięcy, nie powinno przekraczać 5 g.", "Od kwoty trzystu tysięcy, nie powinno przekraczać pięciu gramów."),
    ("pt", "Em 2022, 22 mil pessoas pagaram R$ 29,90 às 21h30.", "Em dois mil e vinte e dois, vinte e duas mil pessoas pagaram vinte e nove reais e noventa centavos às vinte e uma e trinta."),
    ("pt", "No 1º de julho, 95% e 1,6 bilhões; artigo 80º, 13ª edição.", "No primeiro de julho, noventa e cinco por cento e um vírgula seis bilhões; artigo oitenta, décima terceira edição."),
    ("pt", "São 2 da tarde, a Covid-19 e o 4x4 de 2016-2022.", "São duas da tarde, a Covid dezenove e o quatro por quatro de dois mil e dezesseis a dois mil e vinte e dois."),
    ("it", "Nel 1984 erano 21 persone, il 12,4% e 15.000 euro alle 21.30.", "Nel millenovecentottantaquattro erano ventuno persone, il dodici virgola quattro per cento e quindicimila euro alle ventuno e trenta."),
    ("it", "Il 21esimo gol, l'80% e la CO2 nel 2013-2014, versione 2.79.", "Il ventunesimo gol, l'ottanta per cento e la CO due nel duemilatredici-duemilaquattordici, versione due punto settantanove."),
    ("it", "Il 1 maggio a 180 gradi, 74 mila di reddito e 6 milioni di euro.", "Il primo maggio a centottanta gradi, settantaquattromila di reddito e sei milioni di euro."),
]


@pytest.mark.parametrize("lang, text, expected", CASES)
def test_verbalize(lang: str, text: str, expected: str) -> None:
    assert verbalize(text, lang) == expected


def test_unsupported_language() -> None:
    with pytest.raises(ValueError):
        verbalize("1", "nl")


@pytest.mark.parametrize("lang, text, expected", CASES)
def test_spans_join_to_verbalize(lang: str, text: str, expected: str) -> None:
    assert "".join(p.default if isinstance(p, Span) else p for p in verbalize_spans(text, lang)) == expected


@pytest.mark.parametrize("lang, text, original, alternative", [
    ("en", "Built in 1800 by them.", "1800", "one thousand eight hundred"),
    ("en", "The ratio was 0.83 overall.", "0.83", "point eight three"),
    ("en", "Music until 2021, so.", "2021", "two thousand and twenty-one"),
    ("en", "The 120 cell.", "120", "one hundred and twenty"),
    ("pl", "W 2013 roku było ich 480.", "2013", "dwa tysiące trzynaście"),
    ("es", "Hay 480 líneas y cuesta $5.", "$5", "cinco pesos"),
    ("de", "Das war 1990 so.", "1990", "eintausendneunhundertneunzig"),
    ("de", "Es sind 3,14 Prozent.", "3,14", "drei Komma vierzehn"),
    ("fr", "En 1789, tout change.", "1789", "dix-sept cent quatre-vingt-neuf"),
    ("fr", "Elle a 21 ans.", "21", "vingt et une"),
    ("pt", "Custa R$ 29,90 hoje.", "R$ 29,90", "vinte e nove e noventa"),
    ("pt", "Começa às 14h30.", "14h30", "catorze e meia"),
    ("it", "Pesa 0,83 chili.", "0,83", "zero virgola otto tre"),
    ("it", "Iniziamo alle 21.30 stasera.", "21.30", "ventuno trenta"),
])
def test_span_alternatives(lang: str, text: str, original: str, alternative: str) -> None:
    span = next(p for p in verbalize_spans(text, lang) if isinstance(p, Span) and p.original == original)
    assert alternative in span.alternatives and span.default not in span.alternatives


@pytest.mark.parametrize("text, expected", [
    (" Er sagte „ja” … und «nein» ", 'Er sagte "ja" ... und "nein"'),
    ("Wokulski. — A ma pan? — spytał", "Wokulski. - A ma pan? - spytał"),
    ("Szczurek--Żelazko, 1990–2000, it’s ok", "Szczurek-Żelazko, 1990-2000, it's ok"),
    ("nawoływanie – — Jeżeli to pan -- tak ---- nie", "nawoływanie - Jeżeli to pan - tak - nie"),
    ("Bonjour ,  ça va ? Oui !", "Bonjour, ça va? Oui!"),
])
def test_normalize_characters(text: str, expected: str) -> None:
    assert normalize_characters(text) == expected
