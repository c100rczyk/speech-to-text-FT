"""Run with: python -m pytest evaluation/tests"""
import pytest

from textnorm_pl import CARDINAL_WORDS, NUMERALS, normalize_text, number_words

CASES = [
    # numerals in any case become digits
    ("Od trzech dni czekam na pięćset złotych, dwa razy dzwoniłem.", "od 3 dni czekam na 500 zł 2 razy dzwoniłem"),
    ("Od 3 dni czekam na 500 zł, 2 razy dzwoniłem.", "od 3 dni czekam na 500 zł 2 razy dzwoniłem"),
    ("Sto czterdzieści osiem głosów na dziewięćdziesiąt dwa.", "148 głosów na 92"),
    ("po dwudziestu jeden dniach", "po 21 dniach"),
    ("z trzystu pięćdziesięcioma złotymi", "z 350 zł"),
    ("tysiąc dwieście", "1200"),
    ("sto tysięcy", "100000"),
    ("dwa miliony pięćset tysięcy sto", "2500100"),
    ("zero przecinek pięć", "0,5"),
    # numbers that cannot be one number stay apart
    ("pięć sześć siedem", "5 6 7"),
    ("jeden dwadzieścia", "1 20"),
    ("dwadzieścia trzydzieści", "20 30"),
    ("trzy, cztery", "3 4"),
    ("tysiąc tysiąc", "1000 1000"),
    # ordinals, also compound ones and years
    ("dwudziestego piątego maja", "25 maja"),
    ("w tysiąc dziewięćset dziewięćdziesiątym ósmym roku", "w 1998 roku"),
    ("W dwa tysiące dwudziestym szóstym roku.", "w 2026 roku"),
    ("w dwutysięcznym dwudziestym piątym", "w 2025"),
    ("o siedemnastej", "o 17"),
    ("drugi raz, trzecia doba", "2 raz 3 doba"),
    ("dwudziestego pięć", "20 5"),
    # units after a number, percent everywhere
    ("pięćset mililitrów", "500 ml"),
    ("kilka kilogramów", "kilka kilogramów"),
    ("7,8%", "7,8 procent"),
    ("siedem przecinek osiem procent", "7,8 procent"),
    ("kilka procentów", "kilka procent"),
    ("sto dwadzieścia milimetrów słupka rtęci", "120 mmhg"),
    ("120 mmHg", "120 mmhg"),
    ("21,5°C", "21,5 stopni celsjusza"),
    ("dwadzieścia jeden stopni Celsjusza", "21 stopni celsjusza"),
    ("w dwóch litrach", "w 2 l"),
    # glued tokens are split, other text is only lowercased and stripped of punctuation
    ("5km", "5 km"),
    ("Dzień dobry, proszę usiąść.", "dzień dobry proszę usiąść"),
    ("spółkę komandytowo-akcyjną", "spółkę komandytowo akcyjną"),
    ("o 17:00, potem o 8:30", "o 17:00 potem o 8:30"),
    ("pół etatu, półtora roku", "0,5 etatu 1,5 roku"),
    # a multiplier after digits, also abbreviated; a fraction with a multiplier
    ("40 tysięcy ludzi", "40000 ludzi"),
    ("2 mln zł i 15 tys. osób", "2000000 zł i 15000 osób"),
    ("1,5 miliona, półtora miliona, pół miliona", "1500000 1500000 500000"),
    ("tysiąc dwa tysiące", "1000 2000"),
    # an ordinal unit does not continue a cardinal ten
    ("za dwadzieścia ósma", "za 20 8"),
    ("dwadzieścia pierwszych minut", "20 1 minut"),
    ("dwudziesta ósma, sto pierwszy", "28 101"),
    # units with "/" and "na", a column of mercury, halves
    ("152 km/h", "152 km/h"),
    ("sto pięćdziesiąt dwa kilometry na godzinę", "152 km/h"),
    ("czterdzieści pięć złotych na godzinę, 6,2 µg/l", "45 zł/h 6,2 µg/l"),
    ("sześćdziesiąt litrów na minutę", "60 l/min"),
    ("90 na minutę", "90 na minutę"),
    ("szesnaście milimetrów słupa rtęci", "16 mmhg"),
    ("osiem i pół litra, art. 5 i 6", "8,5 l art 5 i 6"),
    ("50 mcg, 50 \u03bcg, 10 j., 10 jednostek, 5 j.m.", "50 µg 50 µg 10 j. 10 j. 5 j.m."),
    ("1200 złotych", "1200 zł"),
    ("mapa w skali 1:1000 o 17:30", "mapa w skali 1:1000 o 17:30"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_normalize_text(text, expected):
    assert normalize_text(text) == expected


def test_words_and_digits_meet():
    pairs = [("Działka sto trzydzieści na osiemdziesiąt pięć metrów.", "Działka 130 na 85 metrów."),
             ("Dwadzieścia tysięcy złotych zadatku od czterech tygodni.", "20 tysięcy zł zadatku od 4 tygodni."),
             ("Odsetki ustawowe siedem przecinek osiem procent.", "Odsetki ustawowe 7,8%."),
             ("Niecałe czterdzieści tysięcy ludzi.", "Niecałe 40 tysięcy ludzi."),
             ("Za dwadzieścia ósma.", "Za 20 ósma."),
             ("Dziesięć złotych na kilogram towaru.", "10 zł na kilogram towaru."),
             ("Półtorej jednostki na osobę.", "1,5 jednostki na osobę."),
             ("Dwanaście i pół.", "12,5.")]
    for spoken, written in pairs:
        assert normalize_text(spoken) == normalize_text(written)


def test_numeral_table():
    assert NUMERALS["osiemnastu"] == (18, "t", False)
    assert NUMERALS["dziewięćdziesiątym"] == (90, "d", True)
    assert NUMERALS["drugiej"] == (2, "u", True) and NUMERALS["trzecią"] == (3, "u", True)
    assert "jeden" not in CARDINAL_WORDS and "trzech" in CARDINAL_WORDS and "piąty" not in CARDINAL_WORDS
    assert number_words("Od trzech dni, po jednej rozprawie, piąty raz.") == ["trzech"]


def test_numbers_of_any_size_spelled_by_the_grammar():
    """Every number below 10000 and a sample of larger ones, spelled in words, come back as the same digits."""
    units = "zero jeden dwa trzy cztery pięć sześć siedem osiem dziewięć".split()
    teens = ("dziesięć jedenaście dwanaście trzynaście czternaście piętnaście szesnaście siedemnaście osiemnaście "
             "dziewiętnaście").split()
    tens = "dwadzieścia trzydzieści czterdzieści pięćdziesiąt sześćdziesiąt siedemdziesiąt osiemdziesiąt dziewięćdziesiąt".split()
    hundreds = "sto dwieście trzysta czterysta pięćset sześćset siedemset osiemset dziewięćset".split()

    def below_1000(n: int) -> list[str]:
        words = [hundreds[n // 100 - 1]] if n >= 100 else []
        n %= 100
        if 10 <= n < 20:
            return words + [teens[n - 10]]
        return words + ([tens[n // 10 - 2]] if n >= 20 else []) + ([units[n % 10]] if n % 10 else [])

    def spell(n: int) -> str:
        if n == 0:
            return "zero"
        words = []
        for size, forms in ((10 ** 6, ("milion", "miliony", "milionów")), (1000, ("tysiąc", "tysiące", "tysięcy"))):
            k, n = divmod(n, size)
            if k:
                few = k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14)
                words += ([] if k == 1 else below_1000(k)) + [forms[0] if k == 1 else forms[1] if few else forms[2]]
        return " ".join(words + below_1000(n))

    for n in [*range(0, 10000), *range(10000, 10 ** 9, 7919 * 13)]:
        assert normalize_text(spell(n)) == str(n), spell(n)
