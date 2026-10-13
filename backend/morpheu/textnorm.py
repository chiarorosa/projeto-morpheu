"""Spell out what TTS models misread: numbers, money, percentages, ordinals, times, abbreviations.

Models trained mostly on Chinese/English text read "R$ 1.234,50" or "14h30" badly in
Portuguese, so the text is written the way it should be spoken before it reaches them.
"""

import re

from num2words import num2words

# Display language -> (num2words code, decimal separator, thousands separator)
_LANG = {
    "Portuguese": ("pt_BR", ",", "."),
    "Spanish": ("es", ",", "."),
    "French": ("fr", ",", " "),
    "German": ("de", ",", "."),
    "Italian": ("it", ",", "."),
    "Dutch": ("nl", ",", "."),
    "Polish": ("pl", ",", " "),
    "Russian": ("ru", ",", " "),
    "Turkish": ("tr", ",", "."),
    "English": ("en", ".", ","),
    "Japanese": ("ja", ".", ","),
    "Korean": ("ko", ".", ","),
}

_PT_WORDS = {
    "point": "vírgula", "percent": "por cento", "and": "e",
    "currencies": {"R$": ("real", "reais"), "US$": ("dólar", "dólares"), "$": ("dólar", "dólares"),
                   "€": ("euro", "euros"), "£": ("libra", "libras")},
    "cents": ("centavo", "centavos"),
    "hours": ("hora", "horas"), "minutes": ("minuto", "minutos"),
    "units": {"km": "quilômetros", "kg": "quilos", "g": "gramas", "m": "metros", "cm": "centímetros",
              "mm": "milímetros", "l": "litros", "ml": "mililitros", "h": "horas", "min": "minutos",
              "s": "segundos", "km/h": "quilômetros por hora", "°C": "graus Celsius", "°": "graus"},
    "abbrev": {"Sr.": "senhor", "Sra.": "senhora", "Srta.": "senhorita", "Dr.": "doutor", "Dra.": "doutora",
               "Prof.": "professor", "Profa.": "professora", "etc.": "etcétera", "p.ex.": "por exemplo",
               "Av.": "avenida", "nº": "número", "n.º": "número", "tel.": "telefone", "pág.": "página",
               "vs.": "versus", "aprox.": "aproximadamente"},
}

_EN_WORDS = {
    "point": "point", "percent": "percent", "and": "and",
    "currencies": {"US$": ("dollar", "dollars"), "$": ("dollar", "dollars"), "€": ("euro", "euros"),
                   "£": ("pound", "pounds"), "R$": ("real", "reais")},
    "cents": ("cent", "cents"),
    "hours": ("hour", "hours"), "minutes": ("minute", "minutes"),
    "units": {"km": "kilometers", "kg": "kilograms", "g": "grams", "m": "meters", "cm": "centimeters",
              "mm": "millimeters", "l": "liters", "ml": "milliliters", "h": "hours", "min": "minutes",
              "s": "seconds", "km/h": "kilometers per hour", "mph": "miles per hour", "°C": "degrees Celsius",
              "°F": "degrees Fahrenheit", "°": "degrees"},
    "abbrev": {"Mr.": "mister", "Mrs.": "missus", "Ms.": "miz", "Dr.": "doctor", "Prof.": "professor",
               "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is", "vs.": "versus", "St.": "street",
               "No.": "number"},
}

_ES_WORDS = {
    **_PT_WORDS,
    "point": "coma", "percent": "por ciento", "and": "y",
    "currencies": {"$": ("dólar", "dólares"), "US$": ("dólar", "dólares"), "€": ("euro", "euros"),
                   "R$": ("real", "reales"), "£": ("libra", "libras")},
    "cents": ("centavo", "centavos"),
    "hours": ("hora", "horas"), "minutes": ("minuto", "minutos"),
    "units": {"km": "kilómetros", "kg": "kilos", "m": "metros", "cm": "centímetros", "l": "litros",
              "h": "horas", "min": "minutos", "s": "segundos", "°C": "grados Celsius", "°": "grados"},
    "abbrev": {"Sr.": "señor", "Sra.": "señora", "Dr.": "doctor", "Dra.": "doctora", "etc.": "etcétera"},
}

_WORDS = {"pt_BR": _PT_WORDS, "en": _EN_WORDS, "es": _ES_WORDS}

_PT_ORDINAL_FEM = {"o": "o", "º": "o", "a": "a", "ª": "a"}


def _language_profile(language: str):
    return _LANG.get(language or "", None)


def _to_number(raw: str, dec: str, thou: str) -> float | int | None:
    s = raw
    if thou and thou != " ":
        # A lone separator followed by exactly three digits is a thousands group ("1.000"),
        # otherwise it is a decimal point in the other convention.
        if dec in s or re.fullmatch(r"\d{1,3}(\%s\d{3})+" % thou, s):
            s = s.replace(thou, "")
    s = s.replace(" ", "")
    if dec != ".":
        s = s.replace(dec, ".")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def _n2w(n: int, code: str, **kw) -> str:
    # num2words writes "mil, duzentos" / "one thousand, two hundred"; the comma makes models pause.
    return num2words(n, lang=code, **kw).replace(",", "")


_PT_FEMININE_SWAP = {"um": "uma", "dois": "duas", "duzentos": "duzentas", "trezentos": "trezentas",
                     "quatrocentos": "quatrocentas", "quinhentos": "quinhentas", "seiscentos": "seiscentas",
                     "setecentos": "setecentas", "oitocentos": "oitocentas", "novecentos": "novecentas"}
_PT_MASCULINE_IN_A = {"dia", "mapa", "problema", "sistema", "tema", "programa", "clima", "idioma", "planeta",
                      "cometa", "poema", "telefonema", "drama", "pijama", "esquema", "diploma", "cinema",
                      "dilema", "lema", "enigma", "prisma", "trema", "aroma", "fantasma", "guarda", "atleta",
                      "motorista", "artista", "jornalista", "dentista", "turista", "colega", "papa", "panda"}
_PT_FEMININE_OTHER = {"vez", "mulher", "mãe", "mães", "noite", "tarde", "parte", "gente", "fome", "sede", "dor", "flor", "mão",
                      "foto", "moto", "lei", "rede", "chave", "frase", "classe", "fase", "base", "mente", "morte",
                      "sorte", "árvore", "ponte", "fonte", "febre", "neve", "nave", "ave", "cor", "série",
                      "espécie", "tribo", "libido", "pele", "torre", "carne", "arte", "corte", "ordem", "nuvem",
                      "imagem", "viagem", "garagem", "mensagem", "margem", "origem", "vertigem", "paisagem"}


def _singular_pt(word: str) -> str:
    w = word.lower()
    for plural, singular in (("ões", "ão"), ("ães", "ão"), ("ens", "em"), ("res", "r"), ("zes", "z"), ("es", "e"), ("s", "")):
        if w.endswith(plural) and len(w) > len(plural) + 1:
            candidate = w[: -len(plural)] + singular
            if plural in ("es",) and not candidate.endswith(("e",)):
                continue
            return candidate
    return w


def _is_feminine_pt(word: str) -> bool:
    """Best-effort grammatical gender of a Portuguese noun, enough for number agreement."""
    w = _singular_pt(word)
    if w in _PT_MASCULINE_IN_A:
        return False
    if w in _PT_FEMININE_OTHER or word.lower() in _PT_FEMININE_OTHER:
        return True
    return w.endswith(("a", "ção", "são", "dade", "tude", "agem", "ice"))


def _feminine_pt(spoken: str) -> str:
    """Numbers agree with feminine nouns in Portuguese: uma hora, duas dúzias, duzentas pessoas."""
    return " ".join(_PT_FEMININE_SWAP.get(w, w) for w in spoken.split())


def _say(n, code: str, words: dict | None) -> str:
    if isinstance(n, float) and not n.is_integer():
        whole, frac = f"{n}".split(".")
        point = words["point"] if words else "point"
        # Read decimals digit by digit after a leading zero ("3,05" -> "três vírgula zero cinco").
        frac_words = (" ".join(_n2w(int(d), code) for d in frac) if frac.startswith("0")
                      else _n2w(int(frac), code))
        return f"{_n2w(int(whole), code)} {point} {frac_words}"
    return _n2w(int(n), code)


def _plural(pair: tuple[str, str], n) -> str:
    return pair[0] if n == 1 else pair[1]


def normalize_text(text: str, language: str) -> str:
    """Return text with digits, symbols and common abbreviations spelled out for `language`."""
    profile = _language_profile(language)
    if profile is None:
        return re.sub(r"\s+", " ", text).strip()
    code, dec, thou = profile
    words = _WORDS.get(code)
    num = r"\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?" if thou != " " else r"\d+(?:\s\d{3})*(?:[.,]\d+)?"

    if words:
        for abbr, full in sorted(words["abbrev"].items(), key=lambda kv: -len(kv[0])):
            text = re.sub(r"(?<!\w)" + re.escape(abbr) + r"(?=\s|$)", full, text)

        # Money: "R$ 1.234,50", "$5", "€ 10"
        cur = "|".join(re.escape(c) for c in sorted(words["currencies"], key=len, reverse=True))

        def money(m):
            value = _to_number(m.group(2), dec, thou)
            if value is None:
                return m.group(0)
            unit = words["currencies"][m.group(1)]
            whole = int(value)
            cents = round((float(value) - whole) * 100)
            out = f"{_n2w(whole, code)} {_plural(unit, whole)}"
            if cents:
                out += f" {words['and']} {_n2w(cents, code)} {_plural(words['cents'], cents)}"
            return out

        text = re.sub(rf"({cur})\s?({num})", money, text)

        # Percent
        text = re.sub(rf"({num})\s?%", lambda m: (_say(_to_number(m.group(1), dec, thou), code, words)
                                                  + " " + words["percent"]) if _to_number(m.group(1), dec, thou) is not None else m.group(0), text)

        # Times: 14h30, 14h, 14:30
        def clock(m):
            h, mnt = int(m.group(1)), int(m.group(2) or 0)
            if h > 24 or mnt > 59:
                return m.group(0)
            if code == "en":
                return f"{_n2w(h, code)} " + (f"{_n2w(mnt, code)}" if mnt >= 10 else f"oh {_n2w(mnt, code)}" if mnt else "o'clock")
            hours = _feminine_pt(_n2w(h, code)) if code == "pt_BR" else _n2w(h, code)
            out = f"{hours} {_plural(words['hours'], h)}"
            if mnt:
                out += f" {words['and']} {_n2w(mnt, code)} {_plural(words['minutes'], mnt)}"
            return out

        text = re.sub(r"\b(\d{1,2})h(\d{2})?\b", clock, text)
        text = re.sub(r"\b(\d{1,2}):(\d{2})(?!\d)", clock, text)

        # Units right after a number: "3 kg", "100km/h"
        units = "|".join(re.escape(u) for u in sorted(words["units"], key=len, reverse=True))
        text = re.sub(rf"({num})\s?({units})(?![\w/])",
                      lambda m: f"{m.group(1)} {words['units'][m.group(2)]}", text)

    if code == "pt_BR":
        # Ordinals: 1º, 2ª, 3o
        def ordinal(m):
            n = int(m.group(1))
            spoken = _n2w(n, code, to="ordinal")
            if _PT_ORDINAL_FEM.get(m.group(2)) == "a":
                spoken = " ".join(w[:-1] + "a" if w.endswith("o") else w for w in spoken.split())
            return spoken

        text = re.sub(r"\b(\d+)\s?([ºª])", ordinal, text)
    elif code == "en":
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _n2w(int(m.group(1)), code, to="ordinal"), text)

    # Remaining plain numbers.
    def plain(m):
        value = _to_number(m.group(0), dec, thou)
        if value is None:
            return m.group(0)
        spoken = _say(value, code, words)
        if code == "pt_BR" and not isinstance(value, float):
            following = re.match(r"\s+([^\W\d_]+)", m.string[m.end():])
            if following and _is_feminine_pt(following.group(1)):
                spoken = _feminine_pt(spoken)
        return spoken

    text = re.sub(num, plain, text)
    return re.sub(r"\s+", " ", text).strip()


_TAG_RE = re.compile(r"\[[^\]\n]{1,40}\]|<[^>\n]{1,40}>")


def strip_expression_tags(text: str) -> str:
    """Drop [laugh]-style tags for models that would read them aloud."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()
