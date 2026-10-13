from morpheu.chunker import chunk_text, split_sentences
from morpheu.textnorm import normalize_text, strip_expression_tags


def pt(text):
    return normalize_text(text, "Portuguese")


def test_money_and_hours_in_portuguese():
    assert pt("Comprei por R$ 1.234,50 às 14h30.") == (
        "Comprei por mil duzentos e trinta e quatro reais e cinquenta centavos às catorze horas e trinta minutos.")


def test_hours_agree_with_feminine_noun():
    assert pt("Chego às 1:05 ou às 22h.") == "Chego às uma hora e cinco minutos ou às vinte e duas horas."


def test_percent_ordinals_and_thousands():
    assert pt("Foram 15% a mais, o 1º e a 2ª vez, 1.000.000 de vezes.") == (
        "Foram quinze por cento a mais, o primeiro e a segunda vez, um milhão de vezes.")


def test_decimals_units_and_abbreviations():
    assert pt("A Dra. Ana correu 3,5 km.") == "A doutora Ana correu três vírgula cinco quilômetros."


def test_english_uses_its_own_separators():
    assert normalize_text("It cost $1,234.50 at 5:30.", "English") == (
        "It cost one thousand two hundred and thirty-four dollars and fifty cents at five thirty.")


def test_unknown_language_only_collapses_spaces():
    assert normalize_text("Tem  3 gatos", "Auto") == "Tem 3 gatos"


def test_expression_tags_are_removed():
    assert strip_expression_tags("Olá [laugh] tudo <breath> bem?") == "Olá tudo bem?"


def test_split_sentences_keeps_punctuation():
    assert split_sentences("Oi. Tudo bem? Sim! Claro") == ["Oi.", "Tudo bem?", "Sim!", "Claro"]


def test_chunks_respect_limit_and_sentence_boundaries():
    text = "Primeira frase curta. Segunda frase um pouco maior que a primeira. Terceira."
    chunks = chunk_text(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == text
    assert chunks[0] == "Primeira frase curta."


def test_long_sentence_is_split_at_commas_then_spaces():
    text = "um, " * 30 + "fim."
    chunks = chunk_text(text, 25)
    assert all(len(c) <= 25 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_numbers_agree_with_feminine_nouns():
    assert pt("2 dúzias, 21 pessoas, 200 casas, 2.000 pessoas, 1 vez, 2 mães") == (
        "duas dúzias, vinte e uma pessoas, duzentas casas, duas mil pessoas, uma vez, duas mães")


def test_numbers_stay_masculine_before_masculine_nouns():
    assert pt("3 quilos, 1 dia, 2 problemas, 2 pães, 2.000.000 de pessoas") == (
        "três quilos, um dia, dois problemas, dois pães, dois milhões de pessoas")
