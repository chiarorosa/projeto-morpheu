import numpy as np

from morpheu import quality
from morpheu.asr import attach_punctuation, is_transcript_mismatch, reference_error
from morpheu.refwindow import Word, best_sentence_window, last_pause_cut, sentence_end_indices, word_gap_window


def words(spec):
    """spec: list of (text, start, end)."""
    return [Word(t, s, e) for t, s, e in spec]


SENTENCES = words([
    ("Olá", 0.2, 0.5), ("pessoal.", 0.6, 1.2),
    ("Hoje", 1.8, 2.1), ("o", 2.15, 2.2), ("dia", 2.25, 2.6), ("está", 2.7, 3.0), ("lindo", 3.05, 3.5),
    ("e", 3.55, 3.6), ("ensolarado.", 3.65, 4.6),
    ("Vamos", 5.6, 5.9), ("à", 6.6, 6.7), ("praia", 7.5, 7.9), ("depois", 8.6, 9.0), ("do", 9.7, 9.8),
    ("almoço.", 10.6, 11.4),
    ("Que", 11.8, 12.0), ("tal", 12.05, 12.3), ("a", 12.35, 12.4), ("ideia", 12.45, 12.9), ("de", 12.95, 13.1),
    ("caminhar", 13.15, 13.7), ("bem", 13.75, 14.0), ("cedo?", 14.05, 14.6),
])


def test_sentence_ends_need_a_following_capital_or_trailing_pause():
    assert sentence_end_indices(SENTENCES, duration=15.0) == [1, 8, 14, 22]
    # The last word counts only when silence follows: the clip may have been cut mid-sentence.
    assert sentence_end_indices(SENTENCES, duration=14.7) == [1, 8, 14]


def test_best_window_prefers_fluent_speech_over_the_earliest_stretch():
    # "Vamos à praia depois do almoço." is halting (long gaps between words), so it loses
    # to a fluent sentence even though it would fit.
    w = best_sentence_window(SENTENCES, max_sec=7.0, duration=15.0, min_sec=2.0)
    assert w is not None and w.reason == "sentence"
    assert "Vamos" not in w.text
    assert w.duration <= 7.0


def test_window_respects_minimum_length():
    # Each fluent sentence alone is ~3.3 s; with the 4 s minimum the first two sentences together win.
    w = best_sentence_window(SENTENCES, max_sec=6.0, duration=15.0)
    assert w.text == "Olá pessoal. Hoje o dia está lindo e ensolarado."
    assert w.duration >= 4.0


def test_window_text_matches_its_words():
    w = best_sentence_window(SENTENCES, max_sec=3.5, duration=15.0, min_sec=2.0)
    assert w.text in ("Hoje o dia está lindo e ensolarado.", "Que tal a ideia de caminhar bem cedo?")
    if w.text.startswith("Hoje"):
        assert 1.2 < w.start < 1.8 and 4.6 < w.end < 5.6


def test_no_sentence_fits_falls_back_to_a_pause():
    one_long = words([(f"palavra{i}", i * 0.5, i * 0.5 + 0.3) for i in range(40)])
    assert best_sentence_window(one_long, max_sec=8.0, duration=20.0) is None
    w = word_gap_window(one_long, max_sec=8.0, duration=20.0)
    assert w is not None and w.end <= 8.0 and w.reason == "pause"


def test_last_pause_cut_lands_in_silence():
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 200 * np.arange(int(3.0 * sr)) / sr).astype(np.float32)
    gap = np.zeros(int(0.4 * sr), dtype=np.float32)
    clip = np.concatenate([speech, gap, speech, gap, speech])
    cut = last_pause_cut(clip, sr, cap_sec=8.0)
    assert 6.4 < cut < 6.8  # middle of the second gap (6.4 s .. 6.8 s)


def test_aligner_words_get_their_punctuation_back():
    items = words([("Olá", 0.0, 0.3), ("pessoal", 0.4, 0.9), ("tudo", 1.2, 1.4), ("bem", 1.5, 1.8)])
    out = attach_punctuation("Olá, pessoal! Tudo bem?", items)
    assert [w.text for w in out] == ["Olá,", "pessoal!", "Tudo", "bem?"]
    assert out[1].start == 0.4


def test_aligner_mismatch_returns_bare_items():
    items = words([("outra", 0.0, 0.3), ("coisa", 0.4, 0.9)])
    assert attach_punctuation("Olá pessoal.", items) == items


def test_transcript_mismatch_only_on_plainly_different_speech():
    typed = "essa é uma gravação da minha voz para clonar"
    assert not is_transcript_mismatch(typed, "essa é uma gravação da minha voz pra clonar")
    assert not is_transcript_mismatch(typed + " e muito mais texto", "essa é uma gravação")  # trimmed clip
    assert is_transcript_mismatch(typed, "o rato roeu a roupa do rei de roma ontem à noite em casa com fome demais")
    assert reference_error("a b c", "a b c") == 0.0


def test_runaway_and_collapsed_outputs_are_flagged():
    sr = 24000
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    assert quality.problem(tone, sr, "Olá mundo") is None
    long = np.tile(tone, 40)
    assert "longo" in quality.problem(long, sr, "Olá mundo")
    collapsed = np.concatenate([tone, np.zeros(5 * sr, dtype=np.float32), tone])
    assert "silêncio" in quality.problem(collapsed, sr, "Uma frase razoavelmente longa para caber aqui")
    assert quality.problem(np.zeros(sr, dtype=np.float32), sr, "Oi") == "saída vazia"


def test_join_inserts_pauses_between_chunks():
    sr = 24000
    a = np.ones(sr, dtype=np.float32) * 0.1
    out = quality.join([a, a], sr, pause_sec=0.5)
    assert len(out) == 2 * sr + int(0.5 * sr)


def test_loudness_normalize_reaches_target():
    import pyloudnorm

    sr = 24000
    x = 0.05 * np.sin(2 * np.pi * 220 * np.arange(3 * sr) / sr).astype(np.float32)
    y = quality.loudness_normalize(x, sr, target_lufs=-18.0)
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(y.astype(np.float64)) + 18.0) < 0.5


def test_zero_length_words_stay_in_the_window_text():
    from morpheu.refwindow import clean_words

    ws = clean_words(words([("Hoje", 0.0, 0.3), ("o", 0.3, 0.3), ("céu", 0.3, 0.7), ("abriu.", 0.7, 4.5)]), 5.0)
    assert [w.text for w in ws] == ["Hoje", "o", "céu", "abriu."]
    assert best_sentence_window(ws, 6.0, 5.0, end_pause=0.0).text == "Hoje o céu abriu."
