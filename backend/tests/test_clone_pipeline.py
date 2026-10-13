import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from morpheu.app import create_app
from morpheu.engines.fake import FakeAsr, FakeEngine
from morpheu.manager import ModelManager

LONG_TRANSCRIPT = ("Bom dia a todos. Hoje vou ler um texto curto para testar. "
                   "Esta é a segunda parte da gravação. E aqui termina a referência.")


def speech_like(seconds: float, sr: int = 24000) -> bytes:
    """Tone bursts separated by short gaps, so silence trimming keeps the full length."""
    t = np.arange(int(0.45 * sr)) / sr
    burst = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    gap = np.zeros(int(0.05 * sr), dtype=np.float32)
    unit = np.concatenate([burst, gap])
    data = np.tile(unit, int(seconds / 0.5) + 1)[: int(seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def engines():
    return {
        "short-ref": FakeEngine("short-ref", {"clone"}, ref_max_sec=6.0),
        "no-text": FakeEngine("no-text", {"clone"}, ref_max_sec=6.0, uses_ref_text=False),
        "small-chunks": FakeEngine("small-chunks", {"clone"}, max_chunk_chars=40),
        "flaky": FakeEngine("flaky", {"clone"}, runaway_first=1),
    }


@pytest.fixture
def asr():
    return FakeAsr()


@pytest.fixture
def client(tmp_path, engines, asr):
    from morpheu.config import Settings

    app = create_app(Settings(data_dir=tmp_path), manager=ModelManager(list(engines.values())), asr=asr)
    with TestClient(app) as c:
        yield c


def add_voice(client, seconds, transcript=""):
    r = client.post("/api/v1/voices", data={"name": f"v{seconds}", "transcript": transcript, "language": "Portuguese"},
                    files={"audio": ("ref.wav", speech_like(seconds), "audio/wav")})
    assert r.status_code == 201, r.text
    return r.json()


def activate(client, model_id):
    client.post("/api/v1/models/active", json={"model_id": model_id})
    for _ in range(200):
        if client.get("/api/v1/models/active").json()["status"] == "ready":
            return
    raise AssertionError("model did not load")


def clone(client, model_id, voice_id, text="Olá, tudo bem?", **params):
    return client.post("/api/v1/tts/generate", json={
        "mode": "clone", "model_id": model_id, "text": text, "language": "Portuguese",
        "voice_id": voice_id, "params": params})


def test_long_reference_is_cut_at_sentence_boundaries(client, engines, asr):
    asr.heard = LONG_TRANSCRIPT
    voice = add_voice(client, 12.0, LONG_TRANSCRIPT)
    activate(client, "short-ref")
    r = clone(client, "short-ref", voice["id"])
    assert r.status_code == 200, r.text
    req = engines["short-ref"].calls[-1]
    assert "windows" in req.ref_audio_path
    info = sf.info(req.ref_audio_path)
    assert info.frames / info.samplerate <= 6.0 + 2.0
    # The model gets only the sentences it hears, ending on a full stop.
    assert req.ref_text and req.ref_text in LONG_TRANSCRIPT and req.ref_text.endswith(".")
    assert "trecho" in r.json()["params"]["reference"]
    assert "align" in asr.calls  # the typed transcript was timed, not replaced
    assert client.get(f"/api/v1/voices/{voice['id']}").json()["transcript"] == LONG_TRANSCRIPT


def test_contradicted_transcript_falls_back_to_a_pause_cut_transcribed_alone(client, engines, asr):
    voice = add_voice(client, 12.0, LONG_TRANSCRIPT)  # the fake ASR hears something else
    activate(client, "short-ref")
    r = clone(client, "short-ref", voice["id"])
    assert r.status_code == 200
    req = engines["short-ref"].calls[-1]
    assert "windows" in req.ref_audio_path
    # Timings of a transcript the audio contradicts are useless, so the cut is transcribed on its own.
    assert req.ref_text == asr.heard
    assert client.get(f"/api/v1/voices/{voice['id']}").json()["transcript_mismatch"] is True


def test_short_reference_is_used_whole(client, engines):
    voice = add_voice(client, 5.0, "Uma referência curta.")
    activate(client, "short-ref")
    assert clone(client, "short-ref", voice["id"]).status_code == 200
    assert engines["short-ref"].calls[-1].ref_audio_path.endswith("reference.wav")


def test_missing_transcript_is_transcribed_and_kept(client, engines, asr):
    voice = add_voice(client, 5.0)
    activate(client, "short-ref")
    assert clone(client, "short-ref", voice["id"]).status_code == 200
    saved = client.get(f"/api/v1/voices/{voice['id']}").json()
    assert saved["transcript"] == asr.heard and saved["transcript_source"] == "auto"
    assert engines["short-ref"].calls[-1].ref_text == asr.heard


def test_model_that_ignores_transcripts_gets_none(client, engines, asr):
    voice = add_voice(client, 5.0)
    activate(client, "no-text")
    assert clone(client, "no-text", voice["id"]).status_code == 200
    assert engines["no-text"].calls[-1].ref_text is None
    assert asr.calls == []  # no transcription needed


def test_analyze_flags_a_transcript_the_audio_contradicts(client, asr):
    voice = add_voice(client, 5.0, "o rato roeu a roupa do rei de roma ontem à noite")
    r = client.post(f"/api/v1/voices/{voice['id']}/analyze", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript_mismatch"] is True and body["heard_text"] == asr.heard
    assert body["transcript"] == "o rato roeu a roupa do rei de roma ontem à noite"  # typed text is kept
    assert body["has_timings"] is False


def test_analyze_times_a_matching_transcript(client, asr):
    voice = add_voice(client, 5.0, asr.heard)
    body = client.post(f"/api/v1/voices/{voice['id']}/analyze", json={}).json()
    assert body["transcript_mismatch"] is False and body["has_timings"] is True


def test_editing_the_transcript_drops_old_timings(client, asr):
    voice = add_voice(client, 5.0, asr.heard)
    client.post(f"/api/v1/voices/{voice['id']}/analyze", json={})
    body = client.patch(f"/api/v1/voices/{voice['id']}", json={"transcript": "Outro texto."}).json()
    assert body["has_timings"] is False and body["transcript_source"] == "manual"


def test_long_text_is_generated_in_chunks(client, engines):
    voice = add_voice(client, 5.0, "Referência.")
    activate(client, "small-chunks")
    text = "Primeira frase do texto. Segunda frase do texto. Terceira frase do texto."
    r = clone(client, "small-chunks", voice["id"], text)
    assert r.status_code == 200
    assert r.json()["params"]["chunks"] == 3
    assert [c.text for c in engines["small-chunks"].calls] == [
        "Primeira frase do texto.", "Segunda frase do texto.", "Terceira frase do texto."]


def test_runaway_output_is_regenerated_with_a_new_seed(client, engines):
    voice = add_voice(client, 5.0, "Referência.")
    activate(client, "flaky")
    r = clone(client, "flaky", voice["id"], "Olá mundo.", seed=100)
    assert r.status_code == 200
    calls = engines["flaky"].calls
    assert [c.params["seed"] for c in calls] == [100, 101]
    assert r.json()["params"]["retries"] == 1
    assert r.json()["duration"] < 10


def test_numbers_are_spoken_but_history_keeps_what_was_typed(client, engines):
    voice = add_voice(client, 5.0, "Referência.")
    activate(client, "short-ref")
    r = clone(client, "short-ref", voice["id"], "Custa R$ 10.")
    assert engines["short-ref"].calls[-1].text == "Custa dez reais."
    assert r.json()["text"] == "Custa R$ 10."
    assert r.json()["params"]["spoken_text"] == "Custa dez reais."


def test_text_normalization_can_be_turned_off(client, engines):
    voice = add_voice(client, 5.0, "Referência.")
    activate(client, "short-ref")
    clone(client, "short-ref", voice["id"], "Custa R$ 10.", normalize_text=False)
    assert engines["short-ref"].calls[-1].text == "Custa R$ 10."


def test_analysis_adopts_the_detected_language(client):
    r = client.post("/api/v1/voices", data={"name": "auto", "transcript": ""},
                    files={"audio": ("ref.wav", speech_like(5.0), "audio/wav")})
    assert r.json()["language"] == "Auto"
    body = client.post(f"/api/v1/voices/{r.json()['id']}/analyze", json={}).json()
    assert body["language"] == "Portuguese"
