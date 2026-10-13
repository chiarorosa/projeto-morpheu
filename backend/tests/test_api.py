import time


def wait_ready(client, model_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/api/v1/models/active").json()
        if body["model_id"] == model_id and body["status"] in ("ready", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError("model did not load in time")


def activate(client, model_id):
    r = client.post("/api/v1/models/active", json={"model_id": model_id})
    assert r.status_code == 202
    assert wait_ready(client, model_id)["status"] == "ready"


def create_voice(client, wav_bytes, name="Ana", transcript="Olá mundo"):
    r = client.post(
        "/api/v1/voices",
        data={"name": name, "transcript": transcript},
        files={"audio": ("ref.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------- models ----------

def test_lists_models_with_modes_and_params(client):
    models = client.get("/api/v1/models").json()
    ids = {m["id"] for m in models}
    assert {"fake-clone", "fake-design", "fake-tts"} <= ids
    clone = next(m for m in models if m["id"] == "fake-clone")
    assert clone["modes"] == ["clone"]
    assert clone["status"] == "unloaded"
    assert any(p["key"] == "temperature" for p in clone["params"])


def test_activating_unknown_model_is_404(client):
    assert client.post("/api/v1/models/active", json={"model_id": "nope"}).status_code == 404


def test_only_one_model_stays_loaded(client, manager):
    activate(client, "fake-clone")
    activate(client, "fake-design")
    loaded = [e.info.id for e in manager.engines.values() if e.is_loaded]
    assert loaded == ["fake-design"]


def test_tts_speakers_and_languages(client):
    assert client.get("/api/v1/tts/speakers", params={"model_id": "fake-tts"}).json() == ["Ryan", "Vivian"]
    assert "Portuguese" in client.get("/api/v1/tts/languages", params={"model_id": "fake-tts"}).json()


# ---------- voices ----------

def test_voice_crud(client, wav_bytes):
    v = create_voice(client, wav_bytes)
    assert v["name"] == "Ana" and v["transcript"] == "Olá mundo" and v["duration"] > 0

    assert [x["id"] for x in client.get("/api/v1/voices").json()] == [v["id"]]
    audio = client.get(f"/api/v1/voices/{v['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"

    assert client.patch(f"/api/v1/voices/{v['id']}", json={"name": "Bia"}).json()["name"] == "Bia"
    assert client.patch(f"/api/v1/voices/{v['id']}", json={"transcript": "Oi"}).json()["transcript"] == "Oi"

    assert client.delete(f"/api/v1/voices/{v['id']}").status_code == 204
    assert client.get(f"/api/v1/voices/{v['id']}").status_code == 404


def test_voice_upload_rejects_non_audio(client):
    r = client.post(
        "/api/v1/voices",
        data={"name": "x"},
        files={"audio": ("a.wav", b"not audio", "audio/wav")},
    )
    assert r.status_code == 400


def test_voice_upload_rejects_too_short_reference(client):
    from tests.conftest import make_wav_bytes

    r = client.post(
        "/api/v1/voices",
        data={"name": "x"},
        files={"audio": ("a.wav", make_wav_bytes(seconds=0.5), "audio/wav")},
    )
    assert r.status_code == 400


# ---------- generation ----------

def test_generate_requires_loaded_model(client, wav_bytes):
    v = create_voice(client, wav_bytes)
    r = client.post("/api/v1/tts/generate", json={"mode": "clone", "model_id": "fake-clone", "text": "oi", "voice_id": v["id"]})
    assert r.status_code == 409


def test_generate_clone_records_history_and_serves_audio(client, manager, wav_bytes):
    v = create_voice(client, wav_bytes)
    activate(client, "fake-clone")
    r = client.post(
        "/api/v1/tts/generate",
        json={
            "mode": "clone",
            "model_id": "fake-clone",
            "text": "Bom dia, tudo bem?",
            "language": "Portuguese",
            "voice_id": v["id"],
            "params": {"temperature": 0.5},
        },
    )
    assert r.status_code == 200, r.text
    item = r.json()
    assert item["duration"] > 0 and item["voice_id"] == v["id"]

    call = manager.engines["fake-clone"].calls[-1]
    assert call.ref_text == "Olá mundo" and call.ref_audio_path.endswith(".wav")
    assert call.params["temperature"] == 0.5

    hist = client.get("/api/v1/history").json()
    assert [h["id"] for h in hist] == [item["id"]]
    audio = client.get(f"/api/v1/history/{item['id']}/audio")
    assert audio.status_code == 200 and len(audio.content) > 44


def test_generate_rejects_mode_the_model_does_not_support(client):
    activate(client, "fake-design")
    r = client.post("/api/v1/tts/generate", json={"mode": "clone", "model_id": "fake-design", "text": "oi"})
    assert r.status_code == 400


def test_generate_clone_without_voice_is_400(client):
    activate(client, "fake-clone")
    r = client.post("/api/v1/tts/generate", json={"mode": "clone", "model_id": "fake-clone", "text": "oi"})
    assert r.status_code == 400


def test_generate_design_requires_description(client):
    activate(client, "fake-design")
    r = client.post("/api/v1/tts/generate", json={"mode": "design", "model_id": "fake-design", "text": "oi"})
    assert r.status_code == 400


def test_generate_rejects_empty_text(client):
    activate(client, "fake-tts")
    r = client.post("/api/v1/tts/generate", json={"mode": "tts", "model_id": "fake-tts", "text": "   ", "speaker": "Ryan"})
    assert r.status_code == 422


def test_unknown_param_is_rejected(client):
    activate(client, "fake-tts")
    r = client.post(
        "/api/v1/tts/generate",
        json={"mode": "tts", "model_id": "fake-tts", "text": "oi", "speaker": "Ryan", "params": {"bogus": 1}},
    )
    assert r.status_code == 400


def test_param_out_of_range_is_rejected(client):
    activate(client, "fake-tts")
    r = client.post(
        "/api/v1/tts/generate",
        json={"mode": "tts", "model_id": "fake-tts", "text": "oi", "speaker": "Ryan", "params": {"temperature": 99}},
    )
    assert r.status_code == 400


def test_designed_voice_can_be_saved_to_library(client):
    activate(client, "fake-design")
    item = client.post(
        "/api/v1/tts/generate",
        json={"mode": "design", "model_id": "fake-design", "text": "Olá, eu sou uma voz nova.", "instruct": "voz grave e calma"},
    ).json()
    r = client.post("/api/v1/voices/from-history", json={"history_id": item["id"], "name": "Narrador"})
    assert r.status_code == 201
    voice = r.json()
    assert voice["source"] == "design"
    assert voice["transcript"] == "Olá, eu sou uma voz nova."
    assert voice["design_prompt"] == "voz grave e calma"


def test_history_delete(client):
    activate(client, "fake-tts")
    item = client.post("/api/v1/tts/generate", json={"mode": "tts", "model_id": "fake-tts", "text": "oi", "speaker": "Ryan"}).json()
    assert client.delete(f"/api/v1/history/{item['id']}").status_code == 204
    assert client.get("/api/v1/history").json() == []
    assert client.get(f"/api/v1/history/{item['id']}/audio").status_code == 404


# ---------- OpenAI-compatible ----------

def test_openai_compatible_speech_endpoint_uses_saved_voice(client, wav_bytes):
    v = create_voice(client, wav_bytes, name="Ana")
    activate(client, "fake-clone")
    r = client.post("/v1/audio/speech", json={"model": "fake-clone", "input": "Teste", "voice": "Ana"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
