import numpy as np
import pytest
import torch

from morpheu import steer


class Layer(torch.nn.Module):
    """Stand-in decoder layer: identity, returning a tuple like the Qwen3 talker layers."""

    def forward(self, x):
        return (x,)


def run(layers, steps):
    """Prefill with a 5-position prompt, then decode `steps` single frames."""
    outs = []
    x = torch.ones(1, 5, 8)
    for layer in layers:
        x = layer(x)[0]
    for _ in range(steps):
        x = torch.ones(1, 1, 8)
        for layer in layers:
            x = layer(x)[0]
        outs.append(x.clone())
    return outs


def test_all_thirteen_emotions_load_with_the_talker_shape():
    assert len(steer.available()) == 13
    for e in steer.available():
        v = steer.load_vector(e.key)
        assert v.shape == (29, 2048) and v.dtype == np.float32 and np.isfinite(v).all()


def test_steering_changes_only_decoded_frames_and_keeps_the_norm():
    layers = torch.nn.ModuleList([Layer() for _ in range(4)])
    vector = np.zeros((4, 8), dtype=np.float32)
    vector[2, 0] = 1.0
    with steer.Steerer(layers, vector, weight=5.0, decay=0.5, first=2, last=2):
        prefill = torch.ones(1, 5, 8)
        for layer in layers:
            prefill = layer(prefill)[0]
        assert torch.equal(prefill, torch.ones(1, 5, 8))  # the prompt is never steered
        frames = run(layers, 3)
    first, second, third = (f[0, 0] for f in frames)
    for f in (first, second, third):
        assert torch.isclose(f.norm(), torch.tensor(8 ** 0.5), atol=1e-5)  # direction moves, size does not
    assert first[0] > second[0] > third[0] > 1.0  # pulled toward the vector, less every frame


def test_hooks_are_removed_after_generation():
    layers = torch.nn.ModuleList([Layer() for _ in range(4)])
    vector = np.ones((4, 8), dtype=np.float32)
    with steer.Steerer(layers, vector, weight=5.0, first=1, last=3):
        pass
    assert torch.equal(run(layers, 1)[0], torch.ones(1, 1, 8))


def test_weight_scales_with_intensity_up_to_the_emotion_ceiling():
    assert steer.weight_for("fear", "forte") == steer.BY_KEY["fear"].ceiling
    assert steer.weight_for("sad", "leve") < steer.weight_for("sad", "normal") < steer.weight_for("sad", "forte")


def test_flatten_fade_holds_the_decline_and_keeps_the_peak():
    sr = 24000
    t = np.arange(4 * sr) / sr
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    faded = tone * np.linspace(1.0, 0.25, len(tone)).astype(np.float32)  # about -12 dB
    assert steer.measure_fade(faded) < -6
    fixed = steer.flatten_fade(faded, -1.5)
    assert steer.measure_fade(fixed) > -3.0
    assert np.isclose(np.max(np.abs(fixed)), np.max(np.abs(faded)), rtol=1e-3)
    assert steer.flatten_fade(tone, -1.5) is tone  # nothing to fix


@pytest.fixture
def emotional_client(tmp_path):
    from fastapi.testclient import TestClient

    from morpheu.app import create_app
    from morpheu.config import Settings
    from morpheu.engines.fake import FakeAsr, FakeEngine
    from morpheu.manager import ModelManager

    engines = [FakeEngine("emo", {"clone"}, emotions=[{"key": "joy", "label": "Alegria"}]),
               FakeEngine("plain", {"clone"})]
    app = create_app(Settings(data_dir=tmp_path), manager=ModelManager(engines), asr=FakeAsr())
    with TestClient(app) as c:
        yield c, engines


def _prepare(client, model_id):
    from tests.conftest import make_wav_bytes

    voice = client.post("/api/v1/voices", data={"name": "v", "transcript": "Olá a todos."},
                        files={"audio": ("r.wav", make_wav_bytes(4.0), "audio/wav")}).json()
    client.post("/api/v1/models/active", json={"model_id": model_id})
    for _ in range(200):
        if client.get("/api/v1/models/active").json()["status"] == "ready":
            break
    return voice["id"]


def test_emotion_reaches_the_engine_and_is_recorded(emotional_client):
    client, engines = emotional_client
    vid = _prepare(client, "emo")
    r = client.post("/api/v1/tts/generate", json={"mode": "clone", "model_id": "emo", "text": "Que dia lindo!",
                                                  "voice_id": vid, "emotion": "joy", "intensity": "forte"})
    assert r.status_code == 200, r.text
    assert (engines[0].calls[-1].emotion, engines[0].calls[-1].intensity) == ("joy", "forte")
    assert r.json()["params"]["emotion"] == "joy"


def test_emotion_on_a_model_without_vectors_is_rejected(emotional_client):
    client, _ = emotional_client
    vid = _prepare(client, "plain")
    r = client.post("/api/v1/tts/generate", json={"mode": "clone", "model_id": "plain", "text": "Oi.",
                                                  "voice_id": vid, "emotion": "joy"})
    assert r.status_code == 400 and "emo" in r.json()["detail"]


def test_unknown_emotion_and_intensity_are_rejected(emotional_client):
    client, _ = emotional_client
    vid = _prepare(client, "emo")
    base = {"mode": "clone", "model_id": "emo", "text": "Oi.", "voice_id": vid}
    assert client.post("/api/v1/tts/generate", json={**base, "emotion": "tédio"}).status_code == 400
    assert client.post("/api/v1/tts/generate", json={**base, "emotion": "joy", "intensity": "máxima"}).status_code == 422
