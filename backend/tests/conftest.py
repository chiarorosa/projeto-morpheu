import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from morpheu.app import create_app
from morpheu.config import Settings
from morpheu.engines.fake import FakeEngine
from morpheu.manager import ModelManager


def make_wav_bytes(seconds: float = 2.0, sr: int = 24000, silence_pad: float = 0.0) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    pad = np.zeros(int(silence_pad * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([pad, tone, pad]), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


@pytest.fixture
def manager():
    return ModelManager(
        [
            FakeEngine("fake-clone", modes={"clone"}),
            FakeEngine("fake-design", modes={"design"}),
            FakeEngine("fake-tts", modes={"tts"}, speakers=["Ryan", "Vivian"]),
        ]
    )


@pytest.fixture
def client(settings, manager):
    app = create_app(settings, manager=manager)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def wav_bytes():
    return make_wav_bytes()
