"""Deterministic engine for tests and UI work without a GPU (MORPHEU_FAKE_ENGINES=1)."""

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

FAKE_PARAMS = [
    ParamSpec("temperature", "Variação", "Teste.", "float", 0.9, 0.0, 2.0, 0.05),
]


class FakeEngine(Engine):
    def __init__(self, model_id: str, modes: set[str], speakers: list[str] | None = None, *,
                 ref_max_sec: float = 10.0, uses_ref_text: bool = True, max_chunk_chars: int = 400,
                 runaway_first: int = 0):
        self.info = ModelInfo(
            id=model_id,
            name=model_id,
            description="Motor de teste (tom senoidal).",
            modes=sorted(modes),
            params=FAKE_PARAMS,
            ref_max_sec=ref_max_sec,
            uses_ref_text=uses_ref_text,
            max_chunk_chars=max_chunk_chars,
        )
        self._speakers = speakers or []
        self._loaded = False
        self.calls: list[GenerationRequest] = []
        # Return implausibly long audio for the first N calls, to exercise the retry path.
        self.runaway_first = runaway_first

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, progress=lambda f, m: None) -> None:
        progress(1.0, "pronto")
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def speakers(self) -> list[str]:
        return list(self._speakers)

    def generate(self, req: GenerationRequest):
        self.calls.append(req)
        sr = 24000
        seconds = max(0.5, 0.06 * len(req.text))
        if len(self.calls) <= self.runaway_first:
            seconds = 60.0
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


class FakeAsr:
    """Speech recognition stand-in: 'hears' a fixed text and spaces its words evenly over the clip."""

    vram_gb = 0.0

    def __init__(self, heard: str = "Olá a todos. Esta é a minha voz de teste. Obrigado por ouvir."):
        self.heard = heard
        self._loaded = False
        self.calls: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @staticmethod
    def _spread(text: str, duration: float):
        from ..refwindow import Word

        tokens = text.split()
        step = duration / max(1, len(tokens))
        return [Word(t, i * step + 0.02, (i + 1) * step - 0.08) for i, t in enumerate(tokens)]

    @staticmethod
    def _duration(path: str) -> float:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate

    def transcribe(self, audio_path: str, language=None):
        from ..asr import Transcription

        self.calls.append("transcribe")
        return Transcription(self.heard, "Portuguese", self._spread(self.heard, self._duration(audio_path)))

    def align(self, audio_path: str, transcript: str, language: str):
        self.calls.append("align")
        return self._spread(transcript, self._duration(audio_path))
