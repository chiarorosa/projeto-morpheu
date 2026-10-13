"""Deterministic engine for tests and UI work without a GPU (MORPHEU_FAKE_ENGINES=1)."""

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

FAKE_PARAMS = [
    ParamSpec("temperature", "Variação", "Teste.", "float", 0.9, 0.0, 2.0, 0.05),
]


class FakeEngine(Engine):
    def __init__(self, model_id: str, modes: set[str], speakers: list[str] | None = None):
        self.info = ModelInfo(
            id=model_id,
            name=model_id,
            description="Motor de teste (tom senoidal).",
            modes=sorted(modes),
            params=FAKE_PARAMS,
        )
        self._speakers = speakers or []
        self._loaded = False
        self.calls: list[GenerationRequest] = []

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
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr
