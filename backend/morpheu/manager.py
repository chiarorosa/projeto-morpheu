"""Keeps at most one model resident on the GPU and serialises generation."""

import threading
from dataclasses import dataclass

from .engines.base import Engine


@dataclass
class ActiveState:
    model_id: str | None = None
    status: str = "idle"  # idle | loading | ready | error
    progress: float = 0.0
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ModelManager:
    def __init__(self, engines: list[Engine]):
        self.engines: dict[str, Engine] = {e.info.id: e for e in engines}
        self.state = ActiveState()
        self._state_lock = threading.Lock()
        # Held while loading/unloading or generating, so a swap never races a generation.
        self._gpu_lock = threading.Lock()

    def get(self, model_id: str) -> Engine:
        try:
            return self.engines[model_id]
        except KeyError:
            raise KeyError(model_id) from None

    def status_of(self, model_id: str) -> str:
        s = self.state
        if s.model_id == model_id and s.status in ("loading", "error"):
            return s.status
        return "ready" if self.engines[model_id].is_loaded else "unloaded"

    def activate(self, model_id: str) -> None:
        """Start loading model_id in the background, unloading whatever else is resident."""
        engine = self.get(model_id)
        with self._state_lock:
            if self.state.model_id == model_id and self.state.status in ("loading", "ready"):
                return
            self.state = ActiveState(model_id=model_id, status="loading", message="Preparando…")
        threading.Thread(target=self._load, args=(engine,), daemon=True, name=f"load-{model_id}").start()

    def _load(self, engine: Engine) -> None:
        def progress(fraction: float, message: str) -> None:
            with self._state_lock:
                if self.state.model_id == engine.info.id:
                    self.state.progress, self.state.message = fraction, message

        with self._gpu_lock:
            try:
                for other in self.engines.values():
                    if other is not engine and other.is_loaded:
                        other.unload()
                if not engine.is_loaded:
                    engine.load(progress)
                new = ActiveState(engine.info.id, "ready", 1.0, "Pronto")
            except Exception as exc:  # surfaced to the UI via /models/active
                new = ActiveState(engine.info.id, "error", 0.0, "Falha ao carregar", f"{type(exc).__name__}: {exc}")
        with self._state_lock:
            if self.state.model_id == engine.info.id:
                self.state = new

    def run(self, model_id: str, fn):
        """Run fn(engine) with exclusive GPU access. Raises RuntimeError if the model is not ready."""
        engine = self.get(model_id)
        with self._gpu_lock:
            if not engine.is_loaded:
                raise RuntimeError(f"O modelo {model_id} não está carregado. Ative-o primeiro.")
            return fn(engine)

    def forget_voice(self, voice_id: str) -> None:
        for e in self.engines.values():
            e.forget_voice(voice_id)

    def shutdown(self) -> None:
        with self._gpu_lock:
            for e in self.engines.values():
                if e.is_loaded:
                    e.unload()
