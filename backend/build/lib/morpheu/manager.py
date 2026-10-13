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

    def run_aux(self, service, fn, progress=lambda message: None):
        """Run fn(service) for a helper model (speech recognition) with exclusive GPU access.
        When the resident voice model leaves too little memory, it is unloaded first and
        loaded back afterwards, so the caller finds the GPU as it left it."""
        with self._gpu_lock:
            evicted = []
            if service.vram_gb and self._free_vram_gb() < service.vram_gb:
                for e in self.engines.values():
                    if e.is_loaded:
                        progress(f"Liberando a GPU ({e.info.name})…")
                        e.unload()
                        evicted.append(e)
            try:
                if not service.is_loaded:
                    progress("Carregando o reconhecimento de fala…")
                    service.load()
                return fn(service)
            finally:
                service.unload()
                for e in evicted:
                    progress(f"Recarregando {e.info.name}…")
                    try:
                        e.load()
                    except Exception as exc:
                        with self._state_lock:
                            self.state = ActiveState(e.info.id, "error", 0.0, "Falha ao recarregar",
                                                     f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _free_vram_gb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()[0] / 2**30
        except Exception:
            pass
        return float("inf")

    def forget_voice(self, voice_id: str) -> None:
        for e in self.engines.values():
            e.forget_voice(voice_id)

    def shutdown(self) -> None:
        with self._gpu_lock:
            for e in self.engines.values():
                if e.is_loaded:
                    e.unload()
