"""Engine contract shared by every voice model backend."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["clone", "design", "tts"]

LANGUAGES = [
    "Auto", "Portuguese", "English", "Spanish", "French", "German",
    "Italian", "Russian", "Chinese", "Japanese", "Korean",
]


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    description: str
    kind: Literal["float", "int", "bool"]
    default: float | int | bool
    min: float | None = None
    max: float | None = None
    step: float | None = None

    def validate(self, value):
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.key} must be true or false")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.key} must be a number")
        if self.min is not None and value < self.min or self.max is not None and value > self.max:
            raise ValueError(f"{self.key} must be between {self.min} and {self.max}")
        return int(value) if self.kind == "int" else float(value)


# Parameters every engine gets from the pipeline, independent of the model.
COMMON_PARAMS = [
    ParamSpec("preprocess_prompt", "Limpar referência", "Remove silêncio e equaliza o volume da referência antes de clonar.", "bool", True),
    ParamSpec("postprocess_output", "Polir saída", "Corta silêncio morto e nivela a loudness do resultado.", "bool", True),
    ParamSpec("normalize_text", "Ler números por extenso", "Escreve números, valores, porcentagens, horas e abreviações como devem ser falados.", "bool", True),
    ParamSpec("seed", "Semente", "0 = aleatória a cada geração; outro valor reproduz o resultado.", "int", 0, 0, 2**31 - 1, 1),
]


@dataclass
class ModelInfo:
    id: str
    name: str
    description: str
    modes: list[Mode]
    languages: list[str] = field(default_factory=lambda: list(LANGUAGES))
    params: list[ParamSpec] = field(default_factory=list)
    vram_gb: float = 0.0
    # Reference window the model clones best from; longer clips are cut to their most
    # fluent sentence-aligned stretch, shorter ones are padded or rejected.
    ref_min_sec: float = 1.5
    ref_max_sec: float = 10.0
    # Whether the model reads the reference transcript (Chatterbox only hears the audio).
    uses_ref_text: bool = True
    # Longest text sent in one call; longer input is split at sentence boundaries.
    max_chunk_chars: int = 400
    sample_rate: int = 24000
    recommended_for: list[str] = field(default_factory=list)
    # Emotions this model can steer a cloned voice toward: [{"key", "label"}].
    emotions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = [asdict(p) for p in self.all_params]
        return d

    @property
    def all_params(self) -> list[ParamSpec]:
        return [*self.params, *COMMON_PARAMS]

    def resolve_params(self, given: dict) -> dict:
        """Merge user params over defaults; reject unknown or out-of-range values."""
        specs = {p.key: p for p in self.all_params}
        unknown = set(given) - set(specs)
        if unknown:
            raise ValueError(f"parâmetros desconhecidos para {self.id}: {', '.join(sorted(unknown))}")
        return {k: (spec.validate(given[k]) if k in given else spec.default) for k, spec in specs.items()}


@dataclass
class GenerationRequest:
    mode: Mode
    text: str
    language: str = "Auto"
    params: dict = field(default_factory=dict)
    # clone
    voice_id: str | None = None
    ref_audio_path: str | None = None
    ref_text: str | None = None
    # Identifies voice + reference window, so engines can cache features per window.
    ref_key: str | None = None
    # Emotion steering (clone mode, on models that list emotions).
    emotion: str | None = None
    intensity: str = "normal"
    # design / tts
    instruct: str | None = None
    speaker: str | None = None


class Engine(ABC):
    info: ModelInfo

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def load(self, progress=lambda fraction, message: None) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(self, req: GenerationRequest) -> tuple[np.ndarray, int]: ...

    def speakers(self) -> list[str]:
        return []

    def forget_voice(self, voice_id: str) -> None:
        """Drop any cached features for a voice (called when it changes or is deleted)."""
