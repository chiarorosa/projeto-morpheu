"""Voice library stored on disk: data/voices/<id>/{reference.wav, voice.json}."""

import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from . import audio


@dataclass
class Voice:
    id: str
    name: str
    transcript: str
    source: str  # "upload" | "recording" | "design"
    design_prompt: str | None
    duration: float
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return asdict(self)


class VoiceError(ValueError):
    pass


class VoiceStore:
    def __init__(self, root: Path, min_seconds: float, max_seconds: float):
        self.root = root
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self._lock = threading.Lock()

    def _dir(self, voice_id: str) -> Path:
        if not voice_id or not voice_id.isalnum():
            raise KeyError(voice_id)
        return self.root / voice_id

    def audio_path(self, voice_id: str) -> Path:
        path = self._dir(voice_id) / "reference.wav"
        if not path.exists():
            raise KeyError(voice_id)
        return path

    def _save_meta(self, voice: Voice) -> None:
        (self._dir(voice.id) / "voice.json").write_text(json.dumps(voice.to_dict(), ensure_ascii=False, indent=2), "utf-8")

    def create(self, name: str, audio_bytes: bytes, transcript: str = "", source: str = "upload",
               design_prompt: str | None = None, preprocess: bool = True) -> Voice:
        name = name.strip()
        if not name:
            raise VoiceError("Dê um nome para a voz.")
        try:
            data, sr = audio.read_audio(audio_bytes)
        except Exception:
            raise VoiceError("Arquivo de áudio inválido ou formato não suportado (use WAV, FLAC ou OGG).") from None
        if preprocess:
            data = audio.preprocess_reference(data, sr)
        duration = len(data) / sr
        if duration < self.min_seconds:
            raise VoiceError(f"A referência precisa ter pelo menos {self.min_seconds:g} s de fala.")
        if duration > self.max_seconds:
            data = data[: int(self.max_seconds * sr)]
            duration = self.max_seconds

        now = time.time()
        voice = Voice(uuid.uuid4().hex, name, transcript.strip(), source, design_prompt, round(duration, 3), now, now)
        with self._lock:
            d = self._dir(voice.id)
            d.mkdir(parents=True)
            audio.write_wav(d / "reference.wav", data, sr)
            self._save_meta(voice)
        return voice

    def get(self, voice_id: str) -> Voice:
        meta = self._dir(voice_id) / "voice.json"
        if not meta.exists():
            raise KeyError(voice_id)
        return Voice(**json.loads(meta.read_text("utf-8")))

    def find_by_name(self, name: str) -> Voice | None:
        return next((v for v in self.list() if v.name.lower() == name.strip().lower()), None)

    def list(self) -> list[Voice]:
        voices = []
        for meta in self.root.glob("*/voice.json"):
            try:
                voices.append(Voice(**json.loads(meta.read_text("utf-8"))))
            except (OSError, ValueError, TypeError):
                continue  # skip a damaged entry rather than break the whole library
        return sorted(voices, key=lambda v: v.created_at, reverse=True)

    def update(self, voice_id: str, name: str | None = None, transcript: str | None = None) -> Voice:
        with self._lock:
            voice = self.get(voice_id)
            if name is not None:
                if not name.strip():
                    raise VoiceError("O nome não pode ficar vazio.")
                voice.name = name.strip()
            if transcript is not None:
                voice.transcript = transcript.strip()
            voice.updated_at = time.time()
            self._save_meta(voice)
        return voice

    def delete(self, voice_id: str) -> None:
        d = self._dir(voice_id)
        if not d.exists():
            raise KeyError(voice_id)
        with self._lock:
            shutil.rmtree(d)
