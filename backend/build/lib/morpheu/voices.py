"""Voice library stored on disk: data/voices/<id>/{reference.wav, voice.json}."""

from __future__ import annotations

import dataclasses
import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
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
    language: str = "Auto"
    # "manual" when the user typed it, "auto" when transcribed, "" when there is none yet.
    transcript_source: str = ""
    # Per-word timings of the transcript ({text, start, end}), from forced alignment.
    words: list[dict] = field(default_factory=list)
    # The audio plainly says something other than the typed transcript.
    transcript_mismatch: bool = False
    heard_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_timings"] = bool(self.words)
        del d["words"]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Voice":
        known = {f.name for f in dataclasses.fields(Voice)}
        voice = Voice(**{k: v for k, v in d.items() if k in known})
        if voice.transcript and not voice.transcript_source:
            voice.transcript_source = "manual"  # saved before the field existed: typed by the user
        return voice


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
        (self._dir(voice.id) / "voice.json").write_text(json.dumps(asdict(voice), ensure_ascii=False, indent=2), "utf-8")

    def windows_dir(self, voice_id: str) -> Path:
        """Cut-down copies of the reference, one per window a model asked for."""
        d = self._dir(voice_id) / "windows"
        d.mkdir(exist_ok=True)
        return d

    def create(self, name: str, audio_bytes: bytes, transcript: str = "", source: str = "upload",
               design_prompt: str | None = None, preprocess: bool = True, language: str = "Auto") -> Voice:
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
        transcript = transcript.strip()
        voice = Voice(uuid.uuid4().hex, name, transcript, source, design_prompt, round(duration, 3), now, now,
                      language=language or "Auto", transcript_source="manual" if transcript else "")
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
        return Voice.from_dict(json.loads(meta.read_text("utf-8")))

    def find_by_name(self, name: str) -> Voice | None:
        return next((v for v in self.list() if v.name.lower() == name.strip().lower()), None)

    def list(self) -> list[Voice]:
        voices = []
        for meta in self.root.glob("*/voice.json"):
            try:
                voices.append(Voice.from_dict(json.loads(meta.read_text("utf-8"))))
            except (OSError, ValueError, TypeError):
                continue  # skip a damaged entry rather than break the whole library
        return sorted(voices, key=lambda v: v.created_at, reverse=True)

    def update(self, voice_id: str, name: str | None = None, transcript: str | None = None,
               language: str | None = None) -> Voice:
        with self._lock:
            voice = self.get(voice_id)
            if name is not None:
                if not name.strip():
                    raise VoiceError("O nome não pode ficar vazio.")
                voice.name = name.strip()
            if transcript is not None and transcript.strip() != voice.transcript:
                voice.transcript = transcript.strip()
                voice.transcript_source = "manual" if voice.transcript else ""
                # Timings and windows belong to the old text.
                voice.words, voice.transcript_mismatch, voice.heard_text = [], False, ""
                self._clear_windows(voice.id)
            if language is not None:
                voice.language = language
            voice.updated_at = time.time()
            self._save_meta(voice)
        return voice

    def set_analysis(self, voice_id: str, *, transcript: str | None = None, words: list[dict] | None = None,
                     mismatch: bool | None = None, heard_text: str | None = None) -> Voice:
        """Store what speech recognition found. An automatic transcript never replaces one the user typed."""
        with self._lock:
            voice = self.get(voice_id)
            if transcript is not None and voice.transcript_source != "manual":
                voice.transcript, voice.transcript_source = transcript, "auto"
            if words is not None:
                voice.words = words
            if mismatch is not None:
                voice.transcript_mismatch = mismatch
            if heard_text is not None:
                voice.heard_text = heard_text
            voice.updated_at = time.time()
            self._save_meta(voice)
            self._clear_windows(voice.id)
        return voice

    def _clear_windows(self, voice_id: str) -> None:
        shutil.rmtree(self._dir(voice_id) / "windows", ignore_errors=True)

    def delete(self, voice_id: str) -> None:
        d = self._dir(voice_id)
        if not d.exists():
            raise KeyError(voice_id)
        with self._lock:
            shutil.rmtree(d)
