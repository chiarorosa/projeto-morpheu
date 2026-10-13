"""Generation pipeline: validate -> resolve voice -> run engine -> post-process -> record."""

import random
import time

from . import audio
from .engines.base import GenerationRequest
from .history import History
from .manager import ModelManager
from .voices import VoiceStore


class BadRequest(ValueError):
    pass


class NotReady(RuntimeError):
    pass


class Studio:
    def __init__(self, manager: ModelManager, voices: VoiceStore, history: History):
        self.manager = manager
        self.voices = voices
        self.history = history

    def generate(self, *, mode: str, model_id: str, text: str, language: str = "Auto",
                 voice_id: str | None = None, instruct: str | None = None,
                 speaker: str | None = None, params: dict | None = None) -> dict:
        try:
            engine = self.manager.get(model_id)
        except KeyError:
            raise BadRequest(f"Modelo desconhecido: {model_id}") from None
        info = engine.info
        if mode not in info.modes:
            raise BadRequest(f"O modelo {info.name} não suporta o modo '{mode}'.")
        if language not in info.languages:
            raise BadRequest(f"Idioma não suportado por {info.name}: {language}")
        try:
            resolved = info.resolve_params(params or {})
        except ValueError as exc:
            raise BadRequest(str(exc)) from None

        req = GenerationRequest(mode=mode, text=text.strip(), language=language, params=resolved,
                                instruct=(instruct or "").strip() or None, speaker=speaker)
        if mode == "clone":
            if not voice_id:
                raise BadRequest("Escolha uma voz da biblioteca para clonar.")
            try:
                voice = self.voices.get(voice_id)
            except KeyError:
                raise BadRequest("Voz não encontrada.") from None
            req.voice_id = voice.id
            req.ref_audio_path = str(self.voices.audio_path(voice.id))
            req.ref_text = voice.transcript or None
        elif mode == "design" and not req.instruct:
            raise BadRequest("Descreva a voz que você quer criar.")
        elif mode == "tts":
            speakers = engine.speakers()
            if speakers and speaker not in speakers:
                raise BadRequest(f"Escolha uma das vozes prontas: {', '.join(speakers)}")

        seed = resolved["seed"] or random.randint(1, 2**31 - 1)
        req.params = {**resolved, "seed": seed}

        started = time.perf_counter()
        try:
            wav, sr = self.manager.run(model_id, lambda e: e.generate(req))
        except RuntimeError as exc:
            if not engine.is_loaded:
                raise NotReady(str(exc)) from None
            raise
        if resolved["postprocess_output"]:
            wav = audio.postprocess_output(wav, sr)
        elapsed = time.perf_counter() - started

        gen_id, path = self.history.new_audio_path()
        audio.write_wav(path, wav, sr)
        return self.history.add(
            gen_id, mode=mode, model_id=model_id, text=req.text, language=language,
            voice_id=req.voice_id, speaker=speaker, instruct=req.instruct, params=req.params,
            audio_path=path, duration=round(len(wav) / sr, 3), sample_rate=sr, elapsed=round(elapsed, 3),
        )

    def save_voice_from_history(self, history_id: str, name: str):
        item = self.history.get(history_id)
        data = self.history.audio_path(history_id).read_bytes()
        return self.voices.create(
            name, data, transcript=item["text"], source="design" if item["mode"] == "design" else "upload",
            design_prompt=item["instruct"], preprocess=False,
        )
