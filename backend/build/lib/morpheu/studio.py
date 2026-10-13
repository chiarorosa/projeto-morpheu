"""Generation pipeline.

text: strip tags -> spell out numbers/symbols -> split into sentence chunks
voice: transcribe if needed -> cut to the model's reference window (reference.py)
model: generate each chunk -> discard runaway/collapsed output and retry with a new seed
audio: join chunks -> trim and level loudness -> record in history
"""

import random
import time

import numpy as np

from . import audio, quality
from .chunker import chunk_text
from .engines.base import GenerationRequest
from .history import History
from .manager import ModelManager
from .reference import ReferenceError, ReferencePreparer
from .textnorm import normalize_text, strip_expression_tags
from .voices import VoiceStore

# Fresh seeds tried for a chunk whose output fails the quality checks.
_RETRIES = 2


class BadRequest(ValueError):
    pass


class NotReady(RuntimeError):
    pass


class GenerationFailed(RuntimeError):
    pass


class Studio:
    def __init__(self, manager: ModelManager, voices: VoiceStore, history: History, asr=None):
        self.manager = manager
        self.voices = voices
        self.history = history
        self.references = ReferencePreparer(voices, manager, asr)

    def generate(self, *, mode: str, model_id: str, text: str, language: str = "Auto",
                 voice_id: str | None = None, instruct: str | None = None,
                 speaker: str | None = None, params: dict | None = None,
                 emotion: str | None = None, intensity: str = "normal") -> dict:
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
        if emotion:
            if mode != "clone" or not info.emotions:
                raise BadRequest(f"Emoções só funcionam na clonagem com {self._emotion_models()}.")
            if emotion not in {e["key"] for e in info.emotions}:
                raise BadRequest(f"Emoção desconhecida: {emotion}")
            from .steer import INTENSITIES

            if intensity not in INTENSITIES:
                raise BadRequest(f"Intensidade inválida: {intensity}")
        if not engine.is_loaded:
            raise NotReady(f"O modelo {model_id} não está carregado. Ative-o primeiro.")

        spoken = strip_expression_tags(text)
        if resolved["normalize_text"]:
            spoken = normalize_text(spoken, self._text_language(language, voice_id))
        if not spoken:
            raise BadRequest("Digite o texto a ser falado.")

        req = GenerationRequest(mode=mode, text=spoken, language=language, params=resolved,
                                instruct=(instruct or "").strip() or None, speaker=speaker,
                                emotion=emotion or None, intensity=intensity)
        ref_note = None
        if mode == "clone":
            if not voice_id:
                raise BadRequest("Escolha uma voz da biblioteca para clonar.")
            try:
                voice = self.voices.get(voice_id)
            except KeyError:
                raise BadRequest("Voz não encontrada.") from None
            try:
                ref = self.references.prepare(voice, info, language)
            except ReferenceError as exc:
                raise BadRequest(str(exc)) from None
            if info.uses_ref_text and not ref.text and info.id == "omnivoice":
                raise BadRequest("O OmniVoice precisa da transcrição da referência. Digite-a na Biblioteca "
                                 "ou use “Transcrever automaticamente”.")
            req.voice_id, req.ref_audio_path, req.ref_text, req.ref_key = voice.id, ref.path, ref.text, ref.key
            ref_note = ref.note
        elif mode == "design" and not req.instruct:
            raise BadRequest("Descreva a voz que você quer criar.")
        elif mode == "tts":
            speakers = engine.speakers()
            if speakers and speaker not in speakers:
                raise BadRequest(f"Escolha uma das vozes prontas: {', '.join(speakers)}")

        seed = resolved["seed"] or random.randint(1, 2**31 - 1)
        # A designed voice is invented per call, so it must come out of a single call.
        chunks = [spoken] if mode == "design" else chunk_text(spoken, info.max_chunk_chars)

        started = time.perf_counter()
        parts, sr, retries = [], info.sample_rate, 0
        for chunk in chunks:
            wav, sr, used = self._generate_chunk(model_id, req, chunk, seed, resolved)
            retries += used
            parts.append(wav)
        wav = quality.join(parts, sr) if len(parts) > 1 else parts[0]
        if resolved["postprocess_output"]:
            wav = audio.trim_silence(wav, sr, threshold_db=-50.0, margin_s=0.15)
            wav = quality.loudness_normalize(wav, sr)
        elapsed = time.perf_counter() - started

        recorded = {**resolved, "seed": seed, "chunks": len(chunks)}
        if req.emotion:
            recorded.update(emotion=req.emotion, intensity=req.intensity)
        if retries:
            recorded["retries"] = retries
        if ref_note:
            recorded["reference"] = ref_note
        if spoken != text.strip():
            recorded["spoken_text"] = spoken

        gen_id, path = self.history.new_audio_path()
        audio.write_wav(path, wav, sr)
        return self.history.add(
            gen_id, mode=mode, model_id=model_id, text=text.strip(), language=language,
            voice_id=req.voice_id, speaker=speaker, instruct=req.instruct, params=recorded,
            audio_path=path, duration=round(len(wav) / sr, 3), sample_rate=sr, elapsed=round(elapsed, 3),
        )

    def _emotion_models(self) -> str:
        names = [e.info.name for e in self.manager.engines.values() if e.info.emotions]
        return " ou ".join(names) or "nenhum modelo instalado"

    def _text_language(self, language: str, voice_id: str | None) -> str:
        if language != "Auto" or not voice_id:
            return language
        try:
            return self.voices.get(voice_id).language
        except KeyError:
            return language

    def _generate_chunk(self, model_id: str, req: GenerationRequest, chunk: str, seed: int,
                        resolved: dict) -> tuple[np.ndarray, int, int]:
        """Generate one chunk; regenerate with a new seed when the output loops, collapses or is empty."""
        last = None
        for attempt in range(_RETRIES + 1):
            sub = GenerationRequest(**{**req.__dict__, "text": chunk,
                                       "params": {**resolved, "seed": seed + attempt}})
            try:
                wav, sr = self.manager.run(model_id, lambda e: e.generate(sub))
            except RuntimeError as exc:
                if not self.manager.get(model_id).is_loaded:
                    raise NotReady(str(exc)) from None
                raise
            wav = np.asarray(wav, dtype=np.float32)
            last = quality.problem(wav, sr, chunk)
            if last is None:
                return wav, sr, attempt
        raise GenerationFailed(f"O modelo falhou neste trecho após {_RETRIES + 1} tentativas ({last}): “{chunk[:80]}”")

    def save_voice_from_history(self, history_id: str, name: str):
        item = self.history.get(history_id)
        data = self.history.audio_path(history_id).read_bytes()
        return self.voices.create(
            name, data, transcript=item["text"], source="design" if item["mode"] == "design" else "upload",
            design_prompt=item["instruct"], preprocess=False, language=item["language"],
        )
