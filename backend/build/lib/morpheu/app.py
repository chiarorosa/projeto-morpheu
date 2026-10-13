"""FastAPI application: REST API under /api/v1, OpenAI-compatible /v1/audio/speech, and the built UI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import Settings
from .history import History
from .manager import ModelManager
from .reference import ReferenceError
from .studio import BadRequest, GenerationFailed, NotReady, Studio
from .voices import VoiceError, VoiceStore


class ActivateBody(BaseModel):
    model_id: str


class GenerateBody(BaseModel):
    mode: str = Field(pattern="^(clone|design|tts)$")
    model_id: str
    text: str = Field(max_length=5000)
    language: str = "Auto"
    voice_id: str | None = None
    instruct: str | None = Field(default=None, max_length=2000)
    speaker: str | None = None
    params: dict = Field(default_factory=dict)
    emotion: str | None = Field(default=None, max_length=32)
    intensity: str = Field(default="normal", pattern="^(leve|normal|forte)$")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Digite o texto a ser falado.")
        return v


class VoicePatch(BaseModel):
    name: str | None = None
    transcript: str | None = None
    language: str | None = None


class AnalyzeBody(BaseModel):
    language: str | None = None


class SaveFromHistory(BaseModel):
    history_id: str
    name: str


class SpeechBody(BaseModel):
    """Subset of OpenAI's /v1/audio/speech. `voice` is a library voice name or a built-in speaker."""
    model: str
    input: str = Field(min_length=1, max_length=5000)
    voice: str
    instructions: str | None = None
    response_format: str = Field(default="wav", pattern="^wav$")


def default_manager() -> ModelManager:
    import os

    if os.environ.get("MORPHEU_FAKE_ENGINES") == "1":
        from .engines.fake import FakeEngine

        return ModelManager([
            FakeEngine("fake-clone", {"clone"}),
            FakeEngine("fake-design", {"design"}),
            FakeEngine("fake-tts", {"tts"}, speakers=["Ryan", "Vivian"]),
        ])
    from .engines.chatterbox import ChatterboxEngine
    from .engines.omnivoice import OmniVoiceEngine
    from .engines.qwen3 import qwen3_engines
    from .engines.voxcpm import VoxCPMEngine

    return ModelManager([OmniVoiceEngine(), ChatterboxEngine(), VoxCPMEngine(), *qwen3_engines()])


def default_asr():
    import os

    if os.environ.get("MORPHEU_FAKE_ENGINES") == "1":
        return None
    from .asr import AsrService

    return AsrService()


_UNSET = object()


def create_app(settings: Settings | None = None, manager: ModelManager | None = None, asr=_UNSET) -> FastAPI:
    settings = settings or Settings()
    manager = manager or default_manager()
    asr = default_asr() if asr is _UNSET else asr
    voices = VoiceStore(settings.voices_dir, settings.min_reference_seconds, settings.max_reference_seconds)
    history = History(settings.db_path, settings.outputs_dir)
    studio = Studio(manager, voices, history, asr)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.shutdown()
        history.close()

    app = FastAPI(title="Morpheu", version=__version__, lifespan=lifespan)
    app.state.studio = studio

    # ---------------- models ----------------
    @app.get("/api/v1/models")
    def list_models():
        return [{**e.info.to_dict(), "status": manager.status_of(mid)} for mid, e in manager.engines.items()]

    @app.get("/api/v1/models/active")
    def active_model():
        return manager.state.to_dict()

    @app.post("/api/v1/models/active", status_code=202)
    def activate(body: ActivateBody):
        if body.model_id not in manager.engines:
            raise HTTPException(404, "Modelo desconhecido.")
        manager.activate(body.model_id)
        return manager.state.to_dict()

    def _engine(model_id: str):
        if model_id not in manager.engines:
            raise HTTPException(404, "Modelo desconhecido.")
        return manager.engines[model_id]

    @app.get("/api/v1/tts/languages")
    def languages(model_id: str):
        return _engine(model_id).info.languages

    @app.get("/api/v1/tts/speakers")
    def speakers(model_id: str):
        return _engine(model_id).speakers()

    # ---------------- generation ----------------
    @app.post("/api/v1/tts/generate")
    def generate(body: GenerateBody):
        try:
            return studio.generate(**body.model_dump())
        except BadRequest as exc:
            raise HTTPException(400, str(exc)) from None
        except NotReady as exc:
            raise HTTPException(409, str(exc)) from None
        except GenerationFailed as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/v1/system")
    def system():
        return {"asr": asr is not None}

    # ---------------- voices ----------------
    @app.get("/api/v1/voices")
    def list_voices():
        return [v.to_dict() for v in voices.list()]

    @app.post("/api/v1/voices", status_code=201)
    async def create_voice(name: str = Form(...), transcript: str = Form(""), source: str = Form("upload"),
                           preprocess: bool = Form(True), language: str = Form("Auto"), audio: UploadFile = File(...)):
        data = await audio.read()
        if source not in ("upload", "recording"):
            raise HTTPException(400, "Origem inválida.")
        try:
            return voices.create(name, data, transcript, source=source, preprocess=preprocess,
                                 language=language).to_dict()
        except VoiceError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/v1/voices/from-history", status_code=201)
    def voice_from_history(body: SaveFromHistory):
        try:
            return studio.save_voice_from_history(body.history_id, body.name).to_dict()
        except KeyError:
            raise HTTPException(404, "Geração não encontrada.") from None
        except VoiceError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/v1/voices/{voice_id}")
    def get_voice(voice_id: str):
        try:
            return voices.get(voice_id).to_dict()
        except KeyError:
            raise HTTPException(404, "Voz não encontrada.") from None

    @app.get("/api/v1/voices/{voice_id}/audio")
    def voice_audio(voice_id: str):
        try:
            return FileResponse(voices.audio_path(voice_id), media_type="audio/wav")
        except KeyError:
            raise HTTPException(404, "Voz não encontrada.") from None

    @app.patch("/api/v1/voices/{voice_id}")
    def patch_voice(voice_id: str, body: VoicePatch):
        try:
            voice = voices.update(voice_id, body.name, body.transcript, body.language)
        except KeyError:
            raise HTTPException(404, "Voz não encontrada.") from None
        except VoiceError as exc:
            raise HTTPException(400, str(exc)) from None
        if body.transcript is not None:
            manager.forget_voice(voice_id)
        return voice.to_dict()

    @app.post("/api/v1/voices/{voice_id}/analyze")
    def analyze_voice(voice_id: str, body: AnalyzeBody | None = None):
        """Transcribe the reference (when it has no transcript), time its words, and check
        that a typed transcript matches what is actually said."""
        if asr is None:
            raise HTTPException(409, "A transcrição automática não está disponível neste modo.")
        try:
            voice = studio.references.analyze(voice_id, (body.language if body else None))
        except KeyError:
            raise HTTPException(404, "Voz não encontrada.") from None
        except ReferenceError as exc:
            raise HTTPException(422, str(exc)) from None
        manager.forget_voice(voice_id)
        return voice.to_dict()

    @app.delete("/api/v1/voices/{voice_id}", status_code=204)
    def delete_voice(voice_id: str):
        try:
            voices.delete(voice_id)
        except KeyError:
            raise HTTPException(404, "Voz não encontrada.") from None
        manager.forget_voice(voice_id)
        return Response(status_code=204)

    # ---------------- history ----------------
    @app.get("/api/v1/history")
    def list_history(limit: int = 200, offset: int = 0):
        return history.list(min(max(limit, 1), 500), max(offset, 0))

    @app.get("/api/v1/history/{gen_id}/audio")
    def history_audio(gen_id: str, download: bool = False):
        try:
            path = history.audio_path(gen_id)
        except KeyError:
            raise HTTPException(404, "Geração não encontrada.") from None
        return FileResponse(path, media_type="audio/wav", filename=f"morpheu-{gen_id[:8]}.wav" if download else None)

    @app.delete("/api/v1/history/{gen_id}", status_code=204)
    def delete_history(gen_id: str):
        try:
            history.delete(gen_id)
        except KeyError:
            raise HTTPException(404, "Geração não encontrada.") from None
        return Response(status_code=204)

    # ---------------- OpenAI-compatible ----------------
    @app.post("/v1/audio/speech")
    def openai_speech(body: SpeechBody):
        if body.model not in manager.engines:
            raise HTTPException(404, "Modelo desconhecido.")
        modes = manager.engines[body.model].info.modes
        kwargs = {"model_id": body.model, "text": body.input}
        voice = voices.find_by_name(body.voice)
        if "clone" in modes and voice:
            kwargs.update(mode="clone", voice_id=voice.id)
        elif "tts" in modes:
            kwargs.update(mode="tts", speaker=body.voice, instruct=body.instructions)
        elif "design" in modes:
            kwargs.update(mode="design", instruct=body.instructions or body.voice)
        else:
            raise HTTPException(400, f"Voz '{body.voice}' não encontrada na biblioteca.")
        try:
            item = studio.generate(**kwargs)
        except BadRequest as exc:
            raise HTTPException(400, str(exc)) from None
        except NotReady as exc:
            raise HTTPException(409, str(exc)) from None
        except GenerationFailed as exc:
            raise HTTPException(422, str(exc)) from None
        return Response(history.audio_path(item["id"]).read_bytes(), media_type="audio/wav")

    @app.get("/api/v1/health")
    def health():
        return {"ok": True, "version": __version__}

    # ---------------- UI ----------------
    if settings.frontend_dir.joinpath("index.html").exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")

    return app
