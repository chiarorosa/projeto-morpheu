"""Qwen3-TTS engines (Apache-2.0 models by the Qwen team): clone, design and built-in voices."""

import gc
import os

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

SAMPLING_PARAMS = [
    ParamSpec("temperature", "Variação", "Mais alto = mais expressivo e variado entre gerações; mais baixo = consistente e previsível.", "float", 0.9, 0.1, 1.5, 0.05),
    ParamSpec("top_p", "Ousadia", "Mais baixo fica nos sons mais prováveis (leitura estável); mais alto deixa passar escolhas mais ousadas.", "float", 0.95, 0.1, 1.0, 0.05),
    ParamSpec("top_k", "Candidatos", "Quantos sons candidatos são considerados a cada passo. Valores muito baixos deixam a fala monótona.", "int", 50, 1, 100, 1),
    ParamSpec("repetition_penalty", "Controle de repetição", "Mais alto evita gaguejos e loops; alto demais soa artificial.", "float", 1.05, 1.0, 2.0, 0.01),
]

# Display names used by the UI; qwen-tts itself matches languages case-insensitively.
LANGUAGES = ["Auto", "Portuguese", "English", "Spanish", "French", "German", "Italian", "Russian", "Chinese", "Japanese", "Korean"]


def _max_new_tokens(text: str) -> int:
    """Cap generation by text length (12 codec frames per second of speech), so a model that
    misses its end-of-speech token stops after a plausible length instead of running to 2048."""
    return min(2048, max(256, 4 * len(text) + 128))


class Qwen3Engine(Engine):
    def __init__(self, info: ModelInfo, repo_id: str):
        self.info = info
        self.repo_id = repo_id
        self._model = None
        self._prompts: dict[str, tuple[float, str | None, object]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress=lambda f, m: None) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        from ..compat import download

        if not torch.cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA disponível; os modelos Qwen3-TTS precisam de uma GPU NVIDIA.")
        progress(0.05, "Baixando/verificando arquivos do modelo…")
        local = download(self.repo_id)
        progress(0.6, "Carregando na GPU…")
        self._model = Qwen3TTSModel.from_pretrained(
            local,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=os.environ.get("MORPHEU_ATTN", "sdpa"),
        )
        progress(1.0, "Pronto")

    def unload(self) -> None:
        import torch

        self._model = None
        self._prompts.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def speakers(self) -> list[str]:
        if "tts" not in self.info.modes:
            return []
        if self._model is not None:
            return [s.title() for s in (self._model.get_supported_speakers() or [])]
        return list(QWEN3_BUILTIN_SPEAKERS)

    def forget_voice(self, voice_id: str) -> None:
        for key in [k for k in self._prompts if k.split(":", 1)[0] == voice_id]:
            del self._prompts[key]

    def _clone_prompt(self, req: GenerationRequest):
        """Extract reference features once per voice window and reuse them across generations."""
        key = req.ref_key or req.voice_id
        mtime = os.path.getmtime(req.ref_audio_path)
        cached = self._prompts.get(key)
        if cached and cached[0] == mtime and cached[1] == req.ref_text:
            return cached[2]
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=req.ref_audio_path,
            ref_text=req.ref_text,
            # Without a transcript only the speaker embedding is used (lower fidelity, but works).
            x_vector_only_mode=not req.ref_text,
        )
        self._prompts[key] = (mtime, req.ref_text, prompt)
        return prompt

    def generate(self, req: GenerationRequest) -> tuple[np.ndarray, int]:
        import torch

        p = req.params
        torch.manual_seed(p["seed"])
        sampling = {k: p[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")}
        # The code predictor (sub-talker) samples the finer codec layers; give it the same settings
        # so the knobs the UI shows act on the whole voice, not only on the first layer.
        sampling.update(subtalker_temperature=p["temperature"], subtalker_top_p=p["top_p"],
                        subtalker_top_k=p["top_k"], max_new_tokens=_max_new_tokens(req.text))
        language = req.language or "Auto"

        if req.mode == "clone":
            wavs, sr = self._model.generate_voice_clone(
                text=req.text, language=language, voice_clone_prompt=self._clone_prompt(req), **sampling,
            )
        elif req.mode == "design":
            wavs, sr = self._model.generate_voice_design(text=req.text, instruct=req.instruct, language=language, **sampling)
        else:
            wavs, sr = self._model.generate_custom_voice(
                text=req.text, speaker=req.speaker, language=language, instruct=req.instruct, **sampling,
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)


QWEN3_BUILTIN_SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"]


def qwen3_engines() -> list[Engine]:
    def info(id_, name, desc, modes, vram):
        return ModelInfo(id=id_, name=name, description=desc, modes=modes, languages=LANGUAGES, params=SAMPLING_PARAMS,
                         vram_gb=vram, ref_min_sec=1.5, ref_max_sec=10.0, max_chunk_chars=500)

    return [
        Qwen3Engine(info("qwen3-clone-0.6b", "Qwen3 Clone 0.6B",
                         "Clonagem leve e rápida a partir de alguns segundos de áudio.", ["clone"], 3),
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        Qwen3Engine(info("qwen3-clone-1.7b", "Qwen3 Clone 1.7B",
                         "Clonagem de maior qualidade. Mais lenta e usa mais VRAM.", ["clone"], 6),
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        Qwen3Engine(info("qwen3-design", "Qwen3 Design 1.7B",
                         "Cria uma voz nova a partir de uma descrição em linguagem natural.", ["design"], 6),
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        Qwen3Engine(info("qwen3-voices", "Qwen3 Vozes prontas 1.7B",
                         "Vozes embutidas com controle de tom e emoção por instrução.", ["tts"], 6),
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    ]
