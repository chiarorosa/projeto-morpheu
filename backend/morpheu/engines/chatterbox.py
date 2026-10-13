"""Chatterbox Multilingual (Resemble AI, MIT): zero-shot cloning in 23 languages, Portuguese included."""

import gc
import os
import tempfile

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

REPO = "ResembleAI/chatterbox"
FILES = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
         "conds.pt", "Cangjie5_TC.json"]

LANGUAGES = {
    "Portuguese": "pt", "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Dutch": "nl", "Polish": "pl", "Russian": "ru", "Turkish": "tr", "Arabic": "ar", "Chinese": "zh",
    "Japanese": "ja", "Korean": "ko", "Hindi": "hi", "Danish": "da", "Finnish": "fi", "Greek": "el",
    "Hebrew": "he", "Malay": "ms", "Norwegian": "no", "Swedish": "sv", "Swahili": "sw",
}

PARAMS = [
    ParamSpec("exaggeration", "Expressividade", "Intensidade emocional. 0,5 é neutro; valores altos dramatizam e aceleram a fala.", "float", 0.5, 0.0, 2.0, 0.05),
    ParamSpec("cfg_weight", "Fidelidade ao ritmo", "Quanto segue o ritmo da referência. Baixe para ~0,3 se a referência for rápida; 0 reduz sotaque estrangeiro.", "float", 0.5, 0.0, 1.0, 0.05),
    ParamSpec("temperature", "Variação", "Mais alto = mais variado entre gerações; mais baixo = estável.", "float", 0.8, 0.05, 2.0, 0.05),
    ParamSpec("repetition_penalty", "Controle de repetição", "Evita gaguejos e loops.", "float", 2.0, 1.0, 3.0, 0.1),
]

# Chatterbox refuses references under 5 s; a slightly shorter clip is padded with silence.
_MIN_REF_SEC = 5.1


def _pad_reference(path: str, min_seconds: float) -> tuple[str, str | None]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if len(data) / sr >= min_seconds:
        return path, None
    pad = np.zeros(int(min_seconds * sr) - len(data), dtype=np.float32)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, np.concatenate([data if data.ndim == 1 else data.mean(axis=1), pad]), sr)
    return tmp, tmp


class ChatterboxEngine(Engine):
    def __init__(self):
        self.info = ModelInfo(
            id="chatterbox-multilingual",
            name="Chatterbox Multilingual 0.5B",
            description="Clonagem com boa pronúncia em português e controle de expressividade. Leve e rápido.",
            modes=["clone"],
            languages=list(LANGUAGES),
            params=PARAMS,
            vram_gb=4,
            ref_min_sec=3.0,
            ref_max_sec=10.0,
            uses_ref_text=False,
            max_chunk_chars=280,
            sample_rate=24000,
            recommended_for=["Portuguese"],
        )
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress=lambda f, m: None) -> None:
        import torch

        from ..compat import apply_all, download

        apply_all()
        if not torch.cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA disponível.")
        progress(0.05, "Baixando/verificando arquivos do modelo…")
        local = download(REPO, allow_patterns=FILES)
        progress(0.6, "Carregando na GPU…")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_local(local, device="cuda")
        # The alignment analyzer that stops generation at the end of the text reads attention
        # maps, which the default SDPA kernels do not return.
        model.t3.tfmr.config._attn_implementation = "eager"
        self._model = model
        progress(1.0, "Pronto")

    def unload(self) -> None:
        import torch

        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, req: GenerationRequest) -> tuple[np.ndarray, int]:
        import torch

        p = req.params
        torch.manual_seed(p["seed"])
        ref, tmp = _pad_reference(req.ref_audio_path, _MIN_REF_SEC)
        try:
            wav = self._model.generate(
                req.text,
                language_id=LANGUAGES[req.language],
                audio_prompt_path=ref,
                exaggeration=p["exaggeration"],
                cfg_weight=p["cfg_weight"],
                temperature=p["temperature"],
                repetition_penalty=p["repetition_penalty"],
            )
        finally:
            if tmp:
                os.unlink(tmp)
        return wav.squeeze(0).float().cpu().numpy(), int(self._model.sr)
