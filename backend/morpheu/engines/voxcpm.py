"""VoxCPM2 (OpenBMB, Apache-2.0): 2B tokenizer-free TTS with 48 kHz output, 30 languages."""

import gc
import logging

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

REPO = "openbmb/VoxCPM2"

LANGUAGES = ["Auto", "Portuguese", "English", "Spanish", "French", "German", "Italian", "Dutch", "Polish",
             "Russian", "Turkish", "Arabic", "Chinese", "Japanese", "Korean", "Hindi", "Indonesian",
             "Vietnamese", "Thai", "Danish", "Finnish", "Greek", "Hebrew", "Malay", "Norwegian", "Swedish",
             "Swahili", "Tagalog", "Burmese", "Khmer", "Lao"]

PARAMS = [
    ParamSpec("cfg_value", "Fidelidade à referência", "Mais alto segue mais de perto a voz e o texto; alto demais soa forçado.", "float", 2.0, 1.0, 5.0, 0.1),
    ParamSpec("inference_timesteps", "Passos de refinamento", "Mais passos = mais detalhe e mais tempo.", "int", 10, 4, 32, 2),
]


class VoxCPMEngine(Engine):
    def __init__(self):
        self.info = ModelInfo(
            id="voxcpm2",
            name="VoxCPM2 2B",
            description="A maior fidelidade de timbre e áudio em 48 kHz. Mais pesado e mais lento.",
            modes=["clone"],
            languages=LANGUAGES,
            params=PARAMS,
            vram_gb=6,
            ref_min_sec=3.0,
            ref_max_sec=15.0,
            max_chunk_chars=400,
            sample_rate=48000,
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
        local = download(REPO)
        progress(0.6, "Carregando na GPU…")
        from voxcpm import VoxCPM

        logging.getLogger("voxcpm").setLevel(logging.WARNING)
        # torch.compile needs Triton, which has no official Windows build.
        self._model = VoxCPM.from_pretrained(local, load_denoiser=False, optimize=False, device="cuda")
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
        if req.ref_text:
            # Continuation mode: the model hears the reference and its words, then keeps talking.
            source = {"prompt_wav_path": req.ref_audio_path, "prompt_text": req.ref_text}
        else:
            source = {"reference_wav_path": req.ref_audio_path}
        wav = self._model.generate(
            text=req.text, cfg_value=p["cfg_value"], inference_timesteps=p["inference_timesteps"],
            normalize=False, denoise=False, **source,
        )
        return np.asarray(wav, dtype=np.float32).squeeze(), int(self._model.tts_model.sample_rate)
