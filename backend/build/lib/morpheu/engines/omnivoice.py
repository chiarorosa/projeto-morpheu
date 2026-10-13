"""OmniVoice (k2-fsa, Apache-2.0): fast diffusion-style cloning in 600+ languages."""

import gc

import numpy as np

from .base import Engine, GenerationRequest, ModelInfo, ParamSpec

REPO = "k2-fsa/OmniVoice"

LANGUAGES = ["Auto", "Portuguese", "English", "Spanish", "French", "German", "Italian", "Dutch", "Polish",
             "Russian", "Turkish", "Arabic", "Chinese", "Japanese", "Korean", "Hindi", "Indonesian",
             "Vietnamese", "Thai", "Ukrainian", "Swedish", "Greek", "Hebrew", "Romanian", "Czech"]

PARAMS = [
    ParamSpec("speed", "Velocidade", "Ritmo da fala. 0,8 soa mais natural para narração; 1,0 é o ritmo do modelo.", "float", 0.8, 0.5, 2.0, 0.05),
    ParamSpec("num_step", "Passos de refinamento", "Mais passos = mais qualidade e mais tempo.", "int", 32, 4, 64, 1),
    ParamSpec("guidance_scale", "Força da orientação", "Quanto a geração segue o texto e a voz de referência.", "float", 2.0, 0.0, 3.0, 0.05),
    ParamSpec("denoise", "Remover ruído", "Gera como se a referência fosse limpa, sem o ruído de fundo dela.", "bool", True),
]


class OmniVoiceEngine(Engine):
    def __init__(self):
        self.info = ModelInfo(
            id="omnivoice",
            name="OmniVoice 0.8B",
            description="Clonagem muito rápida e fiel ao timbre, com suporte a mais de 600 idiomas.",
            modes=["clone"],
            languages=LANGUAGES,
            params=PARAMS,
            vram_gb=3,
            ref_min_sec=3.0,
            ref_max_sec=10.0,
            max_chunk_chars=600,
            sample_rate=24000,
            recommended_for=["Portuguese"],
        )
        self._model = None
        self._prompts: dict[str, tuple[bool, object]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress=lambda f, m: None) -> None:
        import torch

        from ..compat import apply_all, download
        from ..vendor.higgs_tokenizer import install

        apply_all()
        install()
        if not torch.cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA disponível.")
        progress(0.05, "Baixando/verificando arquivos do modelo…")
        local = download(REPO)
        progress(0.6, "Carregando na GPU…")
        from omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(local, device_map="cuda:0", dtype=torch.float16, load_asr=False)
        # The audio codec loses precision in half precision, so it stays in float32.
        if hasattr(model, "audio_tokenizer"):
            model.audio_tokenizer.to(dtype=torch.float32)
        self._model = model
        progress(1.0, "Pronto")

    def unload(self) -> None:
        import torch

        self._model = None
        self._prompts.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def forget_voice(self, voice_id: str) -> None:
        for key in [k for k in self._prompts if k.split(":", 1)[0] == voice_id]:
            del self._prompts[key]

    def _prompt(self, req: GenerationRequest, preprocess: bool):
        key = req.ref_key or req.ref_audio_path
        cached = self._prompts.get(key)
        if cached and cached[0] == preprocess:
            return cached[1]
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=req.ref_audio_path, ref_text=req.ref_text, preprocess_prompt=preprocess,
        )
        self._prompts[key] = (preprocess, prompt)
        return prompt

    def generate(self, req: GenerationRequest) -> tuple[np.ndarray, int]:
        import torch
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        p = req.params
        torch.manual_seed(p["seed"])
        config = OmniVoiceGenerationConfig(
            num_step=p["num_step"], guidance_scale=p["guidance_scale"], denoise=p["denoise"],
            preprocess_prompt=p["preprocess_prompt"], postprocess_output=p["postprocess_output"],
        )
        audio = self._model.generate(
            text=req.text,
            language=None if req.language == "Auto" else req.language,
            voice_clone_prompt=self._prompt(req, p["preprocess_prompt"]),
            speed=p["speed"],
            generation_config=config,
        )
        return audio[0].float().cpu().numpy().squeeze(), int(self._model.sampling_rate)
