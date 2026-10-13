"""Run OmniVoice 0.1.2 (written for transformers 5) on the transformers 4.57 Qwen3-TTS needs.

`install()` registers the HiggsAudioV2 tokenizer backported from transformers 5.3.0
(Apache-2.0, see NOTICE) and teaches 4.57 two transformers 5 checkpoint conventions.
"""


def _accept_v5_extra_special_tokens() -> None:
    """transformers 5 saves `extra_special_tokens` as a list; 4.57 expects a name->token dict.
    The listed tokens are already in the tokenizer's vocabulary, so a list is simply skipped."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    original = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(original, "_morpheu_patched", False):
        return

    def patched(self, special_tokens):
        if isinstance(special_tokens, (list, tuple)):
            self.extra_special_tokens = {}
            special_tokens = {}
        return original(self, special_tokens)

    patched._morpheu_patched = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


def _accept_v5_rope_parameters() -> None:
    """transformers 5 configs nest RoPE settings under `rope_parameters`; 4.57 reads a flat
    `rope_theta` and would silently fall back to 10000 (OmniVoice was trained with 1e6),
    which scrambles positions and turns the output into babble."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    original = Qwen3Config.__init__
    if getattr(original, "_morpheu_patched", False):
        return

    def patched(self, *args, rope_parameters=None, **kwargs):
        if rope_parameters:
            kwargs.setdefault("rope_theta", rope_parameters.get("rope_theta", 10000.0))
            if rope_parameters.get("rope_type", "default") != "default":
                kwargs.setdefault("rope_scaling", dict(rope_parameters))
        original(self, *args, **kwargs)

    patched._morpheu_patched = True
    Qwen3Config.__init__ = patched


def install() -> None:
    import transformers
    from transformers import AutoConfig, AutoModel

    _accept_v5_extra_special_tokens()
    _accept_v5_rope_parameters()

    if getattr(transformers, "HiggsAudioV2TokenizerModel", None) is not None:
        return
    from .configuration_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerConfig
    from .modeling_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

    AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig, exist_ok=True)
    AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, exist_ok=True)
    transformers.HiggsAudioV2TokenizerModel = HiggsAudioV2TokenizerModel
    transformers.HiggsAudioV2TokenizerConfig = HiggsAudioV2TokenizerConfig
