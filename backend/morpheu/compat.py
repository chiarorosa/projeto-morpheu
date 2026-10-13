"""Runtime adjustments that let the engine libraries share one Python environment."""

import numpy as np
import soundfile as sf

_applied = False


def configure_audio_backend() -> None:
    """torchaudio >= 2.9 routes load()/save() through torchcodec, which is not installed.
    Engines only read and write plain files, so route both through soundfile."""
    import torch
    import torchaudio

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None,
             buffer_size=4096, backend=None):
        data, sr = sf.read(uri, start=frame_offset, frames=num_frames, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(np.ascontiguousarray(data))
        return (tensor.T if channels_first else tensor), sr

    def save(uri, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None,
             buffer_size=4096, backend=None, compression=None):
        data = src.detach().to(torch.float32).cpu().numpy()
        if data.ndim == 2 and channels_first:
            data = data.T
        subtype = {8: "PCM_U8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(bits_per_sample or 16, "PCM_16")
        fmt = (format or "WAV").upper() if not hasattr(uri, "write") or format else "WAV"
        sf.write(uri, data, sample_rate, format=fmt, subtype=subtype)

    torchaudio.load = load
    torchaudio.save = save


def apply_all() -> None:
    global _applied
    if _applied:
        return
    configure_audio_backend()
    _applied = True


def download(repo_id: str, **kwargs) -> str:
    """snapshot_download, minus a Windows race: huggingface_hub marks a folder as symlink-capable
    before testing it, so parallel workers try symlinks the OS then refuses (WinError 1314).
    Running the check once, alone, first makes every worker see the real answer."""
    import os

    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.file_download import are_symlinks_supported, repo_folder_name

    storage = os.path.join(constants.HF_HUB_CACHE, repo_folder_name(repo_id=repo_id, repo_type="model"))
    os.makedirs(storage, exist_ok=True)
    are_symlinks_supported(storage)
    return snapshot_download(repo_id, **kwargs)
