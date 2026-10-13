"""Small, dependency-light audio helpers (numpy + soundfile)."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def read_audio(source) -> tuple[np.ndarray, int]:
    """Read a path or raw bytes into mono float32."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    data, sr = sf.read(source, dtype="float32", always_2d=False)
    return to_mono(data), int(sr)


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, format="WAV", subtype="PCM_16")


def wav_bytes(data: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(data, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -40.0, margin_s: float = 0.1) -> np.ndarray:
    """Cut leading/trailing audio quieter than threshold_db relative to the peak."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak == 0.0:
        return x
    threshold = peak * (10 ** (threshold_db / 20))
    loud = np.flatnonzero(np.abs(x) > threshold)
    margin = int(margin_s * sr)
    start = max(0, loud[0] - margin)
    end = min(len(x), loud[-1] + 1 + margin)
    return x[start:end]


def normalize_peak(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if peak == 0.0 else (x * (target / peak)).astype(np.float32)


def preprocess_reference(x: np.ndarray, sr: int) -> np.ndarray:
    """'Clean reference': trim silence and even out the level before cloning."""
    return normalize_peak(trim_silence(x, sr), 0.9)


def postprocess_output(x: np.ndarray, sr: int) -> np.ndarray:
    """'Polish output': trim dead air and level the result."""
    return normalize_peak(trim_silence(x, sr, threshold_db=-50.0, margin_s=0.15), 0.95)
