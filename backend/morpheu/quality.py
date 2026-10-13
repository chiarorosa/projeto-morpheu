"""Output checks and finishing: catch runaway generations, then level the result."""

import numpy as np

# Natural speech runs roughly 10-20 characters per second across the supported languages.
_SLOWEST_CHARS_PER_SEC = 6.0
_SLACK_SEC = 4.0
# A silence this long between the first and last word means the model collapsed mid-sentence.
_MAX_INTERIOR_SILENCE_SEC = 3.5


def is_runaway(seconds: float, text: str) -> bool:
    """True when the audio is implausibly long for its text (the model looped or never stopped)."""
    chars = len("".join(text.split()))
    if not chars or seconds <= 0:
        return False
    return seconds > chars / _SLOWEST_CHARS_PER_SEC + _SLACK_SEC


def interior_silence_seconds(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    """Longest silent run strictly between the first and last voiced 20 ms frame."""
    hop = int(0.02 * sr)
    n = len(x) // hop
    if n < 3:
        return 0.0
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    voiced = np.flatnonzero(db > floor_db)
    if len(voiced) < 2:
        return 0.0
    gaps = np.diff(voiced) - 1
    return float(gaps.max()) * 0.02


def problem(x: np.ndarray, sr: int, text: str) -> str | None:
    """Why this output should be regenerated, or None when it looks healthy."""
    seconds = len(x) / sr
    if seconds < 0.2 or float(np.max(np.abs(x)) if len(x) else 0.0) < 1e-3:
        return "saída vazia"
    if is_runaway(seconds, text):
        return f"áudio longo demais para o texto ({seconds:.1f} s)"
    if interior_silence_seconds(x, sr) > _MAX_INTERIOR_SILENCE_SEC:
        return "silêncio longo no meio da fala"
    return None


def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float = -18.0, peak_db: float = -1.0) -> np.ndarray:
    """Bring integrated loudness to target_lufs without letting peaks pass peak_db."""
    import pyloudnorm

    if len(x) < int(0.4 * sr):
        return x
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(x.astype(np.float64))
    if not np.isfinite(loudness):
        return x
    y = x * (10 ** ((target_lufs - loudness) / 20))
    ceiling = 10 ** (peak_db / 20)
    peak = float(np.max(np.abs(y)))
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def join(parts: list[np.ndarray], sr: int, pause_sec: float = 0.28, fade_sec: float = 0.01) -> np.ndarray:
    """Concatenate chunk audio with a short pause and tiny fades so no click lands at the seams."""
    fade = max(1, int(fade_sec * sr))
    gap = np.zeros(int(pause_sec * sr), dtype=np.float32)
    out = []
    for i, p in enumerate(parts):
        p = np.asarray(p, dtype=np.float32).copy()
        if len(p) > 2 * fade:
            p[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            p[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        if i:
            out.append(gap)
        out.append(p)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
