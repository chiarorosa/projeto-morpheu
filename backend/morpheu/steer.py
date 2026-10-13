"""Emotion steering for Qwen3-TTS 1.7B: push a cloned voice toward an emotion without changing who speaks.

Each emotion is a direction in the talker's residual stream, one 2048-d row per layer
(vendor/emotion_steer, MIT, from Gabriele Mastrapasqua's qwen3-tts C engine). While the
model generates, the direction is added after layers 21-25, the hidden state is rescaled to
its original norm (so only its direction moves), and the weight decays a little every frame
so the emotion colours the delivery instead of overdriving it. The prompt is never steered.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VECTOR_DIR = Path(__file__).parent / "vendor" / "emotion_steer"
_MAGIC = 0x54534C51  # "QLST"

LAYER_FIRST = 21
LAYER_LAST = 25
DECAY = 0.985


@dataclass(frozen=True)
class Emotion:
    key: str
    label: str
    # Weight at full intensity. The upstream manifest marks fear and surprise as turning
    # metallic above ~w4-6 and anger as harsh, so those sit lower than the rest.
    ceiling: float
    decay: float = DECAY


EMOTIONS = [
    Emotion("joy", "Alegria", 10.0),
    Emotion("sad", "Tristeza", 12.0),
    Emotion("ang", "Raiva", 8.0),
    Emotion("fear", "Medo", 6.0),
    Emotion("surprise", "Surpresa", 6.0),
    Emotion("disgust", "Nojo", 10.0),
    # Blends of two primaries (upstream dyads).
    Emotion("awe", "Admiração", 10.0),
    Emotion("nostalgia", "Nostalgia", 10.0),
    Emotion("contempt", "Desprezo", 10.0, decay=0.988),
    Emotion("disapproval", "Desaprovação", 10.0),
    Emotion("remorse", "Remorso", 10.0),
    Emotion("outrage", "Indignação", 8.0),
    Emotion("despair", "Desespero", 10.0),
]
BY_KEY = {e.key: e for e in EMOTIONS}

INTENSITIES = {"leve": 0.45, "normal": 0.7, "forte": 1.0}


def available() -> list[Emotion]:
    return [e for e in EMOTIONS if (VECTOR_DIR / f"ryan_{e.key}.qlsteer").exists()]


def load_vector(key: str) -> np.ndarray:
    """Read a .qlsteer file: uint32 magic, int32 layers, int32 dim, then layers x dim float32."""
    raw = (VECTOR_DIR / f"ryan_{key}.qlsteer").read_bytes()
    magic, layers, dim = struct.unpack("<Iii", raw[:12])
    if magic != _MAGIC or len(raw) != 12 + layers * dim * 4:
        raise ValueError(f"vetor de emoção inválido: {key}")
    return np.frombuffer(raw[12:], dtype="<f4").reshape(layers, dim).copy()


def weight_for(key: str, intensity: str) -> float:
    return BY_KEY[key].ceiling * INTENSITIES[intensity]


class Steerer:
    """Context manager that steers a Qwen3-TTS talker while generating.

    The first forward pass of a generate call is the prompt prefill (many positions at
    once) and is left untouched; every later pass produces one frame, steered with weight
    `weight * decay ** frame`."""

    def __init__(self, layers, vector: np.ndarray, weight: float, decay: float = DECAY,
                 first: int = LAYER_FIRST, last: int = LAYER_LAST):
        import torch

        if vector.shape[0] < last + 1:
            raise ValueError("o vetor tem menos camadas que o modelo")
        param = next(layers[0].parameters(), None)
        self.layers = layers
        self.vector = torch.as_tensor(vector, dtype=torch.float32,
                                      device=param.device if param is not None else "cpu")
        self.weight, self.decay, self.first, self.last = weight, decay, first, last
        self.frame = -1
        self._handles = []

    def _hook(self, idx: int):
        def fn(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if idx == self.first:
                # One decode step = one frame; the prefill sees the whole prompt at once.
                self.frame = self.frame + 1 if hidden.shape[1] == 1 else -1
            if self.frame < 0 or hidden.shape[1] != 1:
                return output
            w = self.weight * (self.decay ** self.frame)
            h = hidden.float()
            before = h.norm(dim=-1, keepdim=True)
            h = h + w * self.vector[idx]
            h = h * (before / h.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            h = h.to(hidden.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return fn

    def __enter__(self) -> "Steerer":
        self.frame = -1
        for i in range(self.first, self.last + 1):
            self._handles.append(self.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def talker_layers(qwen_tts_model):
    """Decoder layers of the talker inside a qwen_tts.Qwen3TTSModel."""
    return qwen_tts_model.model.talker.model.layers


# ---------------- finishing a steered take ----------------

def _block_levels_db(y: np.ndarray, blocks: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed RMS level per block over time, skipping silent blocks."""
    edges = np.linspace(0, len(y), blocks + 1).astype(int)
    pos, lvl = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        if len(seg) == 0:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 1e-4:
            pos.append((a + b) / 2 / len(y))
            lvl.append(20 * np.log10(rms))
    if len(lvl) < 4:
        return np.array(pos), np.array(lvl)
    lvl = np.convolve(np.pad(lvl, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return np.array(pos), lvl


def measure_fade(y: np.ndarray) -> float:
    """dB the level declines from the first quarter to the last quarter of the take."""
    pos, lvl = _block_levels_db(y)
    if len(lvl) < 4:
        return 0.0
    q = max(1, len(lvl) // 4)
    return float(np.mean(lvl[-q:]) - np.mean(lvl[:q]))


def flatten_fade(y: np.ndarray, max_decline_db: float = -1.5) -> np.ndarray:
    """A decaying direction can leave the end of a take quieter than its start; hold that
    decline to max_decline_db by turning the opening down along the measured trend
    (boosting the tail would clip), then restore the original peak."""
    fade = measure_fade(y)
    if fade >= max_decline_db:
        return y
    pos, lvl = _block_levels_db(y)
    excess = (lvl - lvl[-1]) * (1 - max_decline_db / fade)
    gain_db = -np.clip(excess, 0, None)
    curve = np.interp(np.linspace(0, 1, len(y)), pos, gain_db)
    out = y * (10 ** (curve / 20))
    peak, new_peak = float(np.max(np.abs(y))), float(np.max(np.abs(out)))
    return (out * (peak / new_peak)).astype(np.float32) if new_peak > 0 else out.astype(np.float32)
