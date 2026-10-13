"""Choose which stretch of a reference clip the model hears.

A long reference, or one cut mid-sentence, makes the model speak the reference's last
words before the target text or halt and repeat. So the clip is cut to its most fluent
stretch that starts and ends on a sentence boundary and fits the model's limit, and the
transcript is sliced to match. Word timings come from forced alignment (see asr.py).
"""

import re
from dataclasses import dataclass

import numpy as np

MIN_WINDOW_SEC = 4.0
# Silence kept around the window edges (never past the neighbouring word).
EDGE_PAD_SEC = 0.25
# A clip's last word only ends a sentence when at least this much silence follows it:
# a transcriber also puts a period on speech that was cut off.
END_PAUSE_SEC = 0.3
# A stretch is fluent when pauses take at most this share of it and none is longer than
# _MAX_FLUENT_GAP_SEC; a longer pause inside the reference makes the model pause there too.
_FLUENT_SILENCE_RATIO = 0.20
_MAX_FLUENT_GAP_SEC = 0.6
# Windows up to this much past the limit are considered only when nothing fits under it.
TOLERANCE_SEC = 2.0
# Absolute ceiling, whatever the model declares.
ABS_MAX_SEC = 30.0

_SENTENCE_END_RE = re.compile(r"[.!?…。！？][\"'”’)\]]*$")
_LEADING_QUOTES_RE = re.compile(r"^[\"'(\[“‘]+")


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    @staticmethod
    def from_dict(d: dict) -> "Word":
        return Word(str(d["text"]), float(d["start"]), float(d["end"]))

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    text: str
    reason: str  # "sentence" | "pause" | "cut"

    @property
    def duration(self) -> float:
        return self.end - self.start


def clean_words(words: list[Word], duration: float) -> list[Word]:
    """Drop timings that cannot be right (reversed, out of order, past the clip end). Zero-length
    words stay: the aligner gives short words like "o" or "a" no duration of their own."""
    out: list[Word] = []
    for w in words:
        if not w.text.strip() or w.end < w.start or w.start < 0 or w.end > duration + 0.05:
            continue
        if out and w.start < out[-1].start:
            continue
        out.append(w)
    return out


def _starts_sentence(word: str) -> bool:
    w = _LEADING_QUOTES_RE.sub("", word)
    return bool(w) and not ("a" <= w[0] <= "z")


def sentence_end_indices(words: list[Word], duration: float, end_pause: float = END_PAUSE_SEC) -> list[int]:
    """Indices of words that end a sentence. Pass end_pause=0 when the punctuation was typed by
    a person who heard the whole clip, so the final full stop can be trusted."""
    ends = []
    for i, w in enumerate(words):
        if not _SENTENCE_END_RE.search(w.text.strip()):
            continue
        if i == len(words) - 1:
            if duration - w.end >= end_pause:
                ends.append(i)
        elif _starts_sentence(words[i + 1].text):
            ends.append(i)
    return ends


def _edges(words: list[Word], i: int, j: int, duration: float) -> tuple[float, float]:
    before = words[i - 1].end if i > 0 else 0.0
    after = words[j + 1].start if j + 1 < len(words) else duration
    start = max(before + (words[i].start - before) / 2, words[i].start - EDGE_PAD_SEC, 0.0)
    end = min(after - (after - words[j].end) / 2, words[j].end + EDGE_PAD_SEC, duration)
    return start, end


def _gaps(words: list[Word], i: int, j: int) -> list[float]:
    return [max(0.0, words[k + 1].start - words[k].end) for k in range(i, j)]


def _silence(words: list[Word], i: int, j: int) -> float:
    return sum(_gaps(words, i, j))


def best_sentence_window(words: list[Word], max_sec: float, duration: float,
                         min_sec: float = MIN_WINDOW_SEC, tolerance: float = TOLERANCE_SEC,
                         end_pause: float = END_PAUSE_SEC) -> Window | None:
    """Sentence-aligned stretch of min_sec..max_sec to clone from.

    Among fluent stretches (few and short pauses) the longest wins, since more reference
    speech clones better; halting delivery makes the model halt and repeat, so when no
    stretch is fluent the one with the smallest share of silence wins instead."""
    ends = sentence_end_indices(words, duration, end_pause)
    if not ends:
        return None
    starts = [0] + [e + 1 for e in ends if e + 1 < len(words)]

    def pick(limit: float) -> Window | None:
        best = None
        for i in starts:
            for j in ends:
                if j < i:
                    continue
                s, e = _edges(words, i, j, duration)
                length = e - s
                if length < min_sec or length > limit:
                    continue
                gaps = _gaps(words, i, j)
                ratio = sum(gaps) / length
                fluent = ratio <= _FLUENT_SILENCE_RATIO and max(gaps, default=0.0) <= _MAX_FLUENT_GAP_SEC
                key = (0, 0.0, -length) if fluent else (1, round(ratio, 2), -length)
                if best is None or key < best[0]:
                    best = (key, s, e, i, j)
        if best is None:
            return None
        _, s, e, i, j = best
        return Window(s, e, slice_text(words, i, j), "sentence")

    return pick(max_sec) or pick(min(max_sec + tolerance, ABS_MAX_SEC))


def word_gap_window(words: list[Word], max_sec: float, duration: float,
                    min_sec: float = MIN_WINDOW_SEC, min_gap: float = 0.12) -> Window | None:
    """From the start up to the middle of the last pause between words before max_sec."""
    best = None
    for k in range(len(words) - 1):
        gap = words[k + 1].start - words[k].end
        cut = words[k].end + gap / 2
        if gap >= min_gap and min_sec <= cut <= max_sec:
            best = k
    if best is None:
        return None
    start, _ = _edges(words, 0, best, duration)
    end = words[best].end + (words[best + 1].start - words[best].end) / 2
    return Window(start, end, slice_text(words, 0, best), "pause")


def slice_text(words: list[Word], i: int, j: int) -> str:
    text = " ".join(w.text.strip() for w in words[i:j + 1])
    return re.sub(r"\s+([,.!?;:)\]])", r"\1", text)


def last_pause_cut(samples: np.ndarray, sr: int, cap_sec: float, min_pause: float = 0.12) -> float:
    """Middle of the last quiet stretch before cap_sec (energy-based, for clips without timings).
    Falls back to cap_sec when no pause sits in the back half."""
    hop = int(0.01 * sr)
    n = min(len(samples), int(cap_sec * sr)) // hop
    if n < 10:
        return cap_sec
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    quiet = db < -35.0
    best = None
    k = 0
    while k < n:
        if quiet[k]:
            j = k
            while j < n and quiet[j]:
                j += 1
            if (j - k) * 0.01 >= min_pause and k * 0.01 >= cap_sec / 2:
                best = (k + j) / 2 * 0.01
            k = j
        else:
            k += 1
    return best if best is not None else cap_sec
