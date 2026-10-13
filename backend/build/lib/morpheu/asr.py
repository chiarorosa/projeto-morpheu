"""Speech recognition for reference clips: transcripts and per-word timings.

Qwen3-ASR transcribes a clip that came without a transcript, and Qwen3-ForcedAligner
times every word of a transcript, which is what lets the reference be cut at sentence
boundaries (refwindow.py). Both are loaded on demand and released right after use.
"""

import gc
import re
import unicodedata
from dataclasses import dataclass

from .refwindow import Word

ASR_REPO = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_LANGUAGES = {"Portuguese", "English", "Spanish", "French", "German", "Italian", "Russian",
                 "Chinese", "Japanese", "Korean", "Dutch", "Polish", "Turkish", "Arabic"}


@dataclass
class Transcription:
    text: str
    language: str
    words: list[Word]


def _norm(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).lower()
    return "".join(ch for ch in token if ch.isalnum())


def attach_punctuation(transcript: str, items: list[Word]) -> list[Word]:
    """The aligner times bare words; map them back onto the transcript's tokens so the
    punctuation that marks sentence ends survives. Returns the bare items when the two
    disagree (refwindow then falls back to cutting at pauses)."""
    tokens = [t for t in transcript.split() if _norm(t)]
    out: list[Word] = []
    k = 0
    for token in tokens:
        target = _norm(token)
        acc, start, end = "", None, None
        while k < len(items) and len(acc) < len(target):
            piece = _norm(items[k].text)
            if not piece:
                k += 1
                continue
            acc += piece
            start = items[k].start if start is None else start
            end = items[k].end
            k += 1
        if acc != target or start is None:
            return items
        out.append(Word(token, start, end))
    return out


def is_degenerate_text(text: str | None) -> bool:
    """An ASR that heard nothing returns '' or a stray '!'; treat under 3 real characters as empty."""
    return sum(ch.isalnum() for ch in (text or "")) < 3


class AsrService:
    vram_gb = 6.0

    def __init__(self):
        self._asr = None

    @property
    def is_loaded(self) -> bool:
        return self._asr is not None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        from .compat import apply_all, download

        apply_all()

        if not torch.cuda.is_available():
            raise RuntimeError("A transcrição automática precisa de uma GPU CUDA.")
        self._asr = Qwen3ASRModel.from_pretrained(
            download(ASR_REPO),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            forced_aligner=download(ALIGNER_REPO),
            forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
            max_inference_batch_size=1,
            max_new_tokens=1024,
        )

    def unload(self) -> None:
        import torch

        self._asr = None
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _language(language: str | None) -> str | None:
        return language if language in ASR_LANGUAGES else None

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcription:
        result = self._asr.transcribe(audio=audio_path, language=self._language(language), return_time_stamps=True)[0]
        text = re.sub(r"\s+", " ", result.text or "").strip()
        items = [Word(it.text, float(it.start_time), float(it.end_time)) for it in (result.time_stamps or [])]
        return Transcription(text, result.language or "", attach_punctuation(text, items))

    def align(self, audio_path: str, transcript: str, language: str) -> list[Word]:
        lang = self._language(language) or "English"
        result = self._asr.forced_aligner.align(audio=audio_path, text=transcript, language=lang)[0]
        items = [Word(it.text, float(it.start_time), float(it.end_time)) for it in result]
        return attach_punctuation(transcript, items)


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def reference_error(reference: str, hypothesis: str) -> float | None:
    """Word error rate of what was heard against the transcript the user typed."""
    ref = [_norm(t) for t in reference.split() if _norm(t)]
    hyp = [_norm(t) for t in hypothesis.split() if _norm(t)]
    if not ref or not hyp:
        return None
    return _levenshtein(ref, hyp) / len(ref)


def is_transcript_mismatch(reference: str, hypothesis: str) -> bool:
    """True only when the audio plainly says something else than the typed transcript.

    When the transcript covers more than the clip (a trimmed recording), every edit is a
    deletion: the error rate stays at or under 1.0 and the heard text is much shorter. So a
    rate above 1.0 always means different speech, and a rate above 0.75 does too when the
    ASR heard at least half as many words. Transcripts under five words are not judged."""
    ref = [t for t in reference.split() if _norm(t)]
    hyp = [t for t in hypothesis.split() if _norm(t)]
    if len(ref) < 5:
        return False
    err = reference_error(reference, hypothesis)
    if err is None:
        return False
    return err > 1.0 or (err > 0.75 and len(hyp) >= len(ref) / 2)
