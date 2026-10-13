"""Turn a saved voice into the reference a given model clones from.

A clip without a transcript is transcribed (and the voice keeps that transcript), a clip
longer than the model's limit is cut to its most fluent sentence-aligned stretch with the
transcript sliced to match, and when no sentence boundary fits it is cut at the last pause
and the cut is transcribed on its own.
"""

from dataclasses import dataclass

from . import audio, refwindow
from .asr import is_degenerate_text, is_transcript_mismatch
from .engines.base import ModelInfo
from .refwindow import Window, Word
from .voices import Voice, VoiceStore

# Clips this close to the limit are used whole: a cut would only shave a breath.
_LIMIT_SLACK_SEC = 0.5


class ReferenceError(ValueError):
    pass


@dataclass
class PreparedReference:
    path: str
    text: str | None
    key: str
    note: str  # human-readable summary for the history/UI


class ReferencePreparer:
    def __init__(self, voices: VoiceStore, manager, asr=None):
        self.voices = voices
        self.manager = manager
        self.asr = asr

    # ---------------- speech recognition ----------------
    def _with_asr(self, fn, progress):
        if self.asr is None:
            raise ReferenceError("A transcrição automática não está disponível neste modo.")
        return self.manager.run_aux(self.asr, fn, progress)

    def analyze(self, voice_id: str, language: str | None = None, progress=lambda m: None) -> Voice:
        """Transcribe the clip, time its words, and flag a typed transcript the audio contradicts."""
        voice = self.voices.get(voice_id)
        path = str(self.voices.audio_path(voice_id))
        lang = language or voice.language

        def work(asr):
            heard = asr.transcribe(path, lang)
            if voice.transcript_source == "manual" and voice.transcript:
                mismatch = is_transcript_mismatch(voice.transcript, heard.text)
                words = [] if mismatch else asr.align(path, voice.transcript, heard.language or lang)
                return dict(words=[w.to_dict() for w in words], mismatch=mismatch, heard_text=heard.text,
                            detected=heard.language)
            if is_degenerate_text(heard.text):
                raise ReferenceError("Não foi possível reconhecer fala nessa referência.")
            return dict(transcript=heard.text, words=[w.to_dict() for w in heard.words],
                        mismatch=False, heard_text=heard.text, detected=heard.language)

        result = self._with_asr(work, progress)
        detected = result.pop("detected", "") or ""
        # The ASR reports e.g. "Portuguese" (or "Portuguese,English" for mixed speech).
        spoken = lang if lang not in ("", "Auto", None) else detected.split(",")[0].strip()
        if voice.language in ("", "Auto") and spoken:
            self.voices.update(voice_id, language=spoken)
        return self.voices.set_analysis(voice_id, **result)

    # ---------------- per-model reference ----------------
    def prepare(self, voice: Voice, info: ModelInfo, language: str, progress=lambda m: None) -> PreparedReference:
        path = self.voices.audio_path(voice.id)
        if voice.duration < info.ref_min_sec:
            raise ReferenceError(
                f"{info.name} precisa de pelo menos {info.ref_min_sec:g} s de referência; esta voz tem {voice.duration:.1f} s.")

        # A model that reads the transcript needs one; transcribe and keep it on the voice.
        if info.uses_ref_text and not voice.transcript and self.asr is not None:
            progress("Transcrevendo a referência…")
            voice = self.analyze(voice.id, language if language != "Auto" else None, progress)

        text = (voice.transcript or None) if info.uses_ref_text else None
        if voice.duration <= info.ref_max_sec + _LIMIT_SLACK_SEC:
            return PreparedReference(str(path), text, f"{voice.id}:full", f"referência inteira ({voice.duration:.1f} s)")

        window = self._window(voice, info, language, progress)
        return self._materialize(voice, window, info, language, progress)

    def _window(self, voice: Voice, info: ModelInfo, language: str, progress) -> Window:
        words: list[Word] = []
        if voice.transcript and not voice.words and not voice.transcript_mismatch and self.asr is not None:
            progress("Alinhando a transcrição com o áudio…")
            voice = self.analyze(voice.id, language if language != "Auto" else None, progress)
        if voice.words:
            words = refwindow.clean_words([Word.from_dict(w) for w in voice.words], voice.duration)
        if words:
            # A person who typed the transcript heard the whole clip, so its final full stop is real.
            end_pause = 0.0 if voice.transcript_source == "manual" else refwindow.END_PAUSE_SEC
            window = (refwindow.best_sentence_window(words, info.ref_max_sec, voice.duration, end_pause=end_pause)
                      or refwindow.word_gap_window(words, info.ref_max_sec, voice.duration))
            if window:
                return window
        data, sr = audio.read_audio(self.voices.audio_path(voice.id).read_bytes())
        cut = refwindow.last_pause_cut(data, sr, info.ref_max_sec)
        return Window(0.0, cut, "", "cut")

    def _materialize(self, voice: Voice, window: Window, info: ModelInfo, language: str,
                     progress) -> PreparedReference:
        d = self.voices.windows_dir(voice.id)
        stem = f"{window.start:.2f}-{window.end:.2f}"
        wav, txt = d / f"{stem}.wav", d / f"{stem}.txt"
        if not wav.exists():
            data, sr = audio.read_audio(self.voices.audio_path(voice.id).read_bytes())
            audio.write_wav(wav, data[int(window.start * sr): int(window.end * sr)], sr)
        text: str | None = window.text or None
        if info.uses_ref_text and not text:
            # Cut without timings: the whole-clip transcript would describe speech the model
            # never hears (and it would say the missing tail first), so transcribe the cut itself.
            if txt.exists():
                text = txt.read_text("utf-8") or None
            elif self.asr is not None:
                progress("Transcrevendo o trecho recortado…")
                heard = self._with_asr(lambda asr: asr.transcribe(str(wav), language), progress)
                text = None if is_degenerate_text(heard.text) else heard.text
                txt.write_text(text or "", "utf-8")
        if not info.uses_ref_text:
            text = None
        how = {"sentence": "entre frases", "pause": "numa pausa", "cut": "na última pausa"}[window.reason]
        note = f"trecho {window.start:.1f}–{window.end:.1f} s ({window.duration:.1f} s, cortado {how})"
        return PreparedReference(str(wav), text, f"{voice.id}:{stem}", note)
