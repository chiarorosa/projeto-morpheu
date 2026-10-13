# Morpheu Roadmap

Where Morpheu is going, in phases. Each item says **what** it is, **how** we plan to build it, and **when
it is done** — so anyone can pick one up. Sizes are rough: **S** (a day or two), **M** (a week), **L**
(several weeks), **XL** (a project of its own). Items marked 🟢 are good first contributions.

Want to work on something? Open an issue referencing the item (e.g. "Roadmap 3.4") so we can agree on the
approach before you start. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Principles

- **Local-first.** Everything runs on the user's machine; no account, no telemetry. Network access only to
  download models (or to an LLM endpoint the user configures).
- **Redistributable licenses only.** Code and weights under MIT, Apache-2.0, BSD, CC-BY or similar. Models
  with non-commercial or custom restrictive licenses are optional add-ons at most, never defaults.
- **Consent by design.** Features that clone or convert voices keep the responsible-use notice visible.
- **GPU-free tests.** Every pipeline gets fake engines/services so CI can test it without a GPU.
- **Honest results.** PRs state what was tested on real hardware and what was not.

## Done (v0.1)

- Voice cloning with OmniVoice, Chatterbox Multilingual, VoxCPM2 and Qwen3-TTS (0.6B/1.7B)
- Voice design (Qwen3-TTS VoiceDesign) and built-in voices (Qwen3-TTS CustomVoice)
- Automatic transcription and forced alignment of references (Qwen3-ASR + ForcedAligner)
- Sentence-aligned, fluency-ranked reference windows per model; transcript mismatch detection
- Text normalization (numbers, currency, times, ordinals, units, abbreviations; pt-BR gender agreement)
- Sentence chunking for long text; runaway/collapse detection with automatic retry; loudness normalization
- 13 emotions × 3 intensities on Qwen3-TTS 1.7B via activation steering
- Voice library, history, recording scripts, OpenAI-compatible `/v1/audio/speech`

---

## Phase 0 — Foundations

Things every later phase depends on.

| # | Item | Size |
|---|---|---|
| 0.1 | 🟢 **UI internationalization** — extract strings to locale files (react-i18next or a small in-house loader), ship `pt-BR` and `en`, language picker in settings. Done when every visible string comes from a locale file and switching language needs no reload. | M |
| 0.2 | **Job system** — one background job model for long tasks (generation batches, transcription, dubbing, exports): queued/running/done/failed/cancelled, progress %, ETA, cancel, persisted across restarts, polled or pushed over SSE. Done when two features use it and cancelling frees the GPU. | M |
| 0.3 | 🟢 **Output formats** — WAV, MP3, FLAC, OGG/Opus for downloads and the API (`response_format`), via the bundled ffmpeg or `soundfile`. | S |
| 0.4 | **Settings page** — output format, default model per mode, low-VRAM mode (unload after each job, smaller batch), Hugging Face mirror/endpoint, data and model folders, UI language. | M |
| 0.5 | **Model manager** — list downloaded models with disk usage, download with progress before first use, verify, delete. | M |
| 0.6 | **Hardware backends** — first-run detection of NVIDIA (CUDA), AMD (ROCm), Intel Arc (XPU) or CPU; install the matching PyTorch build; VRAM budget per card. | L |
| 0.7 | **Linux support** — `morpheu.sh` launcher, CI job on Ubuntu, documented install. macOS (MLX) later. | M |
| 0.8 | **Packaging** — a portable Windows build (embedded Python + prebuilt UI) and an installer; first-run wizard for backend and model downloads. | L |

## Phase 1 — Voice studio, complete

| # | Item | Size |
|---|---|---|
| 1.1 | **Streaming generation** — sentence-level streaming of PCM chunks over SSE/WebSocket so playback starts in < 2 s; crossfade between chunks; streaming variant of `/v1/audio/speech`. | M |
| 1.2 | **Batch generation** — generate many lines with one voice in one batched model call where the engine supports it (Qwen3-TTS, Kokoro), falling back to sequential. Basis for audiobooks and dubbing. | M |
| 1.3 | **Speed control** — per-generation speed for engines that support it natively; pitch-preserving time-stretch (WSOLA, e.g. ffmpeg `atempo`) for the rest. | S |
| 1.4 | 🟢 **Expression tags** — per-engine list of supported paralinguistic tags (`[laugh]`, `[sigh]`, `[breath]`…), shown in the editor, kept for engines that render them and stripped for the rest. | S |
| 1.5 | **Pronunciation lexicon** — user dictionary of phrase → respelling (global and per project), applied before generation with word-boundary matching; marks generated audio as stale when an entry changes. | M |
| 1.6 | **Voice design on more engines** — OmniVoice (attribute-based instructions) and VoxCPM2 (description prefix); a design "chip" vocabulary mapped per engine. | M |
| 1.7 | **More engines** — Kokoro-82M (Apache-2.0, fast preset voices, CPU-capable); NeuTTS Air (Apache-2.0, small on-device cloning); evaluate further open models as they appear. Each with honest `ModelInfo` limits and tests. | M each |
| 1.8 | **Emotions, next steps** — per-voice intensity calibration; mixing two emotions (weighted sum of directions); emotion on other engines via instruction-capable models (design/CustomVoice) plus voice conversion back to the target voice; emotion tags inline in text (`[joy] … [sad] …`). | L |
| 1.9 | **Voice conversion (speech-to-speech)** — change the speaker of an existing recording to a library voice, keeping words and timing (e.g. Chatterbox's S3Gen voice-conversion stack, MIT). New "Voice changer" page; also a building block for dubbing. | M |
| 1.10 | **Reference clean-up** — optional vocal isolation (Demucs, MIT) and denoising/enhancement (e.g. DeepFilterNet, MIT/Apache) before saving a reference; waveform trimmer to pick the span by hand. | M |
| 1.11 | 🟢 **Reference from a URL** — download audio from a video URL (yt-dlp), preview, select a span, transcribe, save as a voice — with a consent reminder. | S |
| 1.12 | **Faster Qwen3-TTS** — CUDA-graph decoding (e.g. the open faster-qwen3-tts approach) and persisted voice prompts on disk, to cut latency 2–5×. | L |
| 1.13 | 🟢 **Generation presets** — save/load named parameter sets per engine ("narration", "calm", "energetic"). | S |

## Phase 2 — Speech-to-text

| # | Item | Size |
|---|---|---|
| 2.1 | **Transcription page** — transcribe files or microphone input with Qwen3-ASR (multilingual) or Parakeet (CC-BY-4.0, fast for European languages); word timestamps; export TXT, SRT, VTT, JSON. | M |
| 2.2 | 🟢 **Dictation** — push-to-talk shortcut that transcribes and copies the text to the clipboard. | S |
| 2.3 | **Speaker diarization** — who spoke when (pyannote community model), labels editable in the transcript; prerequisite for dubbing and clips. | M |
| 2.4 | **Robust long-form ASR** — chunked decoding with overlap, language probe on the first seconds, guards against hallucinated or empty output, re-run on a cleaner (vocals-only) stem. | M |

## Phase 3 — Audiobooks

| # | Item | Size |
|---|---|---|
| 3.1 | **Import** — TXT, Markdown, DOCX, EPUB and PDF; OCR for scanned PDFs (Tesseract); other ebook formats through a user-installed Calibre. Text clean-up: mojibake repair, invisible characters, scene-break symbols. | L |
| 3.2 | **Segmentation** — per-language sentence splitting (e.g. pysbd), word-capped segments, chapters/sections, manual split/merge/move/delete. | M |
| 3.3 | **Speaker-marked scripts** — import `NAME: line` scripts directly into a multi-voice project. | S |
| 3.4 | **Character detection** — with a user-configured LLM (OpenAI-compatible endpoint, local or remote): find speakers and quoted dialogue, link the narrator, fold minor characters into shared voices; tag the delivery (emotion) of each line. Strict JSON schemas, retries, coverage checks. | L |
| 3.5 | **Automatic casting** — design a voice for every uncast character from the traits found in 3.4, save it to the library, keep voices distinct. | M |
| 3.6 | **Generation** — per-segment voice spans (several voices in one paragraph), resumable batch with cancel, take history with promote/rename/upload, speaker-change pauses, regenerate a single span. | L |
| 3.7 | **Read-along** — player that highlights each word using forced alignment of the generated audio. | M |
| 3.8 | **Export** — WAV/MP3/M4B with chapter markers, ID3 tags and cover; two-pass loudness normalization with presets (e.g. ACX-style targets); handles books past the 4 GiB WAV limit. | M |

## Phase 4 — Video dubbing

A pipeline of independent, restartable stages, each cached so a re-dub restarts from translation.

| # | Item | Size |
|---|---|---|
| 4.1 | **Ingest & separation** — extract audio from video (ffmpeg), split into vocals and background stems (Demucs). | M |
| 4.2 | **Transcribe & segment** — ASR with word timestamps on the vocals stem, re-flowed into one sentence per segment; diarization merged into speaker turns; optional active-speaker detection on video (e.g. Light-ASD, MIT) to fix overlaps. | L |
| 4.3 | **Translate** — target-language text that fits the time slot: candidate phrasings ranked by estimated spoken duration, glossary of protected/forced terms, plain mode for subtitles only. Engines: an open MT/LLM with a permissive license (e.g. Qwen3 or MADLAD-400) or the user's LLM endpoint. | L |
| 4.4 | **Speaker voices** — pick each speaker's cleanest whole-sentence span as clone reference; let the user swap it from any segment or a trimmed clip. | M |
| 4.5 | **Synthesize to fit** — best-of-N takes per line to land inside the slot, silence trimming and internal-pause compression before any speed-up, keep the original's phrase pauses, runaway/truncation guard. | L |
| 4.6 | **Timing & mix** — place lines with shift/stretch/hybrid fitting (capped pitch-preserving tempo), match the dubbed voice loudness to the original speech, mix over the background stem. | M |
| 4.7 | **Editor** — segment list with source/target text, split/merge/delete/regenerate one line (re-fit and re-mix only what changed), per-speaker settings. | L |
| 4.8 | **Subtitles & export** — SRT/VTT, burned-in styled captions with saved presets, video export with hardware encoders and automatic software fallback. | M |
| 4.9 | **Lip sync (optional)** — re-render mouths on face spans with an open lip-sync model (e.g. LatentSync, Apache-2.0); face detection with permissively licensed models only. | XL |

## Phase 5 — Video tools

| # | Item | Size |
|---|---|---|
| 5.1 | **AI cut (spoken-video editor)** — detect filler words, stutters and abandoned restarts; group repeated takes of the same line and keep the best; compare against a script to find off-script parts; review keep/cut decisions, then render with captions. | XL |
| 5.2 | **Short clips** — find the most engaging 30–90 s moments in a long video (transcript windows snapped to word boundaries, ranked by an LLM), detect scene cuts (TransNetV2, MIT, with ffmpeg fallback), reframe to 9:16 following the active speaker, word-by-word captions with keyword highlights, optional hook title card, batch export. | XL |

## Phase 6 — Integrations

| # | Item | Size |
|---|---|---|
| 6.1 | **MCP server** — tools `speak`, `list_voices`, `generate`, `transcribe` over Streamable HTTP, so AI agents can talk with a library voice; audio can play in the app via SSE. | M |
| 6.2 | **API hardening** — optional bearer token, configurable host/port with a warning outside loopback, `/v1/audio/speech` formats and streaming, OpenAPI docs page. | M |
| 6.3 | **LLM provider settings** — one place to configure an OpenAI-compatible endpoint (Ollama, LM Studio, OpenRouter…) used by audiobooks, dubbing and clips; JSON-schema calls with retries, timeouts and cost estimates when the provider reports them. | M |

## Out of scope

Accounts, cloud execution, billing and quotas, license keys or DRM, watermarking free tiers, telemetry, and
scraping third-party voice catalogs. Morpheu stays a local, open tool.
