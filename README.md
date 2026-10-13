# Morpheu — local AI voice cloning studio

**English** · [Português](README.pt-BR.md)

[![CI](https://github.com/chiarorosa/projeto-morpheu/actions/workflows/ci.yml/badge.svg)](https://github.com/chiarorosa/projeto-morpheu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Morpheu is an open-source, offline **voice cloning and text-to-speech (TTS) studio** that runs on your own
NVIDIA GPU. Clone a voice from a short recording, design a new voice from a text description, add emotions
to a cloned voice, and generate speech in 20+ languages — with first-class **Brazilian Portuguese (pt-BR)**
support. No cloud, no account, no per-character fees.

It wraps state-of-the-art open TTS models — **OmniVoice**, **Chatterbox Multilingual**, **VoxCPM2** and
**Qwen3-TTS** — behind one desktop app (React + FastAPI + pywebview) and an **OpenAI-compatible
`/v1/audio/speech` API**.

## Features

- **Zero-shot voice cloning** from 5–30 s of audio, with four engines to choose from.
- **Automatic reference preparation**: speech recognition (Qwen3-ASR) transcribes the clip, forced alignment
  times every word, and each model gets the most fluent sentence-aligned stretch that fits its limit.
- **Emotional voice cloning**: 13 emotions (joy, sadness, anger, fear, surprise, awe, nostalgia…) at three
  intensities, via activation steering of Qwen3-TTS 1.7B — the voice stays yours.
- **Voice design**: describe a voice ("deep male voice, calm, like a storyteller") and save it to the library.
- **Text normalization** for natural reading of numbers, currency, percentages, times, ordinals and
  abbreviations (with grammatical gender agreement in Portuguese).
- **Long-form TTS**: long text is split at sentence boundaries; looping or collapsed output is detected and
  regenerated automatically; loudness is normalized to −18 LUFS.
- **Recording scripts** built for cloning (phonetically rich, sentence-structured) right in the app.
- **OpenAI-compatible API** on `127.0.0.1`, so OpenAI TTS clients can point at it (WAV output; `voice` is a library voice name).
- Everything stays inside the project folder: Python env, model weights and your voices.

## Requirements

- Windows 10/11 (the only platform tested so far — Linux reports and PRs are welcome)
- Python 3.12
- NVIDIA GPU with 8 GB+ VRAM (tested on an RTX 3060 Ti)
- Node.js 20+ (only to build the UI)
- ~30 GB of disk for all models (each model downloads on first use)

## Installation

```
git clone https://github.com/chiarorosa/projeto-morpheu.git
cd projeto-morpheu
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install --no-deps -r backend\requirements.lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e backend
cd frontend && npm ci && npm run build
```

The engine libraries pin mutually incompatible versions (torch 2.6/2.8, transformers 4.46/5.3), so the
environment is installed without dependency resolution from `requirements.lock.txt`, the tested set.
Small compatibility shims (`backend/morpheu/compat.py`, `backend/morpheu/vendor/`) make them share it.

## Usage

```
morpheu.cmd              desktop window (pywebview)
morpheu.cmd --browser    open in the browser (http://127.0.0.1:8765)
morpheu.cmd --server     API only
```

1. **Library** — read the on-screen recording script aloud (or upload 10–30 s of clean speech), pick the
   language, and save. The clip is transcribed and aligned automatically.
2. **Clone** — pick a model (the ones tagged "good in Portuguese" are best for pt-BR), load it, choose the
   voice, type the text and click **Generate**. With Qwen3 Clone 1.7B you can also pick an emotion.
3. **Design** — describe a voice, generate, then **Save to library** to use it for cloning.

The UI is in Portuguese; English localization is a welcome contribution.

### OpenAI-compatible API

```
curl http://127.0.0.1:8765/v1/audio/speech -H "content-type: application/json" ^
  -d "{\"model\":\"omnivoice\",\"input\":\"Hello!\",\"voice\":\"Voice name in the library\"}" -o speech.wav
```

The API listens on 127.0.0.1 only and has no authentication. The full REST API lives under `/api/v1`
(models, voices, history, generation); see `backend/morpheu/app.py`.

## Models

| ID | Mode | Reference used | VRAM | License |
|---|---|---|---|---|
| `omnivoice` | Clone · very fast, 600+ languages | 3–10 s | ~2 GB | Apache-2.0 |
| `chatterbox-multilingual` | Clone · adjustable expressiveness, audio only | 3–10 s | ~4 GB | MIT |
| `voxcpm2` | Clone · 48 kHz, highest fidelity | 3–15 s | ~6 GB | Apache-2.0 |
| `qwen3-clone-0.6b` / `-1.7b` | Clone (1.7B: + emotions) | 1.5–10 s | ~3 / ~6 GB | Apache-2.0 |
| `qwen3-design` | Voice design from a description | — | ~6 GB | Apache-2.0 |
| `qwen3-voices` | Built-in voices + tone instructions | — | ~6 GB | Apache-2.0 |

Measured on an RTX 3060 Ti with a real pt-BR reference and the same 225-character sentence:

| Model | Time | Word error rate* | Speaker similarity** |
|---|---|---|---|
| OmniVoice | 5 s | 0.04 | 0.959 |
| Chatterbox | 21 s | 0.06 | 0.959 |
| VoxCPM2 | 30 s | 0.04 | 0.960 |
| Qwen3 1.7B | 56 s | 0.06 | 0.960 |

\* Qwen3-ASR transcribing the output vs. the input text. \*\* Cosine similarity of speaker embeddings.
These measure pronunciation and timbre; accent and naturalness can only be judged by listening.

Only one voice model stays on the GPU at a time. Speech recognition (Qwen3-ASR 1.7B + Qwen3-ForcedAligner
0.6B) loads only while analysing a reference, temporarily unloading the active model if VRAM is short.

## How cloning works

1. **Text** — `[laugh]`-style tags are removed; numbers, currency, percentages, times, ordinals, units and
   abbreviations are spelled out for the target language.
2. **Reference** — missing transcripts are generated and saved; words are force-aligned. When the recording
   exceeds the model's limit, the longest *fluent* stretch (pauses ≤ 20%, none over 0.6 s) that starts and
   ends on a sentence boundary is used, with the transcript sliced to match. A typed transcript that
   contradicts the audio is flagged.
3. **Generation** — long text is chunked by sentences; runaway, collapsed or empty chunks are regenerated
   with a new seed (up to 3 tries).
4. **Finishing** — chunks are joined with short pauses, edges trimmed, loudness normalized.

### Emotions

Each emotion is a direction in the residual stream of the Qwen3-TTS 1.7B talker. During generation it is
added after layers 21–25, the hidden state is rescaled to its original norm, and the weight decays every
frame (×0.985); the prompt is never steered, so the speaker identity is preserved. Steering vectors come
from [Gabriele Mastrapasqua's qwen3-tts C engine](https://github.com/gabriele-mastrapasqua/qwen3-tts) (MIT).

## Project structure

```
backend/morpheu/
  app.py            FastAPI app: /api/v1, OpenAI-compatible /v1/audio/speech, static UI
  studio.py         pipeline: validate → prepare reference → generate chunks → finish → history
  manager.py        one resident model at a time, background loading, GPU lock
  engines/          omnivoice, chatterbox, voxcpm, qwen3 (+ fake.py: GPU-free test engine)
  reference.py      auto-transcription, alignment and per-model reference window
  refwindow.py      fluent sentence-aligned window selection
  asr.py            Qwen3-ASR + ForcedAligner, transcript mismatch check
  textnorm.py       spelling out numbers, money, times and abbreviations
  chunker.py        sentence chunking for long text
  quality.py        runaway/collapse detection, loudness, chunk joining
  steer.py          emotion steering for Qwen3-TTS 1.7B
  compat.py         runtime shims (torchaudio via soundfile, HF download race on Windows)
  vendor/           third-party code and data with their own licenses (see NOTICE files)
  voices.py         voice library in data/voices/<id>/
  history.py        SQLite history + WAVs in data/outputs/
frontend/src/       React + TypeScript (Vite)
```

## Development

```
cd backend && ..\.venv\Scripts\python.exe -m pytest -q   # GPU-free tests (fake engines)
cd frontend && npm run dev                                # UI with hot reload; run `morpheu.cmd --server` too
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, and the roadmap below for ideas.

## Roadmap

Streaming, speech-to-text, audiobooks, video dubbing, AI video editing, more engines, MCP and Linux support —
see the phased plan with sizes and first-issue picks in **[ROADMAP.md](ROADMAP.md)**.

## Responsible use

Only clone your own voice or the voice of someone who has given **explicit consent**. Do not use Morpheu to
impersonate, deceive, defraud or defame anyone, or to produce content that violates likeness, voice,
copyright or privacy laws. When you publish synthetic audio, disclose that it was AI-generated. Chatterbox
output carries Resemble AI's inaudible PerTh watermark.

## License

The code is released under the [MIT License](LICENSE). Model weights are downloaded from their authors and
follow their own licenses (table above). Third-party files in `backend/morpheu/vendor/` keep their licenses
(MIT and Apache-2.0; see the `NOTICE` and `LICENSE` files in each folder).
