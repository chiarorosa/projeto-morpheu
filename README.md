# Morpheu

Estúdio local de voz: **clonar** uma voz a partir de alguns segundos de áudio, **desenhar** uma voz nova por descrição e gerar fala com **vozes prontas**. Tudo roda na sua GPU, sem nuvem.

## Instalação

Requisitos: Windows, Python 3.12, GPU NVIDIA com 8 GB de VRAM ou mais, Node.js 20+ (só para compilar a interface).

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -e "backend[dev]"
cd frontend && npm ci && npm run build
```

## Uso

```
morpheu.cmd              janela desktop (pywebview)
morpheu.cmd --browser    abre no navegador (http://127.0.0.1:8765)
morpheu.cmd --server     só a API
```

Fluxo típico:

1. **Biblioteca** → envie ou grave de 3 a 15 s de fala limpa e cole a transcrição exata.
2. **Clonar** → carregue o modelo, escolha a voz, digite o texto e clique em **Gerar fala**.
3. **Desenhar** → descreva a voz, gere e use **Salvar na biblioteca**; ela passa a valer em Clonar.

Na primeira vez que cada modelo é carregado, ele é baixado do Hugging Face para `models/`.

## Modelos (Qwen3-TTS, Apache-2.0)

| ID | Modo | VRAM aprox. |
|---|---|---|
| `qwen3-clone-0.6b` | Clonar (rápido) | ~3 GB |
| `qwen3-clone-1.7b` | Clonar (qualidade) | ~6 GB |
| `qwen3-design` | Desenhar por descrição | ~6 GB |
| `qwen3-voices` | Vozes prontas + instruções de tom | ~6 GB |

Só um modelo fica na GPU por vez; ao trocar, o anterior é descarregado.

## Estrutura

```
backend/morpheu/
  app.py            API FastAPI (/api/v1/...) + OpenAI-compatível (/v1/audio/speech) + UI
  studio.py         pipeline: valida → resolve voz → gera → pós-processa → grava histórico
  manager.py        um modelo residente por vez, carregamento em segundo plano, lock de GPU
  engines/base.py   contrato de motor (ModelInfo, ParamSpec, GenerationRequest)
  engines/qwen3.py  motores Qwen3-TTS (cache das features de referência por voz)
  engines/fake.py   motor de teste sem GPU (MORPHEU_FAKE_ENGINES=1)
  voices.py         biblioteca em data/voices/<id>/
  history.py        histórico em SQLite + WAVs em data/outputs/
  audio.py          leitura/escrita, corte de silêncio, normalização
frontend/src/       React + TypeScript (Vite)
```

Tudo fica dentro desta pasta: `.venv/` (Python), `models/` (pesos), `data/` (vozes e histórico).

## Desenvolvimento

```
# testes do backend
cd backend && ..\.venv\Scripts\python.exe -m pytest -q

# frontend com recarga automática (backend em outro terminal: morpheu.cmd --server)
set PATH=C:\dev\rea\.tools\node;%PATH%
cd frontend && npm run dev        # http://localhost:5173
npm run build                     # gera frontend/dist, servido pelo backend
```

### API OpenAI-compatível

```
curl http://127.0.0.1:8765/v1/audio/speech -H "content-type: application/json" ^
  -d "{\"model\":\"qwen3-clone-0.6b\",\"input\":\"Olá!\",\"voice\":\"Nome da voz na biblioteca\"}" -o fala.wav
```

A API escuta só em 127.0.0.1 e não tem autenticação.

## Próximas fases

- Mais motores: Chatterbox (leve, tags paralinguísticas), VoxCPM (48 kHz)
- Streaming da geração e textos longos divididos em trechos
- STT (ditado e transcrição automática da referência)
- Audiobook: importar PDF/DOCX/EPUB, personagens e uma voz por personagem
- Dublagem: transcrever, traduzir, revozear e legendar
- Servidor MCP
