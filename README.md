# Morpheu

Estúdio local de voz: **clonar** uma voz a partir de alguns segundos de áudio, **desenhar** uma voz nova por descrição e gerar fala com **vozes prontas**. Tudo roda na sua GPU, sem nuvem.

## Instalação

Requisitos: Windows, Python 3.12, GPU NVIDIA com 8 GB de VRAM ou mais, Node.js 20+ (só para compilar a interface).

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install --no-deps -r backend\requirements.lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e backend
cd frontend && npm ci && npm run build
```

As bibliotecas dos motores fixam versões incompatíveis entre si (torch 2.6/2.8, transformers 4.46/5.3), por isso
o ambiente é instalado sem resolver dependências, a partir de `requirements.lock.txt`, o conjunto testado.

## Uso

```
morpheu.cmd              janela desktop (pywebview)
morpheu.cmd --browser    abre no navegador (http://127.0.0.1:8765)
morpheu.cmd --server     só a API
```

Fluxo típico:

1. **Biblioteca** → envie ou grave de 10 a 30 s de fala limpa, escolha o idioma e, se quiser, cole a transcrição
   (em branco, ela é feita automaticamente). A referência é analisada logo depois de salva.
2. **Clonar** → escolha um modelo (os marcados "bom em português" são os indicados para pt-BR), carregue-o,
   escolha a voz, digite o texto e clique em **Gerar fala**.
3. **Desenhar** → descreva a voz, gere e use **Salvar na biblioteca**; ela passa a valer em Clonar.

Na primeira vez que cada modelo é carregado, ele é baixado do Hugging Face para `models/`.

## Modelos

| ID | Modo | Referência usada | VRAM | Licença |
|---|---|---|---|---|
| `omnivoice` | Clonar · muito rápido, 600+ idiomas | 3–10 s | ~2 GB | Apache-2.0 |
| `chatterbox-multilingual` | Clonar · expressividade ajustável, só o áudio | 3–10 s | ~4 GB | MIT |
| `voxcpm2` | Clonar · 48 kHz, maior fidelidade | 3–15 s | ~6 GB | Apache-2.0 |
| `qwen3-clone-0.6b` / `-1.7b` | Clonar | 1,5–10 s | ~3 / ~6 GB | Apache-2.0 |
| `qwen3-design` | Desenhar por descrição | — | ~6 GB | Apache-2.0 |
| `qwen3-voices` | Vozes prontas + instruções de tom | — | ~6 GB | Apache-2.0 |

Medido numa RTX 3060 Ti com uma referência real em pt-BR e a mesma frase de 225 caracteres:

| Modelo | Tempo | Erro de palavras* | Semelhança de voz** |
|---|---|---|---|
| OmniVoice | 5 s | 0,04 | 0,959 |
| Chatterbox | 21 s | 0,06 | 0,959 |
| VoxCPM2 | 30 s | 0,04 | 0,960 |
| Qwen3 1.7B | 56 s | 0,06 | 0,960 |

\* Qwen3-ASR transcrevendo a saída, comparado ao texto. \*\* Cosseno entre embeddings de locutor (codificador do Chatterbox).
Os números medem pronúncia e timbre; sotaque e naturalidade só se avaliam ouvindo.

Só um modelo de voz fica na GPU por vez; ao trocar, o anterior é descarregado. O reconhecimento de fala
(Qwen3-ASR 1.7B + Qwen3-ForcedAligner 0.6B) é carregado só durante a análise de uma referência; se faltar
memória, o modelo de voz ativo é descarregado e recarregado em seguida.

## Como a clonagem funciona

1. **Texto** — tags como `[laugh]` são removidas; números, valores (R$, US$, €), porcentagens, horas, ordinais,
   unidades e abreviações são escritos por extenso no idioma, com concordância de gênero em português
   ("2 dúzias" → "duas dúzias"). Desligável em "Ler números por extenso".
2. **Referência** — sem transcrição, ela é feita automaticamente e guardada na voz. Cada palavra recebe seu
   tempo (alinhamento forçado). Quando a gravação passa do limite do modelo, é usado o trecho **fluente** mais
   longo que começa e termina em fim de frase (pausas ≤ 20% do trecho, nenhuma acima de 0,6 s), com a
   transcrição cortada no mesmo ponto; sem tempos, corta na última pausa e transcreve só o trecho.
   Uma transcrição digitada que não bate com o áudio é sinalizada na Biblioteca.
3. **Geração** — textos longos são divididos em trechos de frases inteiras, no tamanho que cada modelo aguenta.
   Saída "em loop" (longa demais para o texto), com silêncio colapsado ou vazia é descartada e o trecho é
   refeito com outra semente (até 3 tentativas).
4. **Acabamento** — os trechos são unidos com pausas curtas, o silêncio das pontas é cortado e a loudness é
   nivelada em −18 LUFS (pico ≤ −1 dBFS).

No Histórico, cada geração mostra o trecho da referência usado, o texto efetivamente falado e se algum
trecho foi refeito.

## Emoções na voz clonada

Com o **Qwen3 Clone 1.7B**, a aba Clonar oferece 13 emoções (alegria, tristeza, raiva, medo, surpresa, nojo,
admiração, nostalgia, desprezo, desaprovação, remorso, indignação, desespero) em três intensidades. É o mesmo
mecanismo de *activation steering*: cada emoção é uma direção no fluxo residual do
"talker" do modelo; durante a geração ela é somada após as camadas 21–25, o estado é reescalado para manter
a norma (muda a direção, não a intensidade do sinal) e o peso decai a cada quadro (×0,985). O prompt nunca é
alterado, por isso a voz continua sendo a da referência. Depois, uma correção segura o volume do fim da fala
para ele não cair mais que 1,5 dB (3 dB em tristeza e desespero).

Medido com a mesma frase e semente (referência real em pt-BR): pronúncia sem erros em todas; semelhança de
voz 0,89–0,91 contra 0,90 da versão neutra; tom mediano 100 Hz neutro → 114 alegria, 88 tristeza (e mais
monótona e lenta), 146 raiva, 118 surpresa; medo deixa a fala mais lenta.

Os vetores (`backend/morpheu/vendor/emotion_steer/*.qlsteer`) são do motor
[qwen3-tts em C de Gabriele Mastrapasqua](https://github.com/gabriele-mastrapasqua/qwen3-tts) (licença MIT,
commit `5b43adf`). Funcionam só no modelo 1.7B (dimensão 2048).

## Estrutura

```
backend/morpheu/
  app.py            API FastAPI (/api/v1/...) + OpenAI-compatível (/v1/audio/speech) + UI
  studio.py         pipeline: valida → resolve voz → gera → pós-processa → grava histórico
  manager.py        um modelo residente por vez, carregamento em segundo plano, lock de GPU
  engines/base.py   contrato de motor (ModelInfo, ParamSpec, GenerationRequest)
  engines/          omnivoice, chatterbox, voxcpm, qwen3 (+ fake.py: motor de teste sem GPU)
  reference.py      transcrição automática, alinhamento e recorte da referência por modelo
  refwindow.py      escolha do trecho fluente entre fins de frase
  asr.py            Qwen3-ASR + ForcedAligner, checagem de transcrição divergente
  textnorm.py       números, valores, horas e abreviações por extenso
  chunker.py        divisão de textos longos em frases
  quality.py        detecção de saída defeituosa, loudness, junção de trechos
  compat.py         torchaudio via soundfile; download do HF sem a corrida de symlinks no Windows
  steer.py          emoções por steering do talker do Qwen3 1.7B
  vendor/           vetores de emoção (MIT) e tokenizer Higgs do transformers 5 (Apache-2.0)
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

- Desenhar voz também com OmniVoice e VoxCPM2
- Streaming da geração
- STT (ditado e transcrição de arquivos)
- Audiobook: importar PDF/DOCX/EPUB, personagens e uma voz por personagem
- Dublagem: transcrever, traduzir, revozear e legendar
- Servidor MCP
