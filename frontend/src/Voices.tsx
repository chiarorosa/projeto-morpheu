import { useEffect, useRef, useState } from "react";
import { api, type Voice } from "./api";
import { LANG_PT, VOICE_LANGUAGES } from "./langs";
import { RECORDING_TIPS, scriptsFor } from "./scripts";
import { toWav } from "./wav";

const SOURCE_LABEL = { upload: "Arquivo", recording: "Gravação", design: "Desenhada" } as const;

export default function Voices({ voices, onChanged }: { voices: Voice[]; onChanged: () => void }) {
  const [asr, setAsr] = useState(false);
  const [analyzing, setAnalyzing] = useState<Record<string, boolean>>({});
  const [analysisError, setAnalysisError] = useState<Record<string, string>>({});

  useEffect(() => { api.system().then((s) => setAsr(s.asr)).catch(() => setAsr(false)); }, []);

  const analyze = async (id: string) => {
    setAnalyzing((a) => ({ ...a, [id]: true }));
    setAnalysisError((e) => ({ ...e, [id]: "" }));
    try {
      await api.analyzeVoice(id);
    } catch (e) {
      setAnalysisError((err) => ({ ...err, [id]: (e as Error).message }));
    } finally {
      setAnalyzing((a) => ({ ...a, [id]: false }));
      onChanged();
    }
  };

  return (
    <div className="studio">
      <NewVoice onCreated={(v) => { onChanged(); if (asr) void analyze(v.id); }} />
      {voices.length === 0 ? (
        <p className="muted">Nenhuma voz ainda. Envie ou grave de 10 a 20 segundos de fala limpa.</p>
      ) : (
        <div className="voice-grid">
          {voices.map((v) => (
            <VoiceCard key={v.id} voice={v} onChanged={onChanged} asr={asr}
              analyzing={!!analyzing[v.id]} analysisError={analysisError[v.id]} onAnalyze={() => analyze(v.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

function NewVoice({ onCreated }: { onCreated: (v: Voice) => void }) {
  const [name, setName] = useState("");
  const [transcript, setTranscript] = useState("");
  const [language, setLanguage] = useState("Portuguese");
  const [clip, setClip] = useState<{ wav: Blob; seconds: number; url: string; source: "upload" | "recording" } | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);

  useEffect(() => () => { if (clip) URL.revokeObjectURL(clip.url); }, [clip]);

  const accept = async (blob: Blob, source: "upload" | "recording") => {
    setError(null);
    try {
      const { wav, seconds } = await toWav(blob);
      setClip({ wav, seconds, url: URL.createObjectURL(wav), source });
    } catch {
      setError("Não consegui ler esse áudio. Tente WAV, MP3, M4A ou OGG.");
    }
  };

  const toggleRecording = async () => {
    if (recording) {
      recorder.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        void accept(new Blob(chunks, { type: rec.mimeType }), "recording");
      };
      recorder.current = rec;
      rec.start();
      setRecording(true);
    } catch {
      setError("Permissão de microfone negada.");
    }
  };

  const save = async () => {
    if (!clip) return;
    setBusy(true);
    setError(null);
    try {
      const voice = await api.createVoice(name, transcript, clip.wav, clip.source, language);
      setName("");
      setTranscript("");
      setClip(null);
      onCreated(voice);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <strong>Nova voz</strong>
      <RecordingScriptCard language={language} transcript={transcript} onUse={setTranscript} />
      <div className="row wrap">
        <label className="btn ghost file">
          Enviar áudio
          <input type="file" accept="audio/*" onChange={(e) => e.target.files?.[0] && accept(e.target.files[0], "upload")} hidden />
        </label>
        <button className={`btn ${recording ? "danger" : "ghost"}`} onClick={toggleRecording}>
          {recording ? "■ Parar gravação" : "● Gravar"}
        </button>
        {clip && <span className="muted small">{clip.seconds.toFixed(1)} s</span>}
      </div>
      {clip && <audio controls src={clip.url} className="player" />}
      {clip && clip.seconds < 5 && <p className="warn small">Referências com menos de 5 s clonam mal na maioria dos modelos.</p>}
      <p className="muted small">
        Pode ser longa: para cada modelo o Morpheu usa o trecho mais fluente que começa e termina em fim de frase
        (até 10–15 s, conforme o modelo).
      </p>
      <div className="row wrap">
        <label className="field grow">
          <span>Nome</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ex.: Narradora" />
        </label>
        <label className="field">
          <span>Idioma falado</span>
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {VOICE_LANGUAGES.map((l) => <option key={l} value={l}>{LANG_PT[l] ?? l}</option>)}
          </select>
        </label>
      </div>
      <label className="field">
        <span>Transcrição do áudio <em className="muted">(opcional: se ficar em branco, é feita automaticamente)</em></span>
        <textarea rows={2} value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="Exatamente o que é falado no áudio de referência, com a pontuação" />
      </label>
      <div className="row end">
        <button className="btn primary" onClick={save} disabled={!clip || !name.trim() || busy}>{busy ? "Salvando…" : "Salvar voz"}</button>
      </div>
      {error && <div className="alert">{error}</div>}
    </section>
  );
}

function RecordingScriptCard({ language, transcript, onUse }: {
  language: string;
  transcript: string;
  onUse: (text: string) => void;
}) {
  const scripts = scriptsFor(language);
  const [index, setIndex] = useState(0);
  const script = scripts[Math.min(index, scripts.length - 1)];
  const inUse = transcript.trim() === script.text;

  return (
    <div className="script-card">
      <div className="row between wrap">
        <span className="small"><strong>Texto para gravar</strong> <span className="muted">· {script.hint}</span></span>
        {scripts.length > 1 && (
          <div className="row">
            {scripts.map((s, i) => (
              <button key={s.id} className={`chip ${s.id === script.id ? "active" : ""}`} onClick={() => setIndex(i)}>
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <p className="script-text">{script.text}</p>
      <div className="row between wrap">
        <ul className="hints small muted">
          {RECORDING_TIPS.map((t) => <li key={t}>{t}</li>)}
        </ul>
        <button className="btn small" onClick={() => onUse(script.text)} disabled={inUse}>
          {inUse ? "Transcrição preenchida ✓" : "Usar como transcrição"}
        </button>
      </div>
    </div>
  );
}

function VoiceCard({ voice, onChanged, asr, analyzing, analysisError, onAnalyze }: {
  voice: Voice;
  onChanged: () => void;
  asr: boolean;
  analyzing: boolean;
  analysisError?: string;
  onAnalyze: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(voice.name);
  const [transcript, setTranscript] = useState(voice.transcript);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    try {
      await api.updateVoice(voice.id, { name, transcript });
      setEditing(false);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const remove = async () => {
    // Inline two-step confirm instead of window.confirm (blocked in some webviews).
    if (!editing) return setEditing(true);
    await api.deleteVoice(voice.id);
    onChanged();
  };

  return (
    <article className="card voice">
      <div className="row between">
        {editing ? <input value={name} onChange={(e) => setName(e.target.value)} /> : <strong>{voice.name}</strong>}
        <span className="tag">{SOURCE_LABEL[voice.source]}</span>
      </div>
      <audio controls preload="none" src={api.voiceAudioUrl(voice.id)} className="player" />
      {editing ? (
        <textarea rows={2} value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="Transcrição" />
      ) : (
        <p className="muted small clamp">{voice.transcript || "Sem transcrição"}</p>
      )}
      {voice.design_prompt && <p className="muted small clamp">“{voice.design_prompt}”</p>}
      <div className="row wrap">
        <span className="tag subtle">{LANG_PT[voice.language] ?? voice.language}</span>
        {voice.transcript_source === "auto" && <span className="tag subtle">transcrição automática</span>}
        {voice.has_timings && <span className="tag subtle" title="Tempos de cada palavra, usados para recortar a referência entre frases">palavras alinhadas</span>}
      </div>
      {voice.transcript_mismatch && (
        <div className="warn small">
          A transcrição não bate com o áudio. Foi ouvido: “{voice.heard_text}”.{" "}
          <button className="link" onClick={async () => {
            await api.updateVoice(voice.id, { transcript: voice.heard_text });
            onChanged();
            if (asr) onAnalyze();
          }}>Usar o que foi ouvido</button>
        </div>
      )}
      {analyzing && <p className="muted small">Analisando a referência (transcrição e alinhamento)…</p>}
      {analysisError && <div className="alert small">{analysisError}</div>}
      <div className="row between">
        <div className="row">
          <span className="muted small">{voice.duration.toFixed(1)} s</span>
          {asr && !editing && !analyzing && voice.source !== "design" && (!voice.has_timings || !voice.transcript) && (
            <button className="btn small ghost" onClick={onAnalyze}>
              {voice.transcript ? "Alinhar palavras" : "Transcrever automaticamente"}
            </button>
          )}
        </div>
        <div className="row">
          {editing ? (
            <>
              <button className="btn small" onClick={save}>Salvar</button>
              <button className="btn small danger" onClick={remove}>Excluir</button>
              <button className="btn small ghost" onClick={() => setEditing(false)}>Cancelar</button>
            </>
          ) : (
            <button className="btn small ghost" onClick={() => setEditing(true)}>Editar</button>
          )}
        </div>
      </div>
      {error && <div className="alert">{error}</div>}
    </article>
  );
}
