import { useEffect, useRef, useState } from "react";
import { api, type Voice } from "./api";
import { toWav } from "./wav";

const SOURCE_LABEL = { upload: "Arquivo", recording: "Gravação", design: "Desenhada" } as const;

export default function Voices({ voices, onChanged }: { voices: Voice[]; onChanged: () => void }) {
  return (
    <div className="studio">
      <NewVoice onCreated={onChanged} />
      {voices.length === 0 ? (
        <p className="muted">Nenhuma voz ainda. Envie um áudio de 3 a 15 segundos de fala limpa, ou grave pelo microfone.</p>
      ) : (
        <div className="voice-grid">
          {voices.map((v) => <VoiceCard key={v.id} voice={v} onChanged={onChanged} />)}
        </div>
      )}
    </div>
  );
}

function NewVoice({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [transcript, setTranscript] = useState("");
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
      await api.createVoice(name, transcript, clip.wav, clip.source);
      setName("");
      setTranscript("");
      setClip(null);
      onCreated();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <strong>Nova voz</strong>
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
      {clip && clip.seconds < 3 && <p className="warn small">Referências com menos de 3 s costumam clonar mal.</p>}
      <label className="field">
        <span>Nome</span>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ex.: Narradora" />
      </label>
      <label className="field">
        <span>Transcrição do áudio <em className="muted">(recomendado: melhora muito a clonagem)</em></span>
        <textarea rows={2} value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="Exatamente o que é falado no áudio de referência" />
      </label>
      <div className="row end">
        <button className="btn primary" onClick={save} disabled={!clip || !name.trim() || busy}>{busy ? "Salvando…" : "Salvar voz"}</button>
      </div>
      {error && <div className="alert">{error}</div>}
    </section>
  );
}

function VoiceCard({ voice, onChanged }: { voice: Voice; onChanged: () => void }) {
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
      <div className="row between">
        <span className="muted small">{voice.duration.toFixed(1)} s</span>
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
