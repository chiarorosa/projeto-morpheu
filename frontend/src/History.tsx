import { useEffect, useState } from "react";
import { api, type Generation } from "./api";

const MODE_LABEL = { clone: "Clonada", design: "Desenhada", tts: "Voz pronta" } as const;

export default function History({ onVoicesChanged }: { onVoicesChanged: () => void }) {
  const [items, setItems] = useState<Generation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => api.history().then(setItems).catch((e) => setError(e.message));
  useEffect(() => void load(), []);

  if (error) return <div className="alert">{error}</div>;
  if (!items) return <p className="muted">Carregando…</p>;
  if (!items.length) return <p className="muted">Nada gerado ainda.</p>;

  return (
    <div className="history">
      {items.map((it) => <Row key={it.id} item={it} onDeleted={load} onVoicesChanged={onVoicesChanged} />)}
    </div>
  );
}

function Row({ item, onDeleted, onVoicesChanged }: { item: Generation; onDeleted: () => void; onVoicesChanged: () => void }) {
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const save = async () => {
    try {
      await api.voiceFromHistory(item.id, name);
      setSaving(false);
      setMsg("Salva na biblioteca.");
      onVoicesChanged();
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  return (
    <article className="card hist">
      <div className="row between">
        <span className="tag">{MODE_LABEL[item.mode]}</span>
        <span className="muted small">
          {new Date(item.created_at * 1000).toLocaleString("pt-BR")} · {item.model_id} · {item.duration.toFixed(1)} s
        </span>
      </div>
      <p className="hist-text">{item.text}</p>
      {item.instruct && <p className="muted small">“{item.instruct}”</p>}
      <audio controls preload="none" src={api.historyAudioUrl(item.id)} className="player" />
      <div className="row between wrap">
        <div className="row">
          <a className="btn small ghost" href={api.historyAudioUrl(item.id, true)}>Baixar</a>
          {saving ? (
            <>
              <input placeholder="Nome da voz" value={name} onChange={(e) => setName(e.target.value)} />
              <button className="btn small" onClick={save} disabled={!name.trim()}>Salvar</button>
            </>
          ) : (
            <button className="btn small ghost" onClick={() => setSaving(true)}>Usar como voz</button>
          )}
        </div>
        <button className="btn small ghost danger-text" onClick={() => api.deleteHistory(item.id).then(onDeleted)}>Excluir</button>
      </div>
      {msg && <p className="small muted">{msg}</p>}
    </article>
  );
}
