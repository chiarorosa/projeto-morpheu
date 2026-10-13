import { useCallback, useEffect, useState } from "react";
import { api, type ActiveState, type Mode, type ModelInfo, type Voice } from "./api";
import Studio from "./Studio";
import Voices from "./Voices";
import History from "./History";

type Page = Mode | "voices" | "history";

const NAV: { id: Page; label: string; hint: string }[] = [
  { id: "clone", label: "Clonar", hint: "Fale com uma voz da sua biblioteca" },
  { id: "design", label: "Desenhar", hint: "Crie uma voz a partir de uma descrição" },
  { id: "tts", label: "Vozes prontas", hint: "Vozes embutidas com instruções de tom" },
  { id: "voices", label: "Biblioteca", hint: "Suas vozes de referência" },
  { id: "history", label: "Histórico", hint: "Tudo o que você gerou" },
];

export default function App() {
  const [page, setPage] = useState<Page>("clone");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [active, setActive] = useState<ActiveState | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshModels = useCallback(() => api.models().then(setModels).catch((e) => setError(e.message)), []);
  const refreshVoices = useCallback(() => api.voices().then(setVoices).catch((e) => setError(e.message)), []);

  useEffect(() => {
    void refreshModels();
    void refreshVoices();
    api.active().then(setActive).catch(() => undefined);
  }, [refreshModels, refreshVoices]);

  // Poll while a model is loading.
  useEffect(() => {
    if (active?.status !== "loading") return;
    const t = setInterval(async () => {
      const s = await api.active();
      setActive(s);
      if (s.status !== "loading") void refreshModels();
    }, 700);
    return () => clearInterval(t);
  }, [active?.status, refreshModels]);

  const activate = async (id: string) => {
    setError(null);
    try {
      setActive(await api.activate(id));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const current = NAV.find((n) => n.id === page)!;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>◐</span> Morpheu
        </div>
        <nav>
          {NAV.map((n) => (
            <button key={n.id} className={`nav-item ${page === n.id ? "on" : ""}`} onClick={() => setPage(n.id)}>
              {n.label}
            </button>
          ))}
        </nav>
        <ActiveBadge active={active} models={models} />
      </aside>

      <main className="main">
        <header className="page-head">
          <h1>{current.label}</h1>
          <p className="muted">{current.hint}</p>
        </header>
        {error && (
          <div className="alert" role="alert" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        {(page === "clone" || page === "design" || page === "tts") && (
          <Studio
            key={page}
            mode={page}
            models={models}
            active={active}
            voices={voices}
            onActivate={activate}
            onVoicesChanged={refreshVoices}
            goToVoices={() => setPage("voices")}
          />
        )}
        {page === "voices" && <Voices voices={voices} onChanged={refreshVoices} />}
        {page === "history" && <History onVoicesChanged={refreshVoices} />}
      </main>
    </div>
  );
}

function ActiveBadge({ active, models }: { active: ActiveState | null; models: ModelInfo[] }) {
  if (!active?.model_id) return <div className="gpu-badge muted">Nenhum modelo carregado</div>;
  const name = models.find((m) => m.id === active.model_id)?.name ?? active.model_id;
  return (
    <div className={`gpu-badge s-${active.status}`}>
      <div className="dot" />
      <div>
        <div className="gpu-name">{name}</div>
        <div className="muted small">
          {active.status === "loading" ? `${active.message} ${Math.round(active.progress * 100)}%` : active.status === "ready" ? "Na GPU" : active.message}
        </div>
      </div>
    </div>
  );
}
