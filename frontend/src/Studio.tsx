import { useEffect, useMemo, useState } from "react";
import { api, type ActiveState, type Generation, type Mode, type ModelInfo, type ParamSpec, type ParamValue, type Voice } from "./api";
import { EMOTION_PT, LANG_PT } from "./langs";

const DESIGN_CHIPS: { group: string; options: string[] }[] = [
  { group: "Gênero", options: ["voz masculina", "voz feminina"] },
  { group: "Idade", options: ["criança", "jovem adulto", "meia-idade", "idoso"] },
  { group: "Tom", options: ["grave", "agudo", "rouca", "suave", "sussurrada"] },
  { group: "Estilo", options: ["calma e acolhedora", "enérgica e animada", "séria e autoritária", "narrador de documentário"] },
];

const DESIGN_EXAMPLES = [
  "Voz feminina jovem, calorosa e gentil, com tom suave e ritmo tranquilo.",
  "Voz masculina grave, cerca de 40 anos, confiante e autoritária, como um apresentador de telejornal.",
  "Voz enérgica e entusiasmada, falando rápido e sorrindo.",
  "Voz calma e serena, como um guia de meditação.",
];

interface Props {
  mode: Mode;
  models: ModelInfo[];
  active: ActiveState | null;
  voices: Voice[];
  onActivate: (id: string) => void;
  onVoicesChanged: () => void;
  goToVoices: () => void;
}

export default function Studio({ mode, models, active, voices, onActivate, onVoicesChanged, goToVoices }: Props) {
  const candidates = useMemo(() => models.filter((m) => m.modes.includes(mode)), [models, mode]);
  const [modelId, setModelId] = useState<string>("");
  const model = candidates.find((m) => m.id === modelId);

  useEffect(() => {
    if (!candidates.length || candidates.some((m) => m.id === modelId)) return;
    const loaded = candidates.find((m) => m.status === "ready");
    const recommended = candidates.find((m) => m.recommended_for.includes("Portuguese"));
    setModelId((loaded ?? recommended ?? candidates[0]).id);
  }, [candidates, modelId]);

  const [text, setText] = useState("");
  const [language, setLanguage] = useState("Auto");
  const [voiceId, setVoiceId] = useState("");
  const [instruct, setInstruct] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [params, setParams] = useState<Record<string, ParamValue>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [emotion, setEmotion] = useState("");
  const [intensity, setIntensity] = useState<"leve" | "normal" | "forte">("normal");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Generation | null>(null);

  useEffect(() => {
    if (!voiceId && voices.length) setVoiceId(voices[0].id);
  }, [voices, voiceId]);

  const voice = voices.find((v) => v.id === voiceId);

  // Default to the voice's language; models without auto-detection need an explicit one.
  useEffect(() => {
    if (!model) return;
    const preferred = mode === "clone" && voice?.language && voice.language !== "Auto" ? voice.language : null;
    setLanguage((cur) => {
      if (preferred && model.languages.includes(preferred)) return preferred;
      if (model.languages.includes(cur)) return cur;
      return model.languages.includes("Portuguese") ? "Portuguese" : model.languages[0];
    });
  }, [model?.id, voice?.id, voice?.language]);

  useEffect(() => {
    if (!model) return;
    setParams(Object.fromEntries(model.params.map((p) => [p.key, p.default])));
    if (mode === "tts")
      api.speakers(model.id).then((s) => {
        setSpeakers(s);
        setSpeaker((cur) => (s.includes(cur) ? cur : s[0] ?? ""));
      });
  }, [model?.id]);

  const isActive = active?.model_id === modelId;
  const ready = isActive && active?.status === "ready";
  const loading = isActive && active?.status === "loading";

  const missing =
    !text.trim() ? "Digite o texto." :
    mode === "clone" && !voiceId ? "Escolha uma voz." :
    mode === "design" && !instruct.trim() ? "Descreva a voz." :
    mode === "tts" && speakers.length > 0 && !speaker ? "Escolha uma voz." : null;

  const generate = async () => {
    if (!model) return;
    setBusy(true);
    setError(null);
    try {
      const item = await api.generate({
        mode, model_id: model.id, text, language, params,
        ...(mode === "clone" ? { voice_id: voiceId } : {}),
        ...(mode !== "clone" && instruct.trim() ? { instruct } : {}),
        ...(mode === "tts" ? { speaker } : {}),
        ...(mode === "clone" && emotion && model.emotions.some((e) => e.key === emotion) ? { emotion, intensity } : {}),
      });
      setResult(item);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!candidates.length) return <p className="muted">Nenhum modelo disponível para este modo.</p>;

  return (
    <div className="studio">
      <section className="card">
        <div className="row between">
          <label className="field grow">
            <span>Modelo</span>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {candidates.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.vram_gb ? `· ~${m.vram_gb} GB` : ""}
                  {m.recommended_for.includes("Portuguese") ? " · bom em português" : ""}
                  {m.status === "ready" ? " · carregado" : ""}
                </option>
              ))}
            </select>
          </label>
          {!ready && (
            <button className="btn" onClick={() => onActivate(modelId)} disabled={loading}>
              {loading ? `Carregando… ${Math.round((active?.progress ?? 0) * 100)}%` : "Carregar modelo"}
            </button>
          )}
        </div>
        {model && <p className="muted small">{model.description}</p>}
        {isActive && active?.status === "error" && <div className="alert">{active.error}</div>}
        {loading && <div className="muted small">{active?.message} (no primeiro uso o modelo é baixado do Hugging Face)</div>}
      </section>

      {mode === "clone" && (
        <section className="card">
          <div className="row between">
            <label className="field grow">
              <span>Voz</span>
              <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)} disabled={!voices.length}>
                {!voices.length && <option value="">Biblioteca vazia</option>}
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} · {v.duration.toFixed(1)} s{v.transcript ? "" : " · sem transcrição"}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn ghost" onClick={goToVoices}>+ Nova voz</button>
          </div>
          {voiceId && <audio controls src={api.voiceAudioUrl(voiceId)} className="player" />}
          {voice && model && <ReferenceHint voice={voice} model={model} />}
          {model && model.emotions.length > 0 ? (
            <div className="row wrap">
              <label className="field">
                <span>Emoção</span>
                <select value={emotion} onChange={(e) => setEmotion(e.target.value)}>
                  <option value="">Neutra (como na referência)</option>
                  {model.emotions.map((e) => <option key={e.key} value={e.key}>{e.label}</option>)}
                </select>
              </label>
              {emotion && (
                <label className="field">
                  <span>Intensidade</span>
                  <select value={intensity} onChange={(e) => setIntensity(e.target.value as typeof intensity)}>
                    <option value="leve">Leve</option>
                    <option value="normal">Normal</option>
                    <option value="forte">Forte</option>
                  </select>
                </label>
              )}
            </div>
          ) : (
            models.some((m) => m.emotions.length > 0) && (
              <p className="muted small">Para clonar com emoção, use {models.filter((m) => m.emotions.length).map((m) => m.name).join(" ou ")}.</p>
            )
          )}
        </section>
      )}

      {mode === "design" && (
        <section className="card">
          <label className="field">
            <span>Descrição da voz</span>
            <textarea rows={4} value={instruct} onChange={(e) => setInstruct(e.target.value)} placeholder="Descreva a voz que você quer criar…" />
          </label>
          <div className="chips">
            {DESIGN_CHIPS.map((g) => (
              <div key={g.group} className="chip-group">
                <span className="muted small">{g.group}</span>
                {g.options.map((o) => (
                  <button key={o} className="chip" onClick={() => setInstruct((cur) => (cur.trim() ? `${cur.trim().replace(/[.,]$/, "")}, ${o}` : o))}>
                    {o}
                  </button>
                ))}
              </div>
            ))}
          </div>
          <details>
            <summary className="muted small">Exemplos</summary>
            {DESIGN_EXAMPLES.map((ex) => (
              <button key={ex} className="example" onClick={() => setInstruct(ex)}>{ex}</button>
            ))}
          </details>
        </section>
      )}

      {mode === "tts" && (
        <section className="card">
          <label className="field">
            <span>Voz pronta</span>
            <select value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
              {speakers.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Instruções de tom <em className="muted">(opcional)</em></span>
            <textarea rows={2} value={instruct} onChange={(e) => setInstruct(e.target.value)} placeholder="Ex.: fale com empolgação; tom calmo e tranquilizador; sussurrando…" />
          </label>
        </section>
      )}

      <section className="card">
        <label className="field">
          <span className="row between">
            Texto <span className="muted small">{text.length} caracteres · {text.trim() ? text.trim().split(/\s+/).length : 0} palavras</span>
          </span>
          <textarea rows={6} value={text} maxLength={5000} onChange={(e) => setText(e.target.value)} placeholder="Digite o que a voz deve falar…" />
        </label>
        <div className="row between wrap">
          <label className="field">
            <span>Idioma</span>
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {(model?.languages ?? ["Auto"]).map((l) => <option key={l} value={l}>{LANG_PT[l] ?? l}</option>)}
            </select>
          </label>
          <button className="btn ghost" onClick={() => setShowAdvanced((v) => !v)}>
            {showAdvanced ? "Ocultar ajustes" : "Ajustes avançados"}
          </button>
        </div>
        {showAdvanced && model && (
          <ParamsPanel specs={model.params} values={params} onChange={(k, v) => setParams((p) => ({ ...p, [k]: v }))}
            onReset={() => setParams(Object.fromEntries(model.params.map((p) => [p.key, p.default])))} />
        )}
        <div className="row end">
          {!ready && <span className="muted small">Carregue o modelo para gerar.</span>}
          {ready && missing && <span className="muted small">{missing}</span>}
          <button className="btn primary" onClick={generate} disabled={!ready || !!missing || busy}>
            {busy ? "Gerando…" : "Gerar fala"}
          </button>
        </div>
        {error && <div className="alert">{error}</div>}
      </section>

      {result && <ResultCard item={result} onSaved={onVoicesChanged} />}
    </div>
  );
}

function ReferenceHint({ voice, model }: { voice: Voice; model: ModelInfo }) {
  const notes: string[] = [];
  if (voice.duration < model.ref_min_sec)
    notes.push(`${model.name} precisa de pelo menos ${model.ref_min_sec} s de referência.`);
  else if (voice.duration > model.ref_max_sec + 0.5)
    notes.push(`Este modelo clona melhor com até ${model.ref_max_sec} s: será usado o trecho mais fluente da referência que começa e termina em fim de frase.`);
  if (model.uses_ref_text && !voice.transcript)
    notes.push("Sem transcrição: ela será feita automaticamente na primeira geração.");
  if (voice.transcript_mismatch)
    notes.push("A transcrição não bate com o que se ouve no áudio. Corrija-a na Biblioteca para clonar melhor.");
  if (!model.uses_ref_text) notes.push("Este modelo usa só o áudio da referência, não a transcrição.");
  if (!notes.length) return null;
  return (
    <ul className={`hints small ${voice.transcript_mismatch || voice.duration < model.ref_min_sec ? "warn" : "muted"}`}>
      {notes.map((n) => <li key={n}>{n}</li>)}
    </ul>
  );
}

export function GenerationDetails({ item }: { item: Generation }) {
  const p = item.params;
  const EMO_INT: Record<string, string> = { leve: "leve", normal: "normal", forte: "forte" };
  const lines = [
    typeof p.emotion === "string" ? `Emoção: ${EMOTION_PT[p.emotion] ?? p.emotion} (${EMO_INT[String(p.intensity)] ?? p.intensity})` : null,
    typeof p.reference === "string" ? `Referência: ${p.reference}` : null,
    Number(p.chunks) > 1 ? `Gerado em ${p.chunks} trechos` : null,
    p.retries ? `${p.retries} trecho(s) refeito(s) por saída defeituosa` : null,
    typeof p.spoken_text === "string" ? `Texto falado: ${p.spoken_text}` : null,
  ].filter(Boolean);
  if (!lines.length) return null;
  return (
    <ul className="hints small muted">
      {lines.map((l) => <li key={l as string}>{l}</li>)}
    </ul>
  );
}

function ParamsPanel({ specs, values, onChange, onReset }: {
  specs: ParamSpec[];
  values: Record<string, ParamValue>;
  onChange: (k: string, v: ParamValue) => void;
  onReset: () => void;
}) {
  return (
    <div className="params">
      {specs.map((p) => (
        <label key={p.key} className="param" title={p.description}>
          <span className="row between">
            {p.label}
            {p.kind !== "bool" && <span className="mono">{String(values[p.key] ?? p.default)}</span>}
          </span>
          {p.kind === "bool" ? (
            <input type="checkbox" checked={Boolean(values[p.key])} onChange={(e) => onChange(p.key, e.target.checked)} />
          ) : p.key === "seed" ? (
            <input type="number" min={0} value={Number(values[p.key] ?? 0)} onChange={(e) => onChange(p.key, Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
          ) : (
            <input type="range" min={p.min ?? 0} max={p.max ?? 1} step={p.step ?? 0.01} value={Number(values[p.key] ?? p.default)}
              onChange={(e) => onChange(p.key, p.kind === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))} />
          )}
          <span className="muted small">{p.description}</span>
        </label>
      ))}
      <button className="btn ghost small" onClick={onReset}>Restaurar padrões</button>
    </div>
  );
}

function ResultCard({ item, onSaved }: { item: Generation; onSaved: () => void }) {
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    try {
      await api.voiceFromHistory(item.id, name);
      setSaved(true);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <section className="card result">
      <div className="row between">
        <strong>Resultado</strong>
        <span className="muted small">
          {item.duration.toFixed(1)} s de áudio · gerado em {item.elapsed.toFixed(1)} s · semente {String(item.params.seed)}
        </span>
      </div>
      <audio controls autoPlay src={api.historyAudioUrl(item.id)} className="player" />
      <GenerationDetails item={item} />
      <div className="row between wrap">
        <a className="btn ghost" href={api.historyAudioUrl(item.id, true)}>Baixar WAV</a>
        {item.mode === "design" && !saved && (
          <div className="row">
            <input placeholder="Nome da nova voz" value={name} onChange={(e) => setName(e.target.value)} />
            <button className="btn" onClick={save} disabled={!name.trim()}>Salvar na biblioteca</button>
          </div>
        )}
        {saved && <span className="ok">Voz salva — já pode usá-la em Clonar.</span>}
      </div>
      {error && <div className="alert">{error}</div>}
    </section>
  );
}
