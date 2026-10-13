export type Mode = "clone" | "design" | "tts";

export interface ParamSpec {
  key: string;
  label: string;
  description: string;
  kind: "float" | "int" | "bool";
  default: number | boolean;
  min: number | null;
  max: number | null;
  step: number | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  modes: Mode[];
  languages: string[];
  params: ParamSpec[];
  vram_gb: number;
  ref_min_sec: number;
  ref_max_sec: number;
  uses_ref_text: boolean;
  max_chunk_chars: number;
  sample_rate: number;
  recommended_for: string[];
  status: "unloaded" | "loading" | "ready" | "error";
}

export interface ActiveState {
  model_id: string | null;
  status: "idle" | "loading" | "ready" | "error";
  progress: number;
  message: string;
  error: string | null;
}

export interface Voice {
  id: string;
  name: string;
  transcript: string;
  source: "upload" | "recording" | "design";
  design_prompt: string | null;
  duration: number;
  created_at: number;
  language: string;
  transcript_source: "manual" | "auto" | "";
  transcript_mismatch: boolean;
  heard_text: string;
  has_timings: boolean;
}

export type ParamValue = number | boolean;

export interface Generation {
  id: string;
  created_at: number;
  mode: Mode;
  model_id: string;
  text: string;
  language: string;
  voice_id: string | null;
  speaker: string | null;
  instruct: string | null;
  params: Record<string, ParamValue | string>;
  duration: number;
  sample_rate: number;
  elapsed: number;
}

export interface GenerateRequest {
  mode: Mode;
  model_id: string;
  text: string;
  language: string;
  voice_id?: string;
  instruct?: string;
  speaker?: string;
  params: Record<string, ParamValue>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  models: () => request<ModelInfo[]>("/api/v1/models"),
  active: () => request<ActiveState>("/api/v1/models/active"),
  activate: (model_id: string) => request<ActiveState>("/api/v1/models/active", json("POST", { model_id })),
  speakers: (model_id: string) => request<string[]>(`/api/v1/tts/speakers?model_id=${encodeURIComponent(model_id)}`),

  generate: (req: GenerateRequest) => request<Generation>("/api/v1/tts/generate", json("POST", req)),

  voices: () => request<Voice[]>("/api/v1/voices"),
  system: () => request<{ asr: boolean }>("/api/v1/system"),
  createVoice: (name: string, transcript: string, wav: Blob, source: "upload" | "recording", language: string) => {
    const fd = new FormData();
    fd.append("name", name);
    fd.append("transcript", transcript);
    fd.append("source", source);
    fd.append("language", language);
    fd.append("audio", wav, "reference.wav");
    return request<Voice>("/api/v1/voices", { method: "POST", body: fd });
  },
  updateVoice: (id: string, patch: { name?: string; transcript?: string; language?: string }) =>
    request<Voice>(`/api/v1/voices/${id}`, json("PATCH", patch)),
  analyzeVoice: (id: string, language?: string) =>
    request<Voice>(`/api/v1/voices/${id}/analyze`, json("POST", language ? { language } : {})),
  deleteVoice: (id: string) => request<void>(`/api/v1/voices/${id}`, { method: "DELETE" }),
  voiceFromHistory: (history_id: string, name: string) =>
    request<Voice>("/api/v1/voices/from-history", json("POST", { history_id, name })),
  voiceAudioUrl: (id: string) => `/api/v1/voices/${id}/audio`,

  history: () => request<Generation[]>("/api/v1/history"),
  deleteHistory: (id: string) => request<void>(`/api/v1/history/${id}`, { method: "DELETE" }),
  historyAudioUrl: (id: string, download = false) => `/api/v1/history/${id}/audio${download ? "?download=true" : ""}`,
};
