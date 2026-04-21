import { ScoreResult, FetchResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

export const api = {
  analyze: (text: string, demo_mode = true) =>
    post<ScoreResult>("/api/analyze", { text, demo_mode }),

  fetchAndAnalyze: (keywords: string[], max_results = 50, demo_mode = true) =>
    post<FetchResponse>("/api/fetch-and-analyze", { keywords, max_results, demo_mode }),

  getAlerts: (level?: string, limit = 50) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (level) params.append("level", level);
    return get<ScoreResult[]>(`/api/alerts?${params}`);
  },

  clearAlerts: () =>
    fetch(`${BASE}/api/alerts`, { method: "DELETE" }).then((r) => r.json()),
};
