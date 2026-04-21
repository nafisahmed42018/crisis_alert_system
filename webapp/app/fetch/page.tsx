"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { FetchResponse } from "@/lib/types";
import { TweetCard } from "@/components/TweetCard";
import { Radio, Plus, X, Loader2, Wifi, WifiOff } from "lucide-react";

const PRESET_KEYWORDS = [
  ["wildfire", "evacuation", "emergency"],
  ["oil spill", "environmental disaster"],
  ["flood warning", "river levels"],
  ["earthquake", "tsunami warning"],
  ["chemical leak", "hazmat"],
];

export default function FetchPage() {
  const [keywords, setKeywords]   = useState<string[]>(["wildfire", "evacuation"]);
  const [input, setInput]         = useState("");
  const [maxResults, setMaxResults] = useState(20);
  const [response, setResponse]   = useState<FetchResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");

  const addKeyword = () => {
    const kw = input.trim();
    if (kw && !keywords.includes(kw)) setKeywords([...keywords, kw]);
    setInput("");
  };

  const removeKeyword = (kw: string) => setKeywords(keywords.filter((k) => k !== kw));

  const fetch = async () => {
    if (keywords.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const r = await api.fetchAndAnalyze(keywords, maxResults);
      setResponse(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to reach backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Radio size={22} className="text-sky-500" /> Fetch from 𝕏 Platform
        </h1>
        <p className="text-gray-500 text-sm mt-1">
          Enter trigger keywords to pull recent tweets from X and score them through the ensemble.
          Requires an <code className="bg-gray-100 px-1 rounded text-xs">X_BEARER_TOKEN</code> in{" "}
          <code className="bg-gray-100 px-1 rounded text-xs">.env</code>; falls back to demo mode without one.
        </p>
      </div>

      {/* Keyword builder */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-4">
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-2">
            Active Keywords
          </label>
          <div className="flex flex-wrap gap-2 mb-3 min-h-[32px]">
            {keywords.map((kw) => (
              <span
                key={kw}
                className="inline-flex items-center gap-1 bg-blue-100 text-blue-800 rounded-full px-3 py-1 text-sm"
              >
                {kw}
                <button onClick={() => removeKeyword(kw)} className="hover:text-red-600 transition-colors">
                  <X size={12} />
                </button>
              </span>
            ))}
            {keywords.length === 0 && (
              <span className="text-xs text-gray-400">Add at least one keyword…</span>
            )}
          </div>

          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addKeyword()}
              placeholder="Add keyword…"
              className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
            <button
              onClick={addKeyword}
              className="flex items-center gap-1 bg-gray-100 border border-gray-200 rounded-lg px-3 py-2 text-sm hover:bg-gray-200 transition-colors"
            >
              <Plus size={14} /> Add
            </button>
          </div>
        </div>

        {/* Presets */}
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-2">
            Presets
          </label>
          <div className="flex flex-wrap gap-2">
            {PRESET_KEYWORDS.map((preset) => (
              <button
                key={preset[0]}
                onClick={() => setKeywords(preset)}
                className="text-xs bg-gray-50 border border-gray-200 rounded-full px-3 py-1 hover:border-blue-400 hover:text-blue-600 transition-colors"
              >
                {preset.join(" + ")}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide block mb-1">
              Max Results
            </label>
            <select
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none"
            >
              {[10, 20, 50, 100].map((n) => (
                <option key={n} value={n}>{n} tweets</option>
              ))}
            </select>
          </div>

          <button
            onClick={fetch}
            disabled={loading || keywords.length === 0}
            className="mt-5 flex items-center gap-2 bg-sky-600 text-white rounded-xl px-5 py-2.5 text-sm font-semibold hover:bg-sky-700 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Radio size={15} />}
            {loading ? "Fetching…" : "Fetch & Analyze"}
          </button>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {/* Results */}
      {response && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            {response.x_api_live ? (
              <span className="flex items-center gap-1.5 text-sm text-green-700 bg-green-50 border border-green-200 rounded-full px-3 py-1">
                <Wifi size={13} /> Live from 𝕏 — {response.fetched} tweets fetched
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-sm text-gray-600 bg-gray-100 border border-gray-200 rounded-full px-3 py-1">
                <WifiOff size={13} /> Demo mode (no X_BEARER_TOKEN configured)
              </span>
            )}
            <span className="text-sm text-gray-500">{response.results.length} scored</span>
          </div>

          <div className="space-y-3">
            {response.results.map((r) => (
              <TweetCard key={r.id} result={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
