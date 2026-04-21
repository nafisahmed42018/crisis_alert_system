"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { ScoreResult } from "@/lib/types";
import { AlertBadge } from "@/components/AlertBadge";
import { ScoreGauge } from "@/components/ScoreGauge";
import { ModelBars } from "@/components/ModelBars";
import { RecommendationCard } from "@/components/RecommendationCard";
import { Search, Loader2 } from "lucide-react";

const EXAMPLES = [
  { label: "Wildfire 🔴",  text: "Massive wildfire destroys thousands of homes, evacuation ordered across three counties" },
  { label: "Oil spill 🔴", text: "Oil spill reported near Gulf coast, marine life at critical risk, emergency teams deployed" },
  { label: "Flood 🟠",     text: "Flash flood warnings issued for low-lying areas, rivers approaching record levels" },
  { label: "Normal 🟢",    text: "Just had the best weekend camping trip, nature is so beautiful this time of year" },
  { label: "Sports 🟢",    text: "What a game last night! Cannot believe that final score, absolute thriller" },
];

export default function AnalyzerPage() {
  const [text, setText]       = useState(EXAMPLES[0].text);
  const [result, setResult]   = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const r = await api.analyze(text.trim());
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to reach backend. Is the API server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Tweet Analyzer</h1>
        <p className="text-gray-500 text-sm mt-1">
          Enter any tweet text to score it through the BERT + LSTM + LDA ensemble.
        </p>
      </div>

      {/* Example picker */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            onClick={() => { setText(ex.text); setResult(null); }}
            className="text-xs bg-white border border-gray-200 rounded-full px-3 py-1 hover:border-blue-400 hover:text-blue-600 transition-colors"
          >
            {ex.label}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="space-y-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          placeholder="Type or paste a tweet here…"
          className="w-full rounded-xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
        />
        <button
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="flex items-center gap-2 bg-blue-600 text-white rounded-xl px-5 py-2.5 text-sm font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
          {loading ? "Analyzing…" : "Analyze Tweet"}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <AlertBadge level={result.alert_level} prob={result.crisis_probability} />
            <p className="mt-3 text-sm text-gray-700 leading-relaxed">{result.text}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm flex items-center justify-center">
              <ScoreGauge prob={result.crisis_probability} level={result.alert_level} />
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                Model Contributions
              </h3>
              <ModelBars bert={result.bert_score} lstm={result.lstm_score} lda={result.lda_score} />
            </div>
          </div>

          <RecommendationCard result={result} />
        </div>
      )}
    </div>
  );
}
