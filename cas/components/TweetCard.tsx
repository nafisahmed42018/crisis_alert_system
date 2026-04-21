"use client";
import { useState } from "react";
import { ScoreResult } from "@/lib/types";
import { AlertBadge } from "./AlertBadge";
import { ModelBars } from "./ModelBars";
import { RecommendationCard } from "./RecommendationCard";
import { ChevronDown, ChevronUp, Heart, Repeat2, MessageCircle } from "lucide-react";

export function TweetCard({ result }: { result: ScoreResult }) {
  const [expanded, setExpanded] = useState(false);
  const m = result.public_metrics;

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="p-4">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <AlertBadge level={result.alert_level} prob={result.crisis_probability} />
          {result.source === "x_api" && (
            <span className="text-[10px] bg-sky-100 text-sky-700 rounded px-2 py-0.5 font-medium">𝕏 Live</span>
          )}
          {result.source === "demo" && (
            <span className="text-[10px] bg-gray-100 text-gray-500 rounded px-2 py-0.5">Demo</span>
          )}
        </div>

        {/* Tweet text */}
        <p className="text-sm text-gray-800 leading-relaxed mb-3">{result.text}</p>

        {/* Engagement metrics (X API only) */}
        {m && Object.keys(m).length > 0 && (
          <div className="flex items-center gap-4 text-xs text-gray-400 mb-3">
            {m.like_count     != null && <span className="flex items-center gap-1"><Heart size={11}/>{m.like_count}</span>}
            {m.retweet_count  != null && <span className="flex items-center gap-1"><Repeat2 size={11}/>{m.retweet_count}</span>}
            {m.reply_count    != null && <span className="flex items-center gap-1"><MessageCircle size={11}/>{m.reply_count}</span>}
          </div>
        )}

        {/* Model scores quick view */}
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span>BERT <b className="text-gray-800">{(result.bert_score * 100).toFixed(0)}%</b></span>
          <span>LSTM <b className="text-gray-800">{(result.lstm_score * 100).toFixed(0)}%</b></span>
          <span>LDA  <b className="text-gray-800">{(result.lda_score  * 100).toFixed(0)}%</b></span>
          <button
            onClick={() => setExpanded(!expanded)}
            className="ml-auto flex items-center gap-1 text-blue-500 hover:text-blue-700 transition-colors"
          >
            {expanded ? <><ChevronUp size={13}/> Less</> : <><ChevronDown size={13}/> Details</>}
          </button>
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="border-t border-gray-100 p-4 bg-gray-50 space-y-4">
          <ModelBars bert={result.bert_score} lstm={result.lstm_score} lda={result.lda_score} />
          <RecommendationCard result={result} />
        </div>
      )}
    </div>
  );
}
