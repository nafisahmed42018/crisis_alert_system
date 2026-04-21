"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { ScoreResult, AlertLevel } from "@/lib/types";
import { TweetCard } from "@/components/TweetCard";
import { RefreshCw, Trash2, Filter } from "lucide-react";

const LEVELS: (AlertLevel | "ALL")[] = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"];

export default function AlertsPage() {
  const [alerts, setAlerts]     = useState<ScoreResult[]>([]);
  const [filter, setFilter]     = useState<AlertLevel | "ALL">("ALL");
  const [loading, setLoading]   = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.getAlerts(filter === "ALL" ? undefined : filter, 200);
      setAlerts(data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [filter]);

  const clearAll = async () => {
    await api.clearAlerts();
    setAlerts([]);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Alert History</h1>
          <p className="text-gray-500 text-sm mt-1">
            {alerts.length} alert{alerts.length !== 1 ? "s" : ""} in memory
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={load}
            className="flex items-center gap-1.5 text-sm bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-gray-50 transition-colors"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          <button
            onClick={clearAll}
            className="flex items-center gap-1.5 text-sm bg-white border border-red-200 text-red-600 rounded-lg px-3 py-2 hover:bg-red-50 transition-colors"
          >
            <Trash2 size={13} />
            Clear All
          </button>
        </div>
      </div>

      {/* Level filter */}
      <div className="flex items-center gap-2">
        <Filter size={13} className="text-gray-400" />
        {LEVELS.map((lvl) => (
          <button
            key={lvl}
            onClick={() => setFilter(lvl)}
            className={`text-xs rounded-full px-3 py-1 border transition-colors ${
              filter === lvl
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-200 hover:border-blue-300"
            }`}
          >
            {lvl}
          </button>
        ))}
      </div>

      {/* Alert list */}
      {loading && alerts.length === 0 ? (
        <div className="text-center text-sm text-gray-400 py-12">Loading…</div>
      ) : alerts.length === 0 ? (
        <div className="text-center text-sm text-gray-400 py-12 border-2 border-dashed border-gray-200 rounded-xl">
          No alerts yet. Analyze tweets or fetch from 𝕏 to populate this list.
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map((a) => (
            <TweetCard key={a.id} result={a} />
          ))}
        </div>
      )}
    </div>
  );
}
