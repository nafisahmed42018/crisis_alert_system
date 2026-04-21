"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { ScoreResult, AlertLevel } from "@/lib/types";
import { TweetCard } from "@/components/TweetCard";
import { Button } from "@/components/ui/button";
import { RefreshCw, Trash2, Filter } from "lucide-react";
import { cn } from "@/lib/utils";

const LEVELS: (AlertLevel | "ALL")[] = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"];

export default function AlertsPage() {
  const [alerts, setAlerts]   = useState<ScoreResult[]>([]);
  const [filter, setFilter]   = useState<AlertLevel | "ALL">("ALL");
  const [loading, setLoading] = useState(true);

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
          <h1 className="text-2xl font-bold">Alert History</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {alerts.length} alert{alerts.length !== 1 ? "s" : ""} in memory
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={load} className="gap-1.5">
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={clearAll}
            className="gap-1.5 border-destructive/30 text-destructive hover:bg-destructive/10"
          >
            <Trash2 size={13} />
            Clear All
          </Button>
        </div>
      </div>

      {/* Level filter */}
      <div className="flex items-center gap-2">
        <Filter size={13} className="text-muted-foreground" />
        {LEVELS.map((lvl) => (
          <button
            key={lvl}
            onClick={() => setFilter(lvl)}
            className={cn(
              "rounded-full border px-3 py-1 text-xs transition-colors",
              filter === lvl
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background text-muted-foreground border-border hover:border-primary/50"
            )}
          >
            {lvl}
          </button>
        ))}
      </div>

      {/* Alert list */}
      {loading && alerts.length === 0 ? (
        <div className="py-12 text-center text-sm text-muted-foreground">Loading…</div>
      ) : alerts.length === 0 ? (
        <div className="rounded-xl border-2 border-dashed border-border py-12 text-center text-sm text-muted-foreground">
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
