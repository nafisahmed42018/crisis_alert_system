"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { FetchResponse } from "@/lib/types";
import { TweetCard } from "@/components/TweetCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Radio, Plus, X, Loader2, Wifi, WifiOff } from "lucide-react";

const PRESET_KEYWORDS = [
  ["wildfire", "evacuation", "emergency"],
  ["oil spill", "environmental disaster"],
  ["flood warning", "river levels"],
  ["earthquake", "tsunami warning"],
  ["chemical leak", "hazmat"],
];

export default function FetchPage() {
  const [keywords, setKeywords]     = useState<string[]>(["wildfire", "evacuation"]);
  const [input, setInput]           = useState("");
  const [maxResults, setMaxResults] = useState("20");
  const [response, setResponse]     = useState<FetchResponse | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");

  const addKeyword = () => {
    const kw = input.trim();
    if (kw && !keywords.includes(kw)) setKeywords([...keywords, kw]);
    setInput("");
  };

  const removeKeyword = (kw: string) => setKeywords(keywords.filter((k) => k !== kw));

  const run = async () => {
    if (keywords.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const r = await api.fetchAndAnalyze(keywords, Number(maxResults));
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
        <h1 className="flex items-center gap-2 text-2xl font-bold">
          <Radio size={22} className="text-sky-500" /> Fetch from 𝕏 Platform
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Enter trigger keywords to pull recent tweets from X and score them through the ensemble.
          Requires an <code className="rounded bg-muted px-1 text-xs">X_BEARER_TOKEN</code> in{" "}
          <code className="rounded bg-muted px-1 text-xs">.env</code>; falls back to demo mode without one.
        </p>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Active Keywords
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Active keyword tags */}
          <div className="flex min-h-8 flex-wrap gap-2">
            {keywords.map((kw) => (
              <Badge
                key={kw}
                variant="secondary"
                className="gap-1 pl-3 pr-2 py-1 text-sm"
              >
                {kw}
                <button onClick={() => removeKeyword(kw)} className="hover:text-destructive transition-colors">
                  <X size={12} />
                </button>
              </Badge>
            ))}
            {keywords.length === 0 && (
              <span className="text-xs text-muted-foreground">Add at least one keyword…</span>
            )}
          </div>

          {/* Add keyword input */}
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addKeyword()}
              placeholder="Add keyword…"
              className="flex-1"
            />
            <Button variant="outline" onClick={addKeyword} className="gap-1">
              <Plus size={14} /> Add
            </Button>
          </div>

          {/* Presets */}
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Presets</div>
            <div className="flex flex-wrap gap-2">
              {PRESET_KEYWORDS.map((preset) => (
                <Button
                  key={preset[0]}
                  variant="outline"
                  size="sm"
                  className="rounded-full text-xs"
                  onClick={() => setKeywords(preset)}
                >
                  {preset.join(" + ")}
                </Button>
              ))}
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-end gap-4">
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Max Results</div>
              <Select value={maxResults} onValueChange={setMaxResults}>
                <SelectTrigger className="w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[10, 20, 50, 100].map((n) => (
                    <SelectItem key={n} value={String(n)}>{n} tweets</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={run}
              disabled={loading || keywords.length === 0}
              className="gap-2 bg-sky-600 hover:bg-sky-700 text-white"
            >
              {loading ? <Loader2 size={15} className="animate-spin" /> : <Radio size={15} />}
              {loading ? "Fetching…" : "Fetch & Analyze"}
            </Button>
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {/* Results */}
      {response && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            {response.x_api_live ? (
              <Badge className="gap-1.5 bg-green-100 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-300">
                <Wifi size={13} /> Live from 𝕏 — {response.fetched} tweets fetched
              </Badge>
            ) : (
              <Badge variant="secondary" className="gap-1.5">
                <WifiOff size={13} /> Demo mode (no X_BEARER_TOKEN configured)
              </Badge>
            )}
            <span className="text-sm text-muted-foreground">{response.results.length} scored</span>
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
