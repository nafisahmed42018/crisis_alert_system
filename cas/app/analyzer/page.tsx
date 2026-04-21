"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { ScoreResult } from "@/lib/types";
import { AlertBadge } from "@/components/AlertBadge";
import { ScoreGauge } from "@/components/ScoreGauge";
import { ModelBars } from "@/components/ModelBars";
import { RecommendationCard } from "@/components/RecommendationCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Search, Loader2 } from "lucide-react";

const EXAMPLES = [
  { label: "Wildfire",  text: "Massive wildfire destroys thousands of homes, evacuation ordered across three counties" },
  { label: "Oil Spill", text: "Oil spill reported near Gulf coast, marine life at critical risk, emergency teams deployed" },
  { label: "Flood",     text: "Flash flood warnings issued for low-lying areas, rivers approaching record levels" },
  { label: "Normal",    text: "Just had the best weekend camping trip, nature is so beautiful this time of year" },
  { label: "Sports",    text: "What a game last night! Cannot believe that final score, absolute thriller" },
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
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Tweet Analyzer</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Enter any tweet text to score it through the BERT + LSTM + LDA ensemble.
        </p>
      </div>

      {/* Example picker */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <Button
            key={ex.label}
            variant="outline"
            size="sm"
            className="rounded-full text-xs"
            onClick={() => { setText(ex.text); setResult(null); }}
          >
            {ex.label}
          </Button>
        ))}
      </div>

      {/* Input */}
      <div className="space-y-3">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          placeholder="Type or paste a tweet here…"
          className="resize-none"
        />
        <Button
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="gap-2"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
          {loading ? "Analyzing…" : "Analyze Tweet"}
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-5">
              <AlertBadge level={result.alert_level} prob={result.crisis_probability} />
              <p className="mt-3 text-sm leading-relaxed">{result.text}</p>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Card>
              <CardContent className="flex items-center justify-center pt-5 pb-5">
                <ScoreGauge prob={result.crisis_probability} level={result.alert_level} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Model Contributions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ModelBars bert={result.bert_score} lstm={result.lstm_score} lda={result.lda_score} />
              </CardContent>
            </Card>
          </div>

          <RecommendationCard result={result} />
        </div>
      )}
    </div>
  );
}
