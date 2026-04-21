"use client";
import { useState } from "react";
import { ScoreResult } from "@/lib/types";
import { AlertBadge } from "./AlertBadge";
import { ModelBars } from "./ModelBars";
import { RecommendationCard } from "./RecommendationCard";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Heart, Repeat2, MessageCircle } from "lucide-react";

export function TweetCard({ result }: { result: ScoreResult }) {
  const [expanded, setExpanded] = useState(false);
  const m = result.public_metrics;

  return (
    <Card className="gap-0 py-0">
      <CardContent className="p-4">
        <div className="mb-2 flex items-start justify-between gap-2">
          <AlertBadge level={result.alert_level} prob={result.crisis_probability} />
          {result.source === "x_api" && (
            <Badge variant="secondary" className="text-sky-700 bg-sky-100 border-sky-200 dark:bg-sky-900/30 dark:text-sky-300 text-[10px]">𝕏 Live</Badge>
          )}
          {result.source === "demo" && (
            <Badge variant="outline" className="text-[10px]">Demo</Badge>
          )}
        </div>

        <p className="mb-3 text-sm leading-relaxed">{result.text}</p>

        {m && Object.keys(m).length > 0 && (
          <div className="mb-3 flex items-center gap-4 text-xs text-muted-foreground">
            {m.like_count    != null && <span className="flex items-center gap-1"><Heart size={11} />{m.like_count}</span>}
            {m.retweet_count != null && <span className="flex items-center gap-1"><Repeat2 size={11} />{m.retweet_count}</span>}
            {m.reply_count   != null && <span className="flex items-center gap-1"><MessageCircle size={11} />{m.reply_count}</span>}
          </div>
        )}

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>BERT <b className="text-foreground">{(result.bert_score * 100).toFixed(0)}%</b></span>
          <span>LSTM <b className="text-foreground">{(result.lstm_score * 100).toFixed(0)}%</b></span>
          <span>LDA  <b className="text-foreground">{(result.lda_score  * 100).toFixed(0)}%</b></span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
            className="ml-auto h-6 gap-1 px-2 text-xs text-primary"
          >
            {expanded ? <><ChevronUp size={13} /> Less</> : <><ChevronDown size={13} /> Details</>}
          </Button>
        </div>
      </CardContent>

      {expanded && (
        <div className="border-t bg-muted/30 p-4 space-y-4">
          <ModelBars bert={result.bert_score} lstm={result.lstm_score} lda={result.lda_score} />
          <RecommendationCard result={result} />
        </div>
      )}
    </Card>
  );
}
