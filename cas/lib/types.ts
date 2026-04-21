export type AlertLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface ScoreResult {
  id: string;
  text: string;
  bert_score: number;
  lstm_score: number;
  lda_score: number;
  crisis_probability: number;
  alert_level: AlertLevel;
  crisis_type: string;
  escalation_path: string[];
  escalation_timing: string;
  recommended_actions: string[];
  stakeholders: string[];
  sentiment_trajectory: "EMERGENCY_INTERVENTION" | "ESCALATING" | "STABILIZING" | "DE_ESCALATING";
  predicted_peak: string | null;
  source: string;
  created_at: string;
  public_metrics: Record<string, number>;
  timestamp: string;
}

export interface FetchResponse {
  keywords: string[];
  fetched: number;
  results: ScoreResult[];
  x_api_live: boolean;
}
