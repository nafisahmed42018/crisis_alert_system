import { ScoreResult } from "@/lib/types";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { AlertTriangle, Users, Clock, TrendingUp, Siren, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

const TRAJECTORY_CONFIG: Record<string, { className: string; label: string; Icon: React.ElementType }> = {
  EMERGENCY_INTERVENTION: { className: "bg-destructive/10 text-destructive border-destructive/30",                                   label: "Emergency Intervention", Icon: Siren        },
  ESCALATING:             { className: "bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/30 dark:text-orange-300", label: "Escalating",             Icon: TrendingUp   },
  STABILIZING:            { className: "bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900/30 dark:text-yellow-300", label: "Stabilizing",            Icon: Minus        },
  DE_ESCALATING:          { className: "bg-green-100 text-green-800 border-green-300 dark:bg-green-900/30 dark:text-green-300",      label: "De-escalating",          Icon: TrendingDown },
};

const CRISIS_LABELS: Record<string, string> = {
  environmental_disaster: "Environmental Disaster",
  corporate_misconduct:   "Corporate Misconduct",
  health_hazard:          "Health Hazard",
  policy_regulatory:      "Policy / Regulatory",
  market_financial:       "Market / Financial",
  civil_unrest:           "Civil Unrest",
  general:                "General Crisis",
};

export function RecommendationCard({ result }: { result: ScoreResult }) {
  const traj = TRAJECTORY_CONFIG[result.sentiment_trajectory] ?? TRAJECTORY_CONFIG.STABILIZING;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between border-b pb-3">
        <CardTitle className="text-sm">
          {CRISIS_LABELS[result.crisis_type] ?? result.crisis_type}
        </CardTitle>
        <span className={cn("inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold", traj.className)}>
          <traj.Icon size={11} /> {traj.label}
        </span>
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2">
        <div>
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <AlertTriangle size={12} /> Escalation
          </div>
          <ul className="space-y-1">
            {result.escalation_path.map((ch) => (
              <li key={ch} className="flex items-start gap-1.5 text-sm">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-orange-400" />
                {ch}
              </li>
            ))}
          </ul>
          <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
            <Clock size={11} /> {result.escalation_timing}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <TrendingUp size={12} /> Actions
          </div>
          <ul className="space-y-1">
            {result.recommended_actions.map((a) => (
              <li key={a} className="flex items-start gap-1.5 text-sm">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                {a}
              </li>
            ))}
          </ul>
        </div>

        <div className="sm:col-span-2">
          <Separator className="mb-3" />
          <div className="flex items-start gap-2">
            <Users size={13} className="mt-0.5 shrink-0 text-muted-foreground" />
            <div className="flex flex-wrap gap-1">
              <span className="mr-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Notify:</span>
              {result.stakeholders.map((s) => (
                <Badge key={s} variant="secondary" className="text-xs">{s}</Badge>
              ))}
            </div>
          </div>
        </div>

        {result.predicted_peak && (
          <div className="sm:col-span-2 flex items-center gap-1 text-xs text-muted-foreground">
            <TrendingUp size={11} />
            Predicted peak:{" "}
            <span className="ml-1 font-medium text-destructive">{result.predicted_peak}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
