import { ScoreResult } from "@/lib/types";
import { AlertTriangle, Users, Clock, TrendingUp } from "lucide-react";

const TRAJECTORY_CONFIG = {
  ESCALATING:    { color: "text-red-600",    label: "↑ Escalating",    bg: "bg-red-50" },
  STABILIZING:   { color: "text-yellow-600", label: "→ Stabilizing",   bg: "bg-yellow-50" },
  DE_ESCALATING: { color: "text-green-600",  label: "↓ De-escalating", bg: "bg-green-50" },
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
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
        <span className="font-semibold text-gray-700 text-sm">
          {CRISIS_LABELS[result.crisis_type] ?? result.crisis_type}
        </span>
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${traj.bg} ${traj.color}`}>
          {traj.label}
        </span>
      </div>

      <div className="p-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* Escalation path */}
        <div>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            <AlertTriangle size={12} />
            Escalation
          </div>
          <ul className="space-y-1">
            {result.escalation_path.map((ch) => (
              <li key={ch} className="text-sm text-gray-700 flex items-start gap-1.5">
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-orange-400 shrink-0" />
                {ch}
              </li>
            ))}
          </ul>
          <div className="flex items-center gap-1 mt-2 text-xs text-gray-500">
            <Clock size={11} />
            {result.escalation_timing}
          </div>
        </div>

        {/* Recommended actions */}
        <div>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            <TrendingUp size={12} />
            Actions
          </div>
          <ul className="space-y-1">
            {result.recommended_actions.map((a) => (
              <li key={a} className="text-sm text-gray-700 flex items-start gap-1.5">
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
                {a}
              </li>
            ))}
          </ul>
        </div>

        {/* Stakeholders */}
        <div className="sm:col-span-2 border-t border-gray-100 pt-3 flex items-start gap-2">
          <Users size={13} className="text-gray-400 mt-0.5 shrink-0" />
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide mr-2">Notify:</span>
            {result.stakeholders.map((s) => (
              <span key={s} className="inline-block text-xs bg-gray-100 text-gray-700 rounded px-2 py-0.5 mr-1 mb-1">
                {s}
              </span>
            ))}
          </div>
        </div>

        {result.predicted_peak && (
          <div className="sm:col-span-2 text-xs text-gray-500 flex items-center gap-1">
            <TrendingUp size={11} />
            Predicted peak: <span className="font-medium text-red-600 ml-1">{result.predicted_peak}</span>
          </div>
        )}
      </div>
    </div>
  );
}
