"use client";
import { AlertLevel } from "@/lib/types";

const COLORS: Record<AlertLevel, string> = {
  LOW:      "#6b7280",
  MEDIUM:   "#f59e0b",
  HIGH:     "#f97316",
  CRITICAL: "#dc2626",
};

export function ScoreGauge({ prob, level }: { prob: number; level: AlertLevel }) {
  const pct   = Math.round(prob * 100);
  const angle = -135 + pct * 2.7; // -135° to +135°
  const color = COLORS[level];

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 70" width="160" height="100">
        {/* track */}
        <path d="M10,65 A50,50 0 0,1 110,65" fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round" />
        {/* filled arc — simple approximation via dasharray */}
        <path d="M10,65 A50,50 0 0,1 110,65" fill="none" stroke={color} strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={`${pct * 1.57} 157`} />
        {/* needle */}
        <g transform={`rotate(${angle}, 60, 65)`}>
          <line x1="60" y1="65" x2="60" y2="22" stroke="#1f2937" strokeWidth="2" strokeLinecap="round" />
        </g>
        <circle cx="60" cy="65" r="4" fill="#1f2937" />
        <text x="60" y="58" textAnchor="middle" fontSize="11" fontWeight="bold" fill={color}>{pct}%</text>
      </svg>
      <span className="text-xs text-gray-500">Crisis Probability</span>
    </div>
  );
}
