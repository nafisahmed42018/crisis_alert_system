import { AlertLevel } from "@/lib/types";

const CONFIG: Record<AlertLevel, { bg: string; text: string; emoji: string }> = {
  LOW:      { bg: "bg-gray-500",   text: "text-white", emoji: "🟢" },
  MEDIUM:   { bg: "bg-yellow-400", text: "text-gray-900", emoji: "🟡" },
  HIGH:     { bg: "bg-orange-500", text: "text-white", emoji: "🟠" },
  CRITICAL: { bg: "bg-red-600",    text: "text-white", emoji: "🔴" },
};

export function AlertBadge({ level, prob }: { level: AlertLevel; prob: number }) {
  const { bg, text, emoji } = CONFIG[level];
  return (
    <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-bold ${bg} ${text}`}>
      {emoji} {level} &nbsp;|&nbsp; {(prob * 100).toFixed(1)}%
    </span>
  );
}
