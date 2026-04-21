import { cn } from "@/lib/utils";
import { AlertLevel } from "@/lib/types";
import { AlertOctagon, AlertTriangle, AlertCircle, ShieldCheck } from "lucide-react";

const CONFIG: Record<AlertLevel, { className: string; Icon: React.ElementType }> = {
  LOW:      { className: "bg-muted text-muted-foreground border-border",                                                              Icon: ShieldCheck   },
  MEDIUM:   { className: "bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900/30 dark:text-yellow-300",               Icon: AlertCircle   },
  HIGH:     { className: "bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/30 dark:text-orange-300",               Icon: AlertTriangle },
  CRITICAL: { className: "bg-destructive/10 text-destructive border-destructive/30",                                                 Icon: AlertOctagon  },
};

export function AlertBadge({ level, prob }: { level: AlertLevel; prob: number }) {
  const { className, Icon } = CONFIG[level];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold",
        className
      )}
    >
      <Icon size={12} /> {level} &nbsp;|&nbsp; {(prob * 100).toFixed(1)}%
    </span>
  );
}
