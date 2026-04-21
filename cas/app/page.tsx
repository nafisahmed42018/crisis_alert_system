"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { ScoreResult, AlertLevel } from "@/lib/types";
import { AlertBadge } from "@/components/AlertBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, Search, Radio, AlertTriangle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const LEVEL_COLORS: Record<AlertLevel, string> = {
  LOW:      "#6b7280",
  MEDIUM:   "#f59e0b",
  HIGH:     "#f97316",
  CRITICAL: "#dc2626",
};

export default function DashboardPage() {
  const [alerts, setAlerts] = useState<ScoreResult[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAlerts = async () => {
    try {
      const data = await api.getAlerts(undefined, 100);
      setAlerts(data);
    } catch {
      /* backend not yet started */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    const id = setInterval(fetchAlerts, 10_000);
    return () => clearInterval(id);
  }, []);

  const counts = (["CRITICAL", "HIGH", "MEDIUM", "LOW"] as AlertLevel[]).map((lvl) => ({
    level: lvl,
    count: alerts.filter((a) => a.alert_level === lvl).length,
  }));

  const avgProb = alerts.length
    ? (alerts.reduce((s, a) => s + a.crisis_probability, 0) / alerts.length).toFixed(3)
    : "—";

  const recent = [...alerts].slice(0, 8);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Crisis Alert Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Real-time social media monitoring — BERT (40%) + LSTM (40%) + LDA (20%)
        </p>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
        {[
          { label: "Total Alerts",    value: alerts.length, className: "text-foreground" },
          { label: "CRITICAL",          value: counts[0].count, className: "text-destructive" },
          { label: "HIGH",             value: counts[1].count, className: "text-orange-500" },
          { label: "MEDIUM",           value: counts[2].count, className: "text-yellow-500" },
          { label: "Avg Probability", value: avgProb,         className: "text-foreground" },
        ].map((k) => (
          <Card key={k.label} size="sm">
            <CardContent className="pt-3 pb-3 text-center">
              <div className="text-xs text-muted-foreground mb-1">{k.label}</div>
              <div className={`text-2xl font-bold ${k.className}`}>{k.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Alert distribution chart */}
        <Card>
          <CardHeader className="pb-0">
            <CardTitle className="text-sm font-semibold text-muted-foreground">Alert Level Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
                No alerts yet — analyze some tweets to see results.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={counts} margin={{ top: 4, bottom: 4 }}>
                  <XAxis dataKey="level" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {counts.map((c) => (
                      <Cell key={c.level} fill={LEVEL_COLORS[c.level]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Quick-start cards */}
        <div className="grid grid-cols-1 gap-4">
          {[
            {
              href: "/analyzer",
              icon: <Search size={22} className="text-primary" />,
              title: "Single Tweet Analyzer",
              desc: "Paste any tweet to get an instant ensemble crisis score and recommendations.",
            },
            {
              href: "/fetch",
              icon: <Radio size={22} className="text-sky-500" />,
              title: "Fetch from 𝕏 Platform",
              desc: "Enter trigger keywords to fetch live tweets and run them through the pipeline.",
            },
            {
              href: "/alerts",
              icon: <AlertTriangle size={22} className="text-orange-500" />,
              title: "Alert History",
              desc: "Browse all processed alerts with full recommendations and escalation paths.",
            },
          ].map((c) => (
            <Link key={c.href} href={c.href}>
              <Card className="cursor-pointer transition-shadow hover:shadow-md">
                <CardContent className="flex items-start gap-3 pt-4 pb-4">
                  <div className="mt-0.5">{c.icon}</div>
                  <div>
                    <div className="font-semibold text-sm">{c.title}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">{c.desc}</div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent alerts */}
      {recent.length > 0 && (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="flex items-center gap-2 font-semibold text-sm text-muted-foreground">
              <TrendingUp size={16} /> Recent Alerts
            </h2>
            <Link href="/alerts" className="text-sm text-primary hover:underline">
              View all →
            </Link>
          </div>
          <div className="space-y-2">
            {recent.map((a) => (
              <Card key={a.id} size="sm">
                <CardContent className="flex items-center gap-3 py-3">
                  <AlertBadge level={a.alert_level} prob={a.crisis_probability} />
                  <p className="flex-1 truncate text-sm">{a.text}</p>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {a.crisis_type.replace(/_/g, " ")}
                  </span>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="py-8 text-center text-sm text-muted-foreground">
          Connecting to backend…
        </div>
      )}
    </div>
  );
}
