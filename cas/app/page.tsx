"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { ScoreResult, AlertLevel } from "@/lib/types";
import { AlertBadge } from "@/components/AlertBadge";
import { TrendingUp, Search, Radio, AlertTriangle } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

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
        <h1 className="text-2xl font-bold text-gray-900">Crisis Alert Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          Real-time social media monitoring — BERT (40%) + LSTM (40%) + LDA (20%)
        </p>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
        {[
          { label: "Total Alerts",    value: alerts.length, color: "text-gray-900" },
          { label: "🔴 CRITICAL",     value: counts[0].count, color: "text-red-600" },
          { label: "🟠 HIGH",         value: counts[1].count, color: "text-orange-500" },
          { label: "🟡 MEDIUM",       value: counts[2].count, color: "text-yellow-500" },
          { label: "Avg Probability", value: avgProb,         color: "text-gray-700" },
        ].map((k) => (
          <div key={k.label} className="bg-white rounded-xl border border-gray-200 p-4 text-center shadow-sm">
            <div className="text-xs text-gray-500 mb-1">{k.label}</div>
            <div className={`text-2xl font-bold ${k.color}`}>{k.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alert distribution chart */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h2 className="font-semibold text-gray-700 mb-4">Alert Level Distribution</h2>
          {alerts.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-gray-400 text-sm">
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
        </div>

        {/* Quick-start cards */}
        <div className="grid grid-cols-1 gap-4">
          {[
            {
              href: "/analyzer",
              icon: <Search size={22} className="text-blue-500" />,
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
            <Link
              key={c.href}
              href={c.href}
              className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm hover:shadow-md hover:border-blue-300 transition-all flex items-start gap-3"
            >
              <div className="mt-0.5">{c.icon}</div>
              <div>
                <div className="font-semibold text-gray-800 text-sm">{c.title}</div>
                <div className="text-gray-500 text-xs mt-0.5">{c.desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent alerts */}
      {recent.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-gray-700 flex items-center gap-2">
              <TrendingUp size={16} /> Recent Alerts
            </h2>
            <Link href="/alerts" className="text-sm text-blue-500 hover:underline">
              View all →
            </Link>
          </div>
          <div className="space-y-2">
            {recent.map((a) => (
              <div
                key={a.id}
                className="bg-white border border-gray-200 rounded-lg px-4 py-3 flex items-center gap-3 shadow-sm"
              >
                <AlertBadge level={a.alert_level} prob={a.crisis_probability} />
                <p className="text-sm text-gray-700 truncate flex-1">{a.text}</p>
                <span className="text-xs text-gray-400 shrink-0">{a.crisis_type.replace(/_/g, " ")}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center text-sm text-gray-400 py-8">
          Connecting to backend…
        </div>
      )}
    </div>
  );
}
