"use client";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";

interface Props { bert: number; lstm: number; lda: number }

export function ModelBars({ bert, lstm, lda }: Props) {
  const data = [
    { name: "BERT (40%)",  score: bert,  fill: "#dc2626" },
    { name: "LSTM (40%)", score: lstm,  fill: "#3b82f6" },
    { name: "LDA  (20%)", score: lda,   fill: "#10b981" },
  ];

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical" margin={{ left: 16, right: 40, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : v} />
        <ReferenceLine x={0.5} stroke="#6b7280" strokeDasharray="4 2" />
        {data.map((d) => (
          <Bar key={d.name} dataKey="score" fill={d.fill} radius={[0, 4, 4, 0]}
               label={{ position: "right", fontSize: 11, formatter: (v: unknown) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "" }} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
