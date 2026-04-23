"use client"
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts"

interface Props {
  bert: number
  lstm: number
  lda: number
}

interface TooltipPayload {
  payload?: { name: string; score: number }
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: TooltipPayload[]
}) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-sm shadow-md">
      <p className="font-medium">{d.name}</p>
      <p className="text-muted-foreground">{(d.score * 100).toFixed(1)}%</p>
    </div>
  )
}

export function ModelBars({ bert, lstm, lda }: Props) {
  const data = [
    { name: "BERT (40%)", score: bert, fill: "#dc2626" },
    { name: "LSTM (20%)", score: lstm, fill: "#3b82f6" },
    { name: "LDA  (40%)", score: lda, fill: "#10b981" },
  ]

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ left: 16, right: 40, top: 4, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, 1]}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={90}
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          content={<CustomTooltip />}
          cursor={{ fill: "hsl(var(--muted))", opacity: 0.5 }}
        />
        <ReferenceLine x={0.5} stroke="#6b7280" strokeDasharray="4 2" />
        <Bar
          dataKey="score"
          radius={[0, 4, 4, 0]}
          label={{
            position: "right",
            fontSize: 11,
            formatter: (v: unknown) =>
              typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "",
          }}
        >
          {data.map((d) => (
            <Cell key={d.name} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
