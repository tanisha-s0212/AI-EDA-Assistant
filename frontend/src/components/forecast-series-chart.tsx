'use client';

import React, { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';

export const FORECAST_SERIES_CHART_COLORS = {
  actual: '#2563eb',
  backtest: '#f59e0b',
  forecast: '#8b5cf6',
  band: '#22c55e',
  bandBase: '#ecfccb',
  grid: '#cbd5e1',
  boundary: '#94a3b8',
} as const;

/** Above this point count, hide per-point markers so trends stay readable. */
export const FORECAST_CHART_DOT_THRESHOLD = 40;

export type ForecastSeriesPoint = {
  period: string;
  actual?: number | null;
  backtest?: number | null;
  forecast?: number | null;
  lower?: number | null;
  upper?: number | null;
  lowerBand?: number | null;
  confidenceRange?: number | null;
};

export function buildForecastSeriesChartData(
  history: Array<{ period: string; actual?: number | null }>,
  testForecast: Array<{ period: string; predicted?: number | null; lower?: number | null; upper?: number | null }>,
  futureForecast: Array<{ period: string; predicted?: number | null; lower?: number | null; upper?: number | null }>,
  options?: { includeConfidence?: boolean },
): ForecastSeriesPoint[] {
  const includeConfidence = Boolean(options?.includeConfidence);
  const testMap = new Map(testForecast.map((item) => [item.period, item]));
  return [
    ...history.map((item) => {
      const testPoint = testMap.get(item.period);
      const lower = includeConfidence ? (testPoint?.lower ?? null) : null;
      const upper = includeConfidence ? (testPoint?.upper ?? null) : null;
      return {
        period: item.period,
        actual: item.actual ?? null,
        backtest: testPoint?.predicted ?? null,
        forecast: null as number | null,
        ...(includeConfidence
          ? {
              lower,
              upper,
              lowerBand: lower,
              confidenceRange:
                lower != null && upper != null ? Math.max(0, upper - lower) : null,
            }
          : {}),
      };
    }),
    ...futureForecast.map((item) => {
      const lower = item.lower ?? null;
      const upper = item.upper ?? null;
      return {
        period: item.period,
        actual: null,
        backtest: null,
        forecast: item.predicted ?? null,
        ...(includeConfidence
          ? {
              lower,
              upper,
              lowerBand: lower,
              confidenceRange:
                lower != null && upper != null ? Math.max(0, upper - lower) : null,
            }
          : {}),
      };
    }),
  ];
}

export function firstForecastPeriod(data: ForecastSeriesPoint[]): string | null {
  const point = data.find((item) => item.forecast != null);
  return point?.period ?? null;
}

function formatChartValue(value: number | null | undefined) {
  return value == null || Number.isNaN(value) ? 'N/A' : value.toLocaleString();
}

function ForecastSeriesTooltip({
  active,
  payload,
  label,
  showConfidence,
}: {
  active?: boolean;
  payload?: Array<{ payload?: ForecastSeriesPoint }>;
  label?: string;
  showConfidence?: boolean;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  const rows = [
    { label: 'Actual', value: point.actual },
    { label: 'Backtest', value: point.backtest },
    { label: 'Forecast', value: point.forecast },
    ...(showConfidence
      ? [
          { label: 'Lower 95%', value: point.lower },
          { label: 'Upper 95%', value: point.upper },
        ]
      : []),
  ].filter((item) => item.value != null);
  return (
    <div className="min-w-[180px] rounded-2xl border border-slate-200/80 bg-white/95 p-3 shadow-[0_18px_45px_rgba(15,23,42,0.14)]">
      <p className="text-sm font-semibold text-slate-900">{label}</p>
      <div className="mt-2 space-y-1.5">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between gap-4 text-xs">
            <span className="text-slate-500">{row.label}</span>
            <span className="font-semibold text-slate-900">{formatChartValue(row.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type ForecastSeriesChartProps = {
  data: ForecastSeriesPoint[];
  showConfidence?: boolean;
  heightClassName?: string;
  yTickFormatter?: (value: number) => string;
};

export default function ForecastSeriesChart({
  data,
  showConfidence = false,
  heightClassName = 'h-80',
  yTickFormatter,
}: ForecastSeriesChartProps) {
  const showDots = data.length <= FORECAST_CHART_DOT_THRESHOLD;
  const forecastStart = useMemo(() => firstForecastPeriod(data), [data]);

  const actualDot = showDots
    ? { r: 4, fill: '#ffffff', stroke: FORECAST_SERIES_CHART_COLORS.actual, strokeWidth: 2.5 }
    : false;
  const backtestDot = showDots
    ? { r: 3.5, fill: '#ffffff', stroke: FORECAST_SERIES_CHART_COLORS.backtest, strokeWidth: 2 }
    : false;
  const forecastDot = showDots
    ? { r: 4, fill: '#ffffff', stroke: FORECAST_SERIES_CHART_COLORS.forecast, strokeWidth: 2.5 }
    : false;

  return (
    <div className={`${heightClassName} w-full`}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 18, left: 0, bottom: 6 }}>
          <CartesianGrid stroke={FORECAST_SERIES_CHART_COLORS.grid} strokeDasharray="3 3" opacity={0.35} vertical={false} />
          <XAxis dataKey="period" tickLine={false} axisLine={false} tickMargin={10} tick={{ fill: '#64748b', fontSize: 12 }} />
          <YAxis
            tickLine={false}
            axisLine={false}
            width={72}
            tick={{ fill: '#64748b', fontSize: 12 }}
            tickFormatter={(value) => (yTickFormatter ? yTickFormatter(Number(value)) : formatChartValue(Number(value)))}
          />
          <RechartsTooltip content={<ForecastSeriesTooltip showConfidence={showConfidence} />} />
          <Legend />
          {showConfidence ? (
            <>
              <Area
                type="monotone"
                dataKey="lowerBand"
                name="Lower 95%"
                stackId="confidence"
                stroke="transparent"
                fill={FORECAST_SERIES_CHART_COLORS.bandBase}
                fillOpacity={0.12}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="confidenceRange"
                name="95% Confidence Band"
                stackId="confidence"
                stroke="transparent"
                fill={FORECAST_SERIES_CHART_COLORS.band}
                fillOpacity={0.18}
                isAnimationActive={false}
              />
            </>
          ) : null}
          {forecastStart ? (
            <ReferenceLine
              x={forecastStart}
              stroke={FORECAST_SERIES_CHART_COLORS.boundary}
              strokeDasharray="4 4"
              label={{ value: 'Forecast', position: 'insideTopRight', fill: '#64748b', fontSize: 11 }}
            />
          ) : null}
          <Line
            type="monotone"
            connectNulls
            dataKey="actual"
            name="Actual"
            stroke={FORECAST_SERIES_CHART_COLORS.actual}
            strokeWidth={3}
            dot={actualDot}
            activeDot={{ r: 6, fill: FORECAST_SERIES_CHART_COLORS.actual, stroke: '#ffffff', strokeWidth: 2 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            connectNulls
            dataKey="backtest"
            name="Backtest"
            stroke={FORECAST_SERIES_CHART_COLORS.backtest}
            strokeWidth={2.5}
            strokeDasharray="6 4"
            dot={backtestDot}
            activeDot={{ r: 5, fill: FORECAST_SERIES_CHART_COLORS.backtest, stroke: '#ffffff', strokeWidth: 2 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            connectNulls
            dataKey="forecast"
            name="Forecast"
            stroke={FORECAST_SERIES_CHART_COLORS.forecast}
            strokeWidth={3}
            dot={forecastDot}
            activeDot={{ r: 6, fill: FORECAST_SERIES_CHART_COLORS.forecast, stroke: '#ffffff', strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
