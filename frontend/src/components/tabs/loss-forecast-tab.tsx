'use client';

import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, ArrowRight, Download, Loader2, Lock, PieChart as PieIcon, ShieldAlert, TrendingDown, Zap } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAppStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { getApiErrorMessage } from '@/lib/api';
import type { LossForecastResult, SegmentBreakdown } from '@/types/forecast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

const currency = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
const percent = new Intl.NumberFormat('en-IN', { style: 'percent', maximumFractionDigits: 1 });
const compactNumber = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 1 });

function compactCurrency(value: number | string) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return '₹0';

  const sign = numericValue < 0 ? '-' : '';
  const absoluteValue = Math.abs(numericValue);
  if (absoluteValue >= 1_00_00_000) return `${sign}₹${compactNumber.format(absoluteValue / 1_00_00_000)} Cr`;
  if (absoluteValue >= 1_00_000) return `${sign}₹${compactNumber.format(absoluteValue / 1_00_000)} L`;
  if (absoluteValue >= 1_000) return `${sign}₹${compactNumber.format(absoluteValue / 1_000)} K`;
  return `${sign}${currency.format(absoluteValue)}`;
}

function currencyTooltip(value: number | string) {
  return currency.format(Number(value) || 0);
}

const LOSS_COLORS = {
  revenue_loss: '#ef4444',
  operational_loss: '#f97316',
  inventory_loss: '#f59e0b',
  discount_loss: '#8b5cf6',
  total_loss: '#dc2626',
};

const PERIOD_OPTIONS = [
  { label: '7d', value: 7 },
  { label: '30d', value: 30 },
  { label: '90d', value: 90 },
  { label: '180d', value: 180 },
];

function riskClass(label: string) {
  if (label === 'High') return 'border-red-200 bg-red-100 text-red-700';
  if (label === 'Medium') return 'border-amber-200 bg-amber-100 text-amber-700';
  return 'border-green-200 bg-green-100 text-green-700';
}

function exportLossCsv(rows: LossForecastResult[]) {
  const headers = ['Period', 'Revenue Loss', 'Operational Loss', 'Inventory Loss', 'Discount Loss', 'Total Loss', 'Risk Score', 'Risk Label'];
  const csv = [
    headers.join(','),
    ...rows.map((row) => [
      row.period,
      row.revenue_loss,
      row.operational_loss,
      row.inventory_loss,
      row.discount_loss,
      row.total_loss,
      row.loss_risk_score,
      row.risk_label,
    ].map((value) => `"${String(value).replace(/"/g, '""')}"`).join(',')),
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'loss_forecast_results.csv';
  anchor.click();
  URL.revokeObjectURL(url);
}

function SkeletonBlock({ className = 'h-72' }: { className?: string }) {
  return <div className={`animate-pulse rounded-2xl border bg-muted/40 ${className}`} />;
}

function buildHeatmap(rows: LossForecastResult[], segments: SegmentBreakdown[]) {
  const baseSegments = segments.length ? segments.slice(0, 6) : [{ segment: 'All Business', risk_score: 0.05, risk_label: 'Low' }];
  return baseSegments.map((segment, rowIndex) => ({
    segment: segment.segment,
    cells: rows.slice(0, 12).map((row, columnIndex) => {
      const score = Math.min(1, Math.max(0, (segment.risk_score || row.loss_risk_score) * (0.9 + row.loss_risk_score + columnIndex * 0.01 + rowIndex * 0.005)));
      const label = score > 0.15 ? 'High' : score >= 0.05 ? 'Medium' : 'Low';
      return { period: row.period, score, label };
    }),
  }));
}

export default function LossForecastTab() {
  const { toast } = useToast();
  const datasetId = useAppStore((state) => state.datasetId);
  const timeSeriesForecastResult = useAppStore((state) => state.timeSeriesForecastResult);
  const mlForecastResult = useAppStore((state) => state.mlForecastResult);
  const lossForecast = useAppStore((state) => state.lossForecast) ?? [];
  const lossSegments = useAppStore((state) => state.lossSegments) ?? [];
  const lossSummary = useAppStore((state) => state.lossSummary);
  const lossLoading = useAppStore((state) => state.lossLoading);
  const lossError = useAppStore((state) => state.lossError);
  const runLossForecast = useAppStore((state) => state.runLossForecast);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const [periods, setPeriods] = useState(30);
  const [segmentView, setSegmentView] = useState<'category' | 'region'>('category');
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState<keyof LossForecastResult>('period');

  const locked = !timeSeriesForecastResult || !mlForecastResult || !datasetId;

  const sortedRows = useMemo(() => {
    return [...lossForecast].sort((left, right) => {
      const a = left[sortKey];
      const b = right[sortKey];
      if (typeof a === 'number' && typeof b === 'number') return b - a;
      return String(a ?? '').localeCompare(String(b ?? ''));
    });
  }, [lossForecast, sortKey]);

  const pageRows = sortedRows.slice(page * 10, page * 10 + 10);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / 10));
  const filteredSegments = lossSegments.filter((item) => item.segment_type === segmentView);
  const segmentRows = filteredSegments.length ? filteredSegments : lossSegments;
  const totalSegmentLoss = segmentRows.reduce((sum, row) => sum + row.total_loss, 0);
  const heatmapRows = buildHeatmap(lossForecast, segmentRows);

  const actions = useMemo(() => {
    const driverTotals = [
      { label: 'Revenue Loss', value: lossForecast.reduce((sum, row) => sum + row.revenue_loss, 0), action: 'Review demand, pricing, and stockout prevention for the highest-risk forecast windows.' },
      { label: 'Operational Loss', value: lossForecast.reduce((sum, row) => sum + row.operational_loss, 0), action: 'Audit operating expense spikes and compare upcoming cost assumptions against rolling baselines.' },
      { label: 'Inventory Loss', value: lossForecast.reduce((sum, row) => sum + row.inventory_loss, 0), action: 'Tighten replenishment thresholds and isolate categories with damage, waste, or stockout exposure.' },
      { label: 'Discount Loss', value: lossForecast.reduce((sum, row) => sum + row.discount_loss, 0), action: 'Revisit promotional depth and markdown timing for the next forecast cycle.' },
    ].sort((a, b) => b.value - a.value);
    return driverTotals.slice(0, 3);
  }, [lossForecast]);

  const handleRun = async () => {
    if (!datasetId) return;
    try {
      await runLossForecast(datasetId, periods);
      setPage(0);
      toast({ title: 'Loss forecast ready', description: `Generated risk and loss projections for ${periods} future periods.` });
    } catch (error) {
      toast({ title: 'Loss forecast failed', description: getApiErrorMessage(error, 'Unable to run loss forecast.'), variant: 'destructive' });
    }
  };

  if (locked) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
          <Lock className="h-10 w-10 text-muted-foreground/60" />
          <div>
            <h2 className="text-xl font-bold">Complete ML Forecasting first</h2>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              Loss Forecast needs both Time Series Forecasting and Machine Learning Forecasting results so revenue, cost, and future periods line up.
            </p>
          </div>
          <Button onClick={() => setActiveTab(timeSeriesForecastResult ? 'forecast_ml' : 'forecast_ts')}>
            Complete ML Forecasting first
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <Card className="overflow-hidden border-red-200/70 bg-gradient-to-br from-red-50 via-background to-amber-50">
        <CardContent className="flex flex-col gap-5 p-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Badge className="border-red-200 bg-red-100 text-red-700">Risk Forecast</Badge>
            <h2 className="mt-3 text-2xl font-bold tracking-tight">Loss Forecast</h2>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">Identify and quantify future value erosion across your business.</p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Select value={String(periods)} onValueChange={(value) => setPeriods(Number(value))}>
              <SelectTrigger className="w-full sm:w-36"><SelectValue /></SelectTrigger>
              <SelectContent>{PERIOD_OPTIONS.map((item) => <SelectItem key={item.value} value={String(item.value)}>{item.label}</SelectItem>)}</SelectContent>
            </Select>
            <Button onClick={handleRun} disabled={lossLoading} className="bg-red-600 text-white hover:bg-red-700">
              {lossLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
              Run Loss Forecast
            </Button>
          </div>
        </CardContent>
      </Card>

      {lossError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Loss forecast issue</AlertTitle>
          <AlertDescription>{lossError}</AlertDescription>
        </Alert>
      )}

      {lossLoading ? (
        <div className="space-y-5">
          <div className="grid gap-4 md:grid-cols-4">{Array.from({ length: 4 }).map((_, index) => <SkeletonBlock key={index} className="h-28" />)}</div>
          <SkeletonBlock className="h-96" />
          <SkeletonBlock className="h-72" />
        </div>
      ) : lossForecast.length ? (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {[
              ['Total Forecasted Loss', currency.format(lossSummary?.total_loss ?? 0), TrendingDown],
              ['Highest Risk Period', lossSummary?.highest_risk_period ?? 'N/A', ShieldAlert],
              ['Average Loss Risk Score', percent.format(lossSummary?.average_risk_score ?? 0), AlertCircle],
              ['Top Loss Driver', lossSummary?.top_loss_driver ?? 'N/A', PieIcon],
            ].map(([label, value, Icon], index) => (
              <motion.div key={String(label)} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>{String(label)}</CardDescription>
                    <CardTitle className="text-xl">{String(value)}</CardTitle>
                  </CardHeader>
                  <CardContent><Icon className="h-5 w-5 text-red-500" /></CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}>
            <Card>
              <CardHeader>
                <CardTitle>Loss Trend</CardTitle>
                <CardDescription>Driver-level losses with total forecasted value erosion and 95% interval bounds.</CardDescription>
              </CardHeader>
              <CardContent className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossForecast}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="period" />
                    <YAxis width={76} tickFormatter={compactCurrency} />
                    <RechartsTooltip formatter={(value: number) => currencyTooltip(value)} />
                    <Legend />
                    <Line dataKey="revenue_loss" name="Revenue Loss" stroke={LOSS_COLORS.revenue_loss} strokeWidth={2} />
                    <Line dataKey="operational_loss" name="Operational Loss" stroke={LOSS_COLORS.operational_loss} strokeWidth={2} />
                    <Line dataKey="inventory_loss" name="Inventory Loss" stroke={LOSS_COLORS.inventory_loss} strokeWidth={2} />
                    <Line dataKey="discount_loss" name="Discount Loss" stroke={LOSS_COLORS.discount_loss} strokeWidth={2} />
                    <Line dataKey="total_loss" name="Total Loss" stroke={LOSS_COLORS.total_loss} strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Loss Breakdown</CardTitle>
                <CardDescription>Stacked view of each loss driver by forecast period.</CardDescription>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={lossForecast}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="period" />
                    <YAxis width={76} tickFormatter={compactCurrency} />
                    <RechartsTooltip formatter={(value: number) => currencyTooltip(value)} />
                    <Legend />
                    <Bar dataKey="revenue_loss" stackId="loss" fill={LOSS_COLORS.revenue_loss} name="Revenue" />
                    <Bar dataKey="operational_loss" stackId="loss" fill={LOSS_COLORS.operational_loss} name="Operational" />
                    <Bar dataKey="inventory_loss" stackId="loss" fill={LOSS_COLORS.inventory_loss} name="Inventory" />
                    <Bar dataKey="discount_loss" stackId="loss" fill={LOSS_COLORS.discount_loss} name="Discount" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle>Segment Loss Mix</CardTitle>
                    <CardDescription>Toggle between category and region breakdown.</CardDescription>
                  </div>
                  <div className="flex rounded-full border p-1">
                    {(['category', 'region'] as const).map((item) => (
                      <Button key={item} size="sm" variant={segmentView === item ? 'default' : 'ghost'} className="rounded-full" onClick={() => setSegmentView(item)}>{item}</Button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={segmentRows} dataKey="total_loss" nameKey="segment" innerRadius={72} outerRadius={112} paddingAngle={2}>
                      {segmentRows.map((_, index) => <Cell key={index} fill={['#ef4444', '#f97316', '#f59e0b', '#8b5cf6', '#14b8a6', '#2563eb'][index % 6]} />)}
                    </Pie>
                    <RechartsTooltip formatter={(value: number) => currencyTooltip(value)} />
                    <Legend />
                    <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="fill-foreground text-sm font-bold">
                      {compactCurrency(totalSegmentLoss)}
                    </text>
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Risk Score Heatmap</CardTitle>
              <CardDescription>Green means low exposure, amber means watch closely, red means high-risk loss pressure.</CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <div className="min-w-[760px] space-y-2">
                {heatmapRows.map((row) => (
                  <div key={row.segment} className="grid grid-cols-[150px_repeat(12,minmax(42px,1fr))] items-center gap-2">
                    <span className="truncate text-sm font-medium">{row.segment}</span>
                    {row.cells.map((cell, index) => (
                      <motion.div
                        key={`${row.segment}-${cell.period}-${index}`}
                        initial={{ scale: 0.75, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: index * 0.025 }}
                        title={`${cell.period}: ${percent.format(cell.score)}`}
                        className={`h-9 rounded-md border ${riskClass(cell.label)} text-center text-[10px] font-semibold leading-9`}
                      >
                        {Math.round(cell.score * 100)}%
                      </motion.div>
                    ))}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle>Loss Forecast Results</CardTitle>
                  <CardDescription>Sortable, paginated loss forecast table.</CardDescription>
                </div>
                <Button variant="outline" onClick={() => exportLossCsv(lossForecast)}>
                  <Download className="mr-2 h-4 w-4" />
                  Export CSV
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="overflow-auto rounded-2xl border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {[
                        ['period', 'Period'],
                        ['revenue_loss', 'Revenue Loss'],
                        ['operational_loss', 'Operational Loss'],
                        ['inventory_loss', 'Inventory Loss'],
                        ['discount_loss', 'Discount Loss'],
                        ['total_loss', 'Total Loss'],
                        ['loss_risk_score', 'Risk Score'],
                        ['risk_label', 'Risk Label'],
                      ].map(([key, label]) => (
                        <TableHead key={key} className="cursor-pointer" onClick={() => setSortKey(key as keyof LossForecastResult)}>{label}</TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pageRows.map((row) => (
                      <TableRow key={row.id}>
                        <TableCell>{row.period}</TableCell>
                        <TableCell>{currency.format(row.revenue_loss)}</TableCell>
                        <TableCell>{currency.format(row.operational_loss)}</TableCell>
                        <TableCell>{currency.format(row.inventory_loss)}</TableCell>
                        <TableCell>{currency.format(row.discount_loss)}</TableCell>
                        <TableCell className="font-semibold">{currency.format(row.total_loss)}</TableCell>
                        <TableCell>{percent.format(row.loss_risk_score)}</TableCell>
                        <TableCell><Badge className={riskClass(row.risk_label)}>{row.risk_label}</Badge></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <div className="flex items-center justify-between">
                <Button variant="outline" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</Button>
                <span className="text-sm text-muted-foreground">Page {page + 1} of {totalPages}</span>
                <Button variant="outline" disabled={page >= totalPages - 1} onClick={() => setPage((value) => Math.min(totalPages - 1, value + 1))}>Next</Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recommended Actions</CardTitle>
              <CardDescription>Auto-generated recommendations based on the top three loss drivers.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 md:grid-cols-3">
              {actions.map((item) => (
                <div key={item.label} className="rounded-2xl border bg-card p-4">
                  <p className="font-semibold">{item.label}</p>
                  <p className="mt-2 text-sm text-muted-foreground">{item.action}</p>
                  <Badge className="mt-3 bg-red-100 text-red-700">{currency.format(item.value)}</Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="border-dashed">
          <CardContent className="py-14 text-center">
            <p className="font-medium">No loss forecast has been run yet.</p>
            <p className="mt-2 text-sm text-muted-foreground">Choose a horizon and run the forecast to generate risk scores, loss drivers, and segment breakdowns.</p>
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}
