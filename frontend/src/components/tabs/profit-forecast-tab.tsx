'use client';

import React, { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, ArrowRight, Landmark, Loader2, Lock, Scale, TrendingUp, Wallet, Zap } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAppStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { useCurrencyFormat } from '@/lib/use-currency-format';
import { getApiErrorMessage } from '@/lib/api';
import type { ProfitForecastResult, ProfitScenario } from '@/types/forecast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

const SCENARIOS: ProfitScenario[] = ['optimistic', 'baseline', 'pessimistic'];
function sum(rows: ProfitForecastResult[], key: keyof ProfitForecastResult) {
  return rows.reduce((total, row) => total + Number(row[key] ?? 0), 0);
}

function average(rows: ProfitForecastResult[], key: keyof ProfitForecastResult) {
  return rows.length ? sum(rows, key) / rows.length : 0;
}

function scenarioColor(scenario: ProfitScenario) {
  if (scenario === 'optimistic') return '#10b981';
  if (scenario === 'pessimistic') return '#f43f5e';
  return '#2563eb';
}

function deltaText(value: number, baseline: number, formatNumber: (value: number | string | null | undefined, digits?: number) => string) {
  if (!baseline) return null;
  const delta = ((value - baseline) / Math.abs(baseline)) * 100;
  return `${delta >= 0 ? '+' : ''}${formatNumber(delta, 1)}%`;
}

function deltaBadgeClass(delta: string | null) {
  if (!delta) return '';
  return delta.startsWith('-') ? 'border-red-200 bg-red-50 text-red-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700';
}

function SkeletonBlock({ className = 'h-72' }: { className?: string }) {
  return <div className={`animate-pulse rounded-2xl border bg-muted/40 ${className}`} />;
}

function waterfallData(row: ProfitForecastResult | null) {
  if (!row) return [];
  return [
    { name: 'Revenue', value: row.forecasted_revenue, fill: '#10b981' },
    { name: 'COGS', value: -row.forecasted_cogs, fill: '#f43f5e' },
    { name: 'Gross Profit', value: row.gross_profit, fill: row.gross_profit >= 0 ? '#059669' : '#e11d48' },
    { name: 'OpEx', value: -row.operating_expenses, fill: '#fb7185' },
    { name: 'Losses', value: -row.total_losses, fill: '#ef4444' },
    { name: 'Net Profit', value: row.net_profit, fill: row.net_profit >= 0 ? '#047857' : '#be123c' },
  ];
}

export default function ProfitForecastTab() {
  const { toast } = useToast();
  const {
    formatCurrency,
    formatAxisCurrency,
    formatCurrencyTooltip,
    formatNumber,
    formatPercentValue,
  } = useCurrencyFormat();
  const percentAxis = (value: number | string) => formatPercentValue(value);
  const datasetId = useAppStore((state) => state.datasetId);
  const columns = useAppStore((state) => state.columns);
  const timeSeriesForecastResult = useAppStore((state) => state.timeSeriesForecastResult);
  const lossForecast = useAppStore((state) => state.lossForecast);
  const scenarios = useAppStore((state) => state.scenarios);
  const profitLoading = useAppStore((state) => state.profitLoading);
  const profitError = useAppStore((state) => state.profitError);
  const profitIllustrativeAssumptions = useAppStore((state) => state.profitIllustrativeAssumptions);
  const profitValidationReport = useAppStore((state) => state.profitValidationReport);
  const breakevenPeriod = useAppStore((state) => state.breakevenPeriod);
  const periodsToBreakeven = useAppStore((state) => state.periodsToBreakeven);
  const runProfitForecast = useAppStore((state) => state.runProfitForecast);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const [periods, setPeriods] = useState(30);
  const [scenario, setScenario] = useState<ProfitScenario>('baseline');
  const [confirmedAssumptions, setConfirmedAssumptions] = useState(false);
  const [assumptionDialogOpen, setAssumptionDialogOpen] = useState(false);
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({});
  const [scenarioParameters, setScenarioParameters] = useState<Record<ProfitScenario, Record<'revenue' | 'cogs' | 'loss', number>>>({
    optimistic: { revenue: 1.15, cogs: 0.97, loss: 0.8 },
    baseline: { revenue: 1, cogs: 1, loss: 1 },
    pessimistic: { revenue: 0.85, cogs: 1.05, loss: 1.2 },
  });

  const locked = !datasetId || !lossForecast?.length;
  const activeRows = scenarios?.[scenario] ?? [];
  const baselineRows = scenarios?.baseline ?? [];
  const firstActive = activeRows[0] ?? null;
  const availablePeriods = lossForecast?.length ?? 0;
  const selectedPeriods = Math.max(1, Math.min(periods, availablePeriods || 1));
  const periodLabel = timeSeriesForecastResult?.period_label || 'period';
  const periodOptions = [...new Set([1, 3, 6, 12, availablePeriods].filter((value) => value > 0 && value <= availablePeriods))]
    .map((value) => ({ value, label: `${value} ${periodLabel}${value === 1 ? '' : 's'}` }));
  const effectiveColumnMapping = {
    date: timeSeriesForecastResult?.date_column,
    revenue: timeSeriesForecastResult?.target_column,
    ...Object.fromEntries(Object.entries(columnMapping).filter(([, value]) => value && value !== '__auto__')),
  } as Record<string, string>;

  const summary = useMemo(() => {
    const selected = {
      revenue: sum(activeRows, 'forecasted_revenue'),
      grossProfit: sum(activeRows, 'gross_profit'),
      netProfit: sum(activeRows, 'net_profit'),
      grossMargin: average(activeRows, 'gross_margin_pct'),
      netMargin: average(activeRows, 'net_margin_pct'),
    };
    const baseline = {
      revenue: sum(baselineRows, 'forecasted_revenue'),
      grossProfit: sum(baselineRows, 'gross_profit'),
      netProfit: sum(baselineRows, 'net_profit'),
      grossMargin: average(baselineRows, 'gross_margin_pct'),
      netMargin: average(baselineRows, 'net_margin_pct'),
    };
    return { selected, baseline };
  }, [activeRows, baselineRows]);

  const combinedLineData = useMemo(() => {
    const periodsSet = new Set<string>();
    SCENARIOS.forEach((item) => scenarios?.[item]?.forEach((row) => periodsSet.add(row.period)));
    return Array.from(periodsSet).sort().map((period) => {
      const row: Record<string, string | number | null> = { period };
      SCENARIOS.forEach((item) => {
        row[item] = scenarios?.[item]?.find((candidate) => candidate.period === period)?.net_profit ?? null;
      });
      return row;
    });
  }, [scenarios]);

  const groupedBarData = activeRows.map((row) => ({
    period: row.period,
    revenue: row.forecasted_revenue,
    costs: row.forecasted_cogs + row.operating_expenses,
    losses: row.total_losses,
  }));

  const comparisonRows = SCENARIOS.map((item) => {
    const rows = scenarios?.[item] ?? [];
    return {
      scenario: item,
      revenue: sum(rows, 'forecasted_revenue'),
      cogs: sum(rows, 'forecasted_cogs'),
      grossProfit: sum(rows, 'gross_profit'),
      losses: sum(rows, 'total_losses'),
      netProfit: sum(rows, 'net_profit'),
      netMargin: average(rows, 'net_margin_pct'),
    };
  });

  const kpiCards: { label: string; value: string; delta: string | null; Icon: LucideIcon }[] = [
    { label: 'Forecasted Revenue', value: formatCurrency(summary.selected.revenue), delta: scenario === 'baseline' ? null : deltaText(summary.selected.revenue, summary.baseline.revenue, formatNumber), Icon: Wallet },
    { label: 'Gross Profit', value: formatCurrency(summary.selected.grossProfit), delta: scenario === 'baseline' ? null : deltaText(summary.selected.grossProfit, summary.baseline.grossProfit, formatNumber), Icon: TrendingUp },
    { label: 'Net Profit', value: formatCurrency(summary.selected.netProfit), delta: scenario === 'baseline' ? null : deltaText(summary.selected.netProfit, summary.baseline.netProfit, formatNumber), Icon: Landmark },
    { label: 'Gross Margin %', value: formatPercentValue(summary.selected.grossMargin), delta: scenario === 'baseline' ? null : deltaText(summary.selected.grossMargin, summary.baseline.grossMargin, formatNumber), Icon: Scale },
    { label: 'Net Margin %', value: formatPercentValue(summary.selected.netMargin), delta: scenario === 'baseline' ? null : deltaText(summary.selected.netMargin, summary.baseline.netMargin, formatNumber), Icon: Scale },
  ];

  const handleRun = async (confirmedOverride?: boolean) => {
    if (!datasetId) return;
    const assumptionsConfirmed = confirmedOverride ?? confirmedAssumptions;
    if (!assumptionsConfirmed) {
      setAssumptionDialogOpen(true);
      return;
    }
    try {
      setConfirmedAssumptions(true);
      await runProfitForecast(datasetId, selectedPeriods, {
        confirmedAssumptions: true,
        columnMapping: effectiveColumnMapping,
        scenarioParameters,
      });
      toast({ title: 'Profit forecast ready', description: 'Generated optimistic, baseline, and pessimistic P&L scenarios.' });
    } catch (error) {
      toast({ title: 'Profit forecast failed', description: getApiErrorMessage(error, 'Unable to run profit forecast.'), variant: 'destructive' });
    }
  };

  if (locked) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
          <Lock className="h-10 w-10 text-muted-foreground/60" />
          <div>
            <h2 className="text-xl font-bold">Run Loss Forecast first</h2>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              Profit Forecast uses total losses from the Loss Forecast workflow, plus upstream revenue and cost projections, to build scenario-based P&L views.
            </p>
          </div>
          <Button onClick={() => setActiveTab('loss_forecast')}>
            Open Loss Forecast
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <Card className="overflow-hidden border-emerald-200/70 bg-gradient-to-br from-emerald-50 via-background to-blue-50">
        <CardContent className="flex flex-col gap-5 p-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Badge className="border-emerald-200 bg-emerald-100 text-emerald-700">Financial Projection</Badge>
            <h2 className="mt-3 text-2xl font-bold tracking-tight">Profit Forecast</h2>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">Project your net financial performance across scenarios.</p>
          </div>
          <div className="flex flex-col gap-3 lg:items-end">
            <div className="flex rounded-full border bg-background p-1">
              {SCENARIOS.map((item) => (
                <Button
                  key={item}
                  size="sm"
                  variant={scenario === item ? 'default' : 'ghost'}
                  className="rounded-full capitalize"
                  style={scenario === item ? { backgroundColor: scenarioColor(item), color: 'white' } : undefined}
                  onClick={() => setScenario(item)}
                >
                  {item}
                </Button>
              ))}
            </div>
            <div className="flex gap-3">
              <Select value={String(selectedPeriods)} onValueChange={(value) => setPeriods(Number(value))}>
                <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
                <SelectContent>{periodOptions.map((item) => <SelectItem key={item.value} value={String(item.value)}>{item.label}</SelectItem>)}</SelectContent>
              </Select>
              <Button onClick={() => handleRun()} disabled={profitLoading} className="bg-emerald-600 text-white hover:bg-emerald-700">
                {profitLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                Run Profit Forecast
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Alert>
        <Scale className="h-4 w-4" />
        <AlertTitle>Scenario Confirmation Required</AlertTitle>
        <AlertDescription className="space-y-3">
          <p>Profit forecasting uses confirmed scenario multipliers, mapped revenue/cost fields, loss forecast output, and an audited P&L calculation trail.</p>
          <label className="flex items-center gap-2 text-sm font-medium">
            <Checkbox checked={confirmedAssumptions} onCheckedChange={(checked) => setConfirmedAssumptions(Boolean(checked))} />
            I confirm the scenario parameters and calculation assumptions for this run.
          </label>
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Financial column mapping</CardTitle>
          <CardDescription>Date and revenue use the completed time-series forecast. Map your accounting columns to avoid fallback cost assumptions.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          {[
            ['cogs', 'COGS / direct cost'],
            ['operating_expense', 'Operating expense'],
          ].map(([key, label]) => (
            <label key={key} className="space-y-1 text-sm font-medium">
              <span>{label}</span>
              <Select value={columnMapping[key] || '__auto__'} onValueChange={(value) => setColumnMapping((current) => ({ ...current, [key]: value }))}>
                <SelectTrigger><SelectValue placeholder="Auto-detect" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__auto__">Auto-detect / unavailable</SelectItem>
                  {columns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </label>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Scenario assumptions</CardTitle>
          <CardDescription>Multipliers apply to the selected forecast. Enter 1.00 for no change; permitted values are 0.25 to 3.00.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 lg:grid-cols-3">
          {SCENARIOS.map((scenarioName) => (
            <div key={scenarioName} className="rounded-lg border p-3">
              <p className="mb-3 font-semibold capitalize" style={{ color: scenarioColor(scenarioName) }}>{scenarioName}</p>
              <div className="grid grid-cols-3 gap-2">
                {(['revenue', 'cogs', 'loss'] as const).map((key) => (
                  <label key={key} className="text-xs font-medium capitalize">
                    {key}
                    <Input
                      className="mt-1 h-8"
                      type="number"
                      min="0.25"
                      max="3"
                      step="0.01"
                      value={scenarioParameters[scenarioName][key]}
                      onChange={(event) => setScenarioParameters((current) => ({
                        ...current,
                        [scenarioName]: { ...current[scenarioName], [key]: Number(event.target.value) },
                      }))}
                    />
                  </label>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <AlertDialog open={assumptionDialogOpen} onOpenChange={setAssumptionDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Confirm Profit Forecast Assumptions</AlertDialogTitle>
            <AlertDialogDescription>
              This run will use the selected scenario parameters, mapped revenue/cost fields, loss forecast output, and audited P&L calculations.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Review Again</AlertDialogCancel>
            <AlertDialogAction onClick={() => handleRun(true)} className="bg-emerald-600 text-white hover:bg-emerald-700">
              Confirm And Run
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {profitIllustrativeAssumptions && activeRows.length > 0 ? (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Illustrative assumptions</AlertTitle>
          <AlertDescription>
            {profitValidationReport?.validation_reasons?.join(' ') || 'Profit scenarios include documented fallback assumptions. Confirm mapped accounting fields before using these results for decisions.'}
          </AlertDescription>
        </Alert>
      ) : null}

      {profitError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Profit forecast issue</AlertTitle>
          <AlertDescription>{profitError}</AlertDescription>
        </Alert>
      )}

      {profitLoading ? (
        <div className="space-y-5">
          <div className="grid gap-4 md:grid-cols-5">{Array.from({ length: 5 }).map((_, index) => <SkeletonBlock key={index} className="h-28" />)}</div>
          <SkeletonBlock className="h-96" />
        </div>
      ) : activeRows.length ? (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            {kpiCards.map(({ label, value, delta, Icon }, index) => (
              <motion.div key={label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>{label}</CardDescription>
                    <CardTitle className="text-lg">{value}</CardTitle>
                  </CardHeader>
                  <CardContent className="flex items-center justify-between">
                    {delta ? <Badge variant="outline" className={deltaBadgeClass(delta)}>{delta} vs baseline</Badge> : <span />}
                    <Icon className="h-5 w-5 text-emerald-600" />
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          <Alert>
            <Scale className="h-4 w-4" />
            <AlertTitle>{profitValidationReport?.validation_status === 'ready' ? 'Financial readiness checks passed' : 'Review inputs before using for decisions'}</AlertTitle>
            <AlertDescription>
              {[profitValidationReport?.reconciliation || 'Each row reconciles revenue, costs, losses, and net profit.', ...(profitValidationReport?.validation_reasons ?? [])].join(' ')}
            </AlertDescription>
          </Alert>

          <AnimatePresence mode="wait">
            <motion.div key={scenario} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-6">
              <Card className="border-primary/20">
                <CardHeader>
                  <CardTitle>P&L Waterfall</CardTitle>
                  <CardDescription>Primary view of how revenue becomes net profit after COGS, OpEx, and losses.</CardDescription>
                </CardHeader>
                <CardContent className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={waterfallData(firstActive)}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" />
                      <YAxis width={76} tickFormatter={formatAxisCurrency} />
                      <RechartsTooltip formatter={(value: number) => formatCurrencyTooltip(value)} />
                      <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                        {waterfallData(firstActive).map((row) => <Cell key={row.name} fill={row.fill} />)}
                      </Bar>
                      <ReferenceLine y={0} stroke="#0f172a" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <div className="grid gap-6 xl:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Net Profit Forecast</CardTitle>
                    <CardDescription>Three scenario lines with a break-even zero marker.</CardDescription>
                  </CardHeader>
                  <CardContent className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={combinedLineData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="period" />
                        <YAxis width={76} tickFormatter={formatAxisCurrency} />
                        <RechartsTooltip formatter={(value: number) => formatCurrencyTooltip(value)} />
                        <Legend />
                        <ReferenceLine y={0} stroke="#64748b" label="Break-even" />
                        <Line connectNulls dataKey="optimistic" stroke="#10b981" strokeWidth={3} />
                        <Line connectNulls dataKey="baseline" stroke="#2563eb" strokeWidth={3} />
                        <Line connectNulls dataKey="pessimistic" stroke="#f43f5e" strokeWidth={3} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Gross vs Net Margin Trend</CardTitle>
                    <CardDescription>Margin percentage movement for the selected scenario.</CardDescription>
                  </CardHeader>
                  <CardContent className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={activeRows}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="period" />
                        <YAxis width={56} tickFormatter={percentAxis} />
                        <RechartsTooltip formatter={(value: number) => percentAxis(value)} />
                        <Legend />
                        <Area dataKey="gross_margin_pct" name="Gross Margin %" stroke="#2563eb" fill="#bfdbfe" />
                        <Area dataKey="net_margin_pct" name="Net Margin %" stroke="#10b981" fill="#bbf7d0" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
                <Card>
                  <CardHeader>
                    <CardTitle>Revenue vs Cost vs Loss</CardTitle>
                    <CardDescription>Cost and loss pressure against projected revenue.</CardDescription>
                  </CardHeader>
                  <CardContent className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={groupedBarData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="period" />
                        <YAxis width={76} tickFormatter={formatAxisCurrency} />
                        <RechartsTooltip formatter={(value: number) => formatCurrencyTooltip(value)} />
                        <Legend />
                        <Bar dataKey="revenue" name="Forecasted Revenue" fill="#2563eb" />
                        <Bar dataKey="costs" name="Total Costs" fill="#f97316" />
                        <Bar dataKey="losses" name="Total Losses" fill="#f43f5e" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card className={periodsToBreakeven && periodsToBreakeven <= 60 ? 'border-emerald-200 bg-emerald-50' : periodsToBreakeven && periodsToBreakeven <= 90 ? 'border-amber-200 bg-amber-50' : 'border-rose-200 bg-rose-50'}>
                  <CardHeader>
                    <CardTitle>Break-even Analysis</CardTitle>
                    <CardDescription>Baseline cumulative net profit timeline to break-even.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-black">{breakevenPeriod ?? 'Not reached'}</p>
                    <p className="mt-2 text-sm text-muted-foreground">{periodsToBreakeven ? `${periodsToBreakeven} period(s) to break even` : 'No period reaches break-even in the current horizon.'}</p>
                    <div className="mt-6 h-3 overflow-hidden rounded-full bg-white/80">
                      <div className="h-full rounded-full bg-emerald-600" style={{ width: `${Math.min(100, ((periodsToBreakeven ?? periods) / Math.max(periods, 1)) * 100)}%` }} />
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>P&L Summary Table</CardTitle>
                  <CardDescription>Period-by-period financial projection for the selected scenario.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="max-h-[440px] overflow-auto rounded-2xl border">
                    <Table>
                      <TableHeader className="sticky top-0 bg-background">
                        <TableRow>
                          <TableHead>Period</TableHead>
                          <TableHead>Revenue</TableHead>
                          <TableHead>COGS</TableHead>
                          <TableHead>Gross Profit</TableHead>
                          <TableHead>OpEx</TableHead>
                          <TableHead>Total Loss</TableHead>
                          <TableHead>Net Profit</TableHead>
                          <TableHead>Gross Margin %</TableHead>
                          <TableHead>Net Margin %</TableHead>
                          <TableHead>Scenario</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {activeRows.map((row) => (
                          <TableRow key={row.id} className={row.net_profit >= 0 ? 'bg-emerald-50/60' : 'bg-rose-50/60'}>
                            <TableCell>{row.period}</TableCell>
                            <TableCell>{formatCurrency(row.forecasted_revenue)}</TableCell>
                            <TableCell>{formatCurrency(row.forecasted_cogs)}</TableCell>
                            <TableCell>{formatCurrency(row.gross_profit)}</TableCell>
                            <TableCell>{formatCurrency(row.operating_expenses)}</TableCell>
                            <TableCell>{formatCurrency(row.total_losses)}</TableCell>
                            <TableCell className={row.net_profit >= 0 ? 'font-bold text-emerald-600' : 'font-bold text-rose-600'}>{formatCurrency(row.net_profit)}</TableCell>
                            <TableCell>{formatNumber(row.gross_margin_pct)}%</TableCell>
                            <TableCell>{formatNumber(row.net_margin_pct)}%</TableCell>
                            <TableCell><Badge className="capitalize">{row.scenario}</Badge></TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Scenario Comparison</CardTitle>
                  <CardDescription>Side-by-side total P&L summary for all three scenarios.</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 lg:grid-cols-3">
                  {comparisonRows.map((row) => (
                    <div key={row.scenario} className={`rounded-2xl border p-4 ${row.scenario === 'baseline' ? 'border-blue-300 bg-blue-50' : 'bg-card'}`}>
                      <p className="text-lg font-bold capitalize" style={{ color: scenarioColor(row.scenario) }}>{row.scenario}</p>
                      <div className="mt-4 space-y-2 text-sm">
                        <p className="flex justify-between"><span>Total Revenue</span><b>{formatCurrency(row.revenue)}</b></p>
                        <p className="flex justify-between"><span>Total COGS</span><b>{formatCurrency(row.cogs)}</b></p>
                        <p className="flex justify-between"><span>Gross Profit</span><b>{formatCurrency(row.grossProfit)}</b></p>
                        <p className="flex justify-between"><span>Total Losses</span><b>{formatCurrency(row.losses)}</b></p>
                        <p className="flex justify-between"><span>Net Profit</span><b>{formatCurrency(row.netProfit)}</b></p>
                        <p className="flex justify-between"><span>Net Margin</span><b>{formatNumber(row.netMargin)}%</b></p>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </AnimatePresence>
        </>
      ) : (
        <Card className="border-dashed">
          <CardContent className="py-14 text-center">
            <p className="font-medium">No profit forecast has been run yet.</p>
            <p className="mt-2 text-sm text-muted-foreground">Run the forecast to generate three P&L scenarios and break-even analysis.</p>
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}
