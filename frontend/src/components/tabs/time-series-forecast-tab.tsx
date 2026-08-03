'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAppStore, type ColumnInfo, type DataRow, type TimeSeriesForecastResult, type TsForecastModelComparison, type TsFutureForecast, type TsInsight, type TsStationarity } from '@/lib/store';
import { toast as showToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertCircle, ArrowRight, CalendarDays, CheckCircle2, ChevronLeft, Loader2, ShieldCheck, TrendingUp, Waves, Zap, RadioTower, Info, AlertTriangle } from 'lucide-react';
import { Area, ComposedChart, CartesianGrid, Legend, Line, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';

import { pickPreferredSalesDateColumn, pickSmartSalesTargetColumn, isDatePartColumn, isForecastDateColumnCandidate } from '@/lib/sales-domain';

const STEP_ITEMS = [
  { step: 1, label: 'Data Config', icon: CalendarDays },
  { step: 2, label: 'TS Models', icon: RadioTower },
  { step: 3, label: 'Forecast', icon: TrendingUp },
];

const TS_CHART_COLORS = {
  actual: '#2563eb',
  backtest: '#f59e0b',
  forecast: '#8b5cf6',
  band: '#22c55e',
  bandBase: '#ecfccb',
  grid: '#cbd5e1',
} as const;

const transition = { duration: 0.25, ease: 'easeOut' } as const;
const HORIZON_OPTIONS = [3, 6, 12, 24] as const;

const MODEL_DESCRIPTIONS: Record<string, string> = {
  SARIMA: 'Auto-selects ARIMA order via pmdarima. Best for stationary series.',
  Prophet: 'Handles trend breaks and missing data. Best for non-stationary series.',
  HoltWinters: 'Exponential smoothing with damped trend. Best for stable seasonal patterns.',
};

const MODEL_STRENGTHS: Record<string, string[]> = {
  SARIMA: ['Seasonal', 'Auto-order', 'AIC'],
  Prophet: ['Trend breaks', 'Robust', 'Intervals'],
  HoltWinters: ['Fast', 'Stable', 'Damped trend'],
};

function formatIndianNumber(num: number | null | undefined) {
  if (num === null || num === undefined || Number.isNaN(num)) return 'N/A';
  return new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(num);
}

function formatChartValue(value: number | null | undefined) {
  return value == null || Number.isNaN(value) ? 'N/A' : value.toLocaleString();
}

function modelStatusClass(status: string) {
  if (status === 'completed') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  if (status === 'failed') return 'border-red-200 bg-red-50 text-red-700';
  return 'border-amber-200 bg-amber-50 text-amber-700';
}

function ForecastTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ payload?: { actual?: number | null; backtest?: number | null; forecast?: number | null; lower?: number | null; upper?: number | null } }>; label?: string }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  const rows = [
    { label: 'Actual', value: point.actual },
    { label: 'Backtest', value: point.backtest },
    { label: 'Forecast', value: point.forecast },
    { label: 'Lower 95%', value: point.lower },
    { label: 'Upper 95%', value: point.upper },
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

function getSmartTargetColumn(columns: ColumnInfo[], data: DataRow[]) {
  return pickSmartSalesTargetColumn(columns, data as Array<Record<string, unknown>>);
}

function getPreferredDateColumn(columns: ColumnInfo[]) {
  return pickPreferredSalesDateColumn(columns);
}

function inferSeriesProfile(data: DataRow[], dateColumn: string, targetColumn: string) {
  const points = data
    .map((row) => ({ date: new Date(String(row[dateColumn] ?? '')), value: Number(row[targetColumn]) }))
    .filter((item) => !Number.isNaN(item.date.getTime()) && Number.isFinite(item.value))
    .sort((left, right) => left.date.getTime() - right.date.getTime());
  if (points.length < 2) {
    return { detected_frequency: 'period', usable_periods: points.length, volatility: 0, zero_value_share: 0 };
  }
  const values = points.map((item) => item.value);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const diffs = points.slice(1).map((item, index) => (item.date.getTime() - points[index].date.getTime()) / 86400000).sort((a, b) => a - b);
  const medianDays = diffs[Math.floor(diffs.length / 2)] ?? 30;
  return {
    detected_frequency: medianDays <= 2 ? 'day' : medianDays <= 10 ? 'week' : medianDays <= 45 ? 'month' : medianDays <= 120 ? 'quarter' : 'year',
    usable_periods: points.length,
    volatility: mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean),
    zero_value_share: values.filter((value) => value === 0).length / values.length,
  };
}

export default function TimeSeriesForecastTab() {
  const rawData = useAppStore((state) => state.rawData);
  const cleanedData = useAppStore((state) => state.cleanedData);
  const columns = useAppStore((state) => state.columns);
  const datasetId = useAppStore((state) => state.datasetId);
  const modelTrained = useAppStore((state) => state.modelTrained);
  const storedResult = useAppStore((state) => state.timeSeriesForecastResult);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const data = cleanedData ?? rawData ?? [];

  const numericColumns = useMemo(() => columns.filter((column) => column.role === 'numeric'), [columns]);
  const dateColumns = useMemo(() => columns.filter(isForecastDateColumnCandidate), [columns]);
  const preferredDateColumn = useMemo(() => getPreferredDateColumn(columns), [columns]);
  const smartTargetColumn = useMemo(() => getSmartTargetColumn(columns, data as DataRow[]), [columns, data]);

  const [currentStep, setCurrentStep] = useState(1);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(3);
  const [trainSplitPercent, setTrainSplitPercent] = useState(80);
  const [result, setResult] = useState<TimeSeriesForecastResult | null>(storedResult);
  const [isTraining, setIsTraining] = useState(false);

  // New multi-model state
  const [stationarity, setStationarity] = useState<TsStationarity | null>(null);
  const [stationarityError, setStationarityError] = useState<string | null>(null);
  const [modelComparison, setModelComparison] = useState<TsForecastModelComparison[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [forecastResults, setForecastResults] = useState<TsFutureForecast[]>([]);
  const [selectionReason, setSelectionReason] = useState('');
  const [insight, setInsight] = useState<TsInsight | null>(null);
  const [stationarityLoading, setStationarityLoading] = useState(false);
  const stationarityAttemptKeyRef = useRef<string | null>(null);
  const stationarityInFlightRef = useRef(false);

  useEffect(() => {
    if ((!dateColumn || isDatePartColumn(dateColumn)) && preferredDateColumn) {
      setDateColumn(preferredDateColumn);
    }
    if (!targetColumn && smartTargetColumn) setTargetColumn(smartTargetColumn);
  }, [columns, dateColumn, targetColumn, preferredDateColumn, smartTargetColumn]);

  useEffect(() => {
    if (storedResult) {
      setResult(storedResult);
      setCurrentStep(3);
    }
  }, [storedResult]);

  // Auto-load stationarity when step 2 opens (one attempt per dataset/column key; no retry loop).
  useEffect(() => {
    if (currentStep !== 2 || !datasetId || !dateColumn || !targetColumn) return;

    const attemptKey = `${datasetId}|${dateColumn}|${targetColumn}`;
    if (stationarityAttemptKeyRef.current === attemptKey) return;
    if (stationarityInFlightRef.current) return;

    let cancelled = false;
    let settled = false;
    stationarityAttemptKeyRef.current = attemptKey;
    stationarityInFlightRef.current = true;
    setStationarity(null);
    setStationarityLoading(true);
    setStationarityError(null);

    apiClient.post('/ts-forecast/stationarity', {
      dataset_id: datasetId,
      date_column: dateColumn || undefined,
      target_column: targetColumn || undefined,
    })
      .then((res) => {
        settled = true;
        if (cancelled || stationarityAttemptKeyRef.current !== attemptKey) return;
        setStationarity(res.data as TsStationarity);
        setStationarityError(null);
      })
      .catch((err) => {
        settled = true;
        if (cancelled || stationarityAttemptKeyRef.current !== attemptKey) return;
        console.error('Stationarity fetch failed:', err);
        const status = axios.isAxiosError(err) ? err.response?.status : undefined;
        const baseMessage = getApiErrorMessage(err, 'Could not load stationarity.');
        const description = status === 404
          ? `${status} on /api/ts-forecast/stationarity — endpoint missing on the running backend. Rebuild/redeploy the backend container so it matches current source, then re-upload the dataset.`
          : status
            ? `${status}: ${baseMessage}`
            : baseMessage;
        setStationarityError(description);
        showToast({ title: 'Stationarity check failed', description, variant: 'destructive' });
      })
      .finally(() => {
        if (cancelled) return;
        stationarityInFlightRef.current = false;
        setStationarityLoading(false);
      });

    return () => {
      cancelled = true;
      stationarityInFlightRef.current = false;
      // Allow React Strict Mode remounts to retry an in-flight (unsettled) attempt.
      if (!settled && stationarityAttemptKeyRef.current === attemptKey) {
        stationarityAttemptKeyRef.current = null;
      }
    };
  }, [currentStep, datasetId, dateColumn, targetColumn]);

  const profile = useMemo(() => inferSeriesProfile(data as DataRow[], dateColumn, targetColumn), [data, dateColumn, targetColumn]);
  const periodGrainLabel = (label?: string | null) => {
    if (!label) return 'period';
    return ({ day: 'daily', week: 'weekly', month: 'monthly', quarter: 'quarterly', year: 'yearly' } as Record<string, string>)[label] ?? label;
  };
  const frequencyNote = stationarity?.frequency_auto_adjusted && stationarity.period_label
    ? `Using ${periodGrainLabel(stationarity.period_label)} aggregation${
        stationarity.inferred_period_label && stationarity.inferred_period_label !== stationarity.period_label
          ? ` (${periodGrainLabel(stationarity.inferred_period_label)} span was too short)`
          : ''
      }.`
    : null;

  const chartData = useMemo(() => {
    if (!result) return [];
    const testMap = new Map(result.test_forecast.map((item) => [item.period, item]));
    return [
      ...result.history.map((item) => {
        const testPoint = testMap.get(item.period);
        return {
          period: item.period,
          actual: item.actual,
          backtest: testPoint?.predicted ?? null,
          forecast: null as number | null,
          lower: testPoint?.lower ?? null,
          upper: testPoint?.upper ?? null,
          lowerBand: testPoint?.lower ?? null,
          confidenceRange: testPoint?.lower != null && testPoint?.upper != null ? Math.max(testPoint.upper - testPoint.lower, 0) : null,
        };
      }),
      ...result.future_forecast.map((item) => ({
        period: item.period,
        actual: null,
        backtest: null,
        forecast: item.predicted,
        lower: item.lower ?? null,
        upper: item.upper ?? null,
        lowerBand: item.lower ?? null,
        confidenceRange: item.lower != null && item.upper != null ? Math.max(item.upper - item.lower, 0) : null,
      })),
    ];
  }, [result]);

  // New multi-model chart data
  const multiChartData = useMemo(() => {
    if (!modelComparison.length || !selectedModel) return [];
    const data: Array<{ period: string; actual?: number | null; backtest?: number | null; forecast?: number | null; lower?: number | null; upper?: number | null; lowerBand?: number | null; confidenceRange?: number | null }> = [];
    forecastResults.forEach((item) => {
      data.push({
        period: item.period,
        actual: null,
        backtest: null,
        forecast: item.forecast,
        lower: item.lower,
        upper: item.upper,
        lowerBand: item.lower ?? null,
        confidenceRange: item.lower != null && item.upper != null ? Math.max(item.upper - item.lower, 0) : null,
      });
    });
    return data;
  }, [modelComparison, selectedModel, forecastResults]);

  const handleRun = async () => {
    if (!dateColumn || !targetColumn) {
      showToast({ title: 'Configuration incomplete', description: smartTargetColumn ? 'Choose both a date column and a sales target.' : 'No suitable numeric target was auto-detected. Please manually select the target column before running the forecast.', variant: 'destructive' });
      return;
    }
    setIsTraining(true);
    try {
      const payload = {
        dataset_id: datasetId ?? null,
        date_column: dateColumn,
        target_column: targetColumn,
        forecast_periods: forecastPeriods,
        training_split: trainSplitPercent / 100,
      };
      const response = await apiClient.post('/forecast/ts/run', payload);
      const nextResult = response.data as TimeSeriesForecastResult;
      setResult(nextResult);
      useAppStore.setState({ timeSeriesForecastResult: nextResult });
      showToast({ title: 'Time-series forecast ready', description: `Projected ${forecastPeriods} future ${nextResult.period_label ?? 'period'}${forecastPeriods === 1 ? '' : 's'}.` });
    } catch (error) {
      showToast({ title: 'Forecast failed', description: getApiErrorMessage(error, 'Time-series forecast failed.'), variant: 'destructive' });
    } finally {
      setIsTraining(false);
    }
  };

  const handleMultiModelRun = async () => {
    if (!dateColumn || !targetColumn) {
      showToast({ title: 'Configuration incomplete', description: 'Choose both a date column and a sales target.', variant: 'destructive' });
      return;
    }
    setIsTraining(true);
    try {
      const res = await apiClient.post('/ts-forecast/run', {
        dataset_id: datasetId ?? null,
        horizon: forecastPeriods,
        training_split: trainSplitPercent / 100,
        date_column: dateColumn,
        target_column: targetColumn,
      });
      const data = res.data as {
        status: string; best_model: string; smape: number; mae: number; rmse: number; mape: number | null;
        reason: string; stationarity: TsStationarity; future_forecast: TsFutureForecast[];
        insight: TsInsight; model_comparison: TsForecastModelComparison[];
      };
      if (data.status !== 'completed') {
        throw new Error('Training failed: ' + (data as any).error || 'Unknown error');
      }
      setModelComparison(data.model_comparison);
      setSelectedModel(data.best_model);
      setForecastResults(data.future_forecast);
      setSelectionReason(data.reason);
      setInsight(data.insight);
      setStationarity(data.stationarity);
      // Map to legacy format for chart display
      const mappedResult: TimeSeriesForecastResult = {
        date_column: dateColumn,
        target_column: targetColumn,
        frequency: data.stationarity.status,
        period_label: 'period',
        dataset_profile: { detected_frequency: profile.detected_frequency, usable_periods: profile.usable_periods, volatility: profile.volatility, zero_value_share: profile.zero_value_share },
        stationarity_check: { test_name: 'ADF-KPSS', p_value: data.stationarity.adf_pvalue, verdict: data.stationarity.status, note: data.stationarity.note },
        history: [],
        test_forecast: [],
        future_forecast: data.future_forecast.map((f) => ({ period: f.period, predicted: f.forecast, lower: f.lower, upper: f.upper })),
        metrics: { mae: data.mae, rmse: data.rmse, mape: data.mape ?? 0 },
        training_summary: { model_name: data.best_model, total_periods: 0, train_periods: 0, test_periods: 0, train_percentage: 0, test_percentage: 0, forecast_periods: forecastPeriods, train_start: '', train_end: '', test_start: '', test_end: '', last_observed_period: '' },
        analysis: data.insight.insight_text,
      };
      setResult(mappedResult);
      useAppStore.setState({ timeSeriesForecastResult: mappedResult });
      showToast({ title: 'Multi-model training complete', description: `${data.best_model} auto-selected with SMAPE ${data.smape}%.` });
    } catch (error) {
      showToast({ title: 'Multi-model training failed', description: getApiErrorMessage(error, 'All 3 models failed.'), variant: 'destructive' });
    } finally {
      setIsTraining(false);
    }
  };

  if (!data.length || !columns.length) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Sales Forecast Using Time Series Analysis</h2>
          <p className="mt-1 text-muted-foreground">This workflow reads the cleaned dataset from Step 3 and models time as the primary signal.</p>
        </div>
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <div>
              <p className="font-medium">Upload and clean a dataset first</p>
              <p className="mt-1 text-sm text-muted-foreground">Step 5 needs the cleaned cached dataset from Step 3 before it can run.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden border border-primary/20 bg-gradient-to-br from-primary/8 via-background to-secondary/70">
        <CardContent className="p-6">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-background/80 px-3 py-1 text-xs font-medium text-primary">
                <RadioTower className="h-3.5 w-3.5" />
                Forecast TS
              </div>
              <h2 className="mt-3 text-2xl font-bold tracking-tight">Sales Forecast Using Time Series Analysis</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Three models (SARIMA, Prophet, HoltWinters) are trained and compared. The best is auto-selected by lowest SMAPE.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge variant="secondary">{dateColumn || 'Pick a date column'}</Badge>
                <Badge variant="secondary">{targetColumn ? `Target: ${targetColumn}` : 'Pick a target column'}</Badge>
                <Badge variant="secondary">{forecastPeriods} future periods</Badge>
                <Badge variant="secondary">Auto-select best of 3</Badge>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-3 xl:w-[360px] xl:grid-cols-1">
              <div className="rounded-2xl border bg-white/80 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/85 dark:shadow-none">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Detected Frequency</p>
                <p className="mt-2 text-lg font-semibold capitalize">{profile.detected_frequency}</p>
              </div>
              <div className="rounded-2xl border bg-white/80 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/85 dark:shadow-none">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Usable Periods</p>
                <p className="mt-2 text-lg font-semibold">{profile.usable_periods}</p>
              </div>
              <div className="rounded-2xl border bg-white/80 p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/85 dark:shadow-none">
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Volatility</p>
                <p className="mt-2 text-lg font-semibold">{profile.volatility.toFixed(2)}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            {STEP_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = currentStep === item.step;
              const isDone = currentStep > item.step;
              return (
                <button key={item.step} type="button" onClick={() => setCurrentStep(item.step)} className="flex items-center gap-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold ${isActive ? 'bg-primary text-primary-foreground shadow-sm shadow-primary/30' : isDone ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground dark:bg-slate-900 dark:text-slate-400'}`}>
                    {isDone ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
                  </div>
                  <span className={`text-sm font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>{item.label}</span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="space-y-6">
        <AnimatePresence mode="wait">
          <motion.div key={`ts-step-${currentStep}`} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={transition}>
            {currentStep === 1 && (
              <Card>
                <CardHeader>
                  <CardTitle>Step 1: Data Configuration</CardTitle>
                  <CardDescription>Choose the date axis, review the smart target default, and set the future horizon.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Date Column</Label>
                      <Select value={dateColumn} onValueChange={setDateColumn}>
                        <SelectTrigger><SelectValue placeholder="Select date column" /></SelectTrigger>
                        <SelectContent>{dateColumns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}</SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">{preferredDateColumn ? `Smart default: ${preferredDateColumn}` : 'No suitable default found. Select the date column manually.'}</p>
                    </div>
                    <div className="space-y-2">
                      <Label>Sales Target</Label>
                      <Select value={targetColumn} onValueChange={setTargetColumn}>
                        <SelectTrigger><SelectValue placeholder="Select sales target" /></SelectTrigger>
                        <SelectContent>{numericColumns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}</SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">{smartTargetColumn ? `Smart default: ${smartTargetColumn}` : 'No suitable default found. Select the target manually.'}</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Future Periods</Label>
                    <Select value={String(forecastPeriods)} onValueChange={(value) => setForecastPeriods(Number(value))}>
                      <SelectTrigger><SelectValue placeholder="Select horizon" /></SelectTrigger>
                      <SelectContent>{HORIZON_OPTIONS.map((value) => <SelectItem key={value} value={String(value)}>{value} periods</SelectItem>)}</SelectContent>
                    </Select>
                  </div>
                  <div className="flex justify-end">
                    <Button onClick={() => setCurrentStep(2)} className="gap-2">
                      Next: Statistical Models <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {currentStep === 2 && (
              <Card>
                <CardHeader>
                  <CardTitle>Step 2: Three Model Recommendations</CardTitle>
                  <CardDescription>Stationarity is checked automatically. The recommended model badge is based on ADF + KPSS tests.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  {/* Stationarity Note */}
                  <div className="rounded-xl border border-primary/20 bg-gradient-to-r from-primary/6 via-background to-secondary/70 p-4">
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg border border-primary/20 bg-background text-primary shadow-sm">
                        <Waves className="h-4 w-4" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-semibold">Stationarity Note</p>
                        {stationarityLoading ? (
                          <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="h-3 w-3 animate-spin" /> Checking stationarity...
                          </div>
                        ) : stationarityError ? (
                          <p className="mt-1 text-sm leading-5 text-destructive">{stationarityError}</p>
                        ) : stationarity ? (
                          <>
                            <p className="mt-1 text-sm leading-5 text-muted-foreground">{stationarity.note}</p>
                            {frequencyNote && (
                              <p className="mt-1 text-xs leading-5 text-amber-700 dark:text-amber-300">{frequencyNote}</p>
                            )}
                            <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted-foreground">
                              <span>ADF: {stationarity.adf_pvalue}</span>
                              <span>KPSS: {stationarity.kpss_pvalue}</span>
                              <span>Status: <span className={`font-medium ${stationarity.status === 'stationary' ? 'text-emerald-600' : stationarity.status === 'non_stationary' ? 'text-red-600' : 'text-amber-600'}`}>{stationarity.status}</span></span>
                              {stationarity.period_label && <span>Frequency: {periodGrainLabel(stationarity.period_label)}</span>}
                            </div>
                          </>
                        ) : (
                          <p className="mt-1 text-sm text-muted-foreground">Run the time-series model to capture the stationarity note.</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 3 Model Cards */}
                  <div className="grid gap-4 md:grid-cols-3">
                    {['SARIMA', 'Prophet', 'HoltWinters'].map((model) => (
                      <div
                        key={model}
                        className={`rounded-2xl border p-5 transition-all ${
                          stationarity?.recommended_model === model
                            ? 'border-primary/40 bg-primary/5 shadow-sm shadow-primary/10 ring-1 ring-primary/20'
                            : 'bg-background dark:bg-slate-950/70'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-base font-semibold">{model}</span>
                          {stationarity?.recommended_model === model && (
                            <Badge className="bg-primary text-primary-foreground text-[10px] px-1.5 py-0">Recommended</Badge>
                          )}
                        </div>
                        <p className="mt-2 text-sm text-muted-foreground">{MODEL_DESCRIPTIONS[model]}</p>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {MODEL_STRENGTHS[model].map((s) => (
                            <span key={s} className="inline-flex items-center rounded-full border bg-background px-2 py-0.5 text-[10px] font-medium text-muted-foreground">{s}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="flex justify-between">
                    <Button variant="outline" onClick={() => setCurrentStep(1)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                    <Button onClick={() => setCurrentStep(3)} className="gap-2">Next: Train & Forecast <ArrowRight className="h-4 w-4" /></Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {currentStep === 3 && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Step 3: Training And Forecasting</CardTitle>
                    <CardDescription>All 3 models are trained. The best is auto-selected by lowest SMAPE (tiebreak: SARIMA &gt; HoltWinters &gt; Prophet).</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>Training Split (%)</Label>
                        <Input type="number" min={50} max={90} value={trainSplitPercent} onChange={(event) => setTrainSplitPercent(Math.max(50, Math.min(90, Number(event.target.value) || 80)))} />
                      </div>
                      <div className="rounded-xl border bg-muted/20 p-4 dark:border-slate-800 dark:bg-slate-900/70">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Selection Mode</p>
                        <p className="mt-2 font-semibold">Auto-select best of 3</p>
                      </div>
                    </div>
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setCurrentStep(2)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                      <div className="flex gap-2">
                        <Button onClick={handleRun} disabled={isTraining} variant="outline" className="gap-2">
                          {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                          {isTraining ? 'Training...' : 'Legacy TS'}
                        </Button>
                        <Button onClick={handleMultiModelRun} disabled={isTraining} className="gap-2">
                          {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                          {isTraining ? 'Training All 3...' : 'Train & Auto-Select'}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Model Comparison Table */}
                {modelComparison.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Model Comparison</CardTitle>
                      <CardDescription>SARIMA, Prophet, and HoltWinters compared with walk-forward validation.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Model</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>MAE</TableHead>
                            <TableHead>RMSE</TableHead>
                            <TableHead>MAPE</TableHead>
                            <TableHead>SMAPE</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {modelComparison.map((row) => (
                            <TableRow key={row.model} className={row.model === selectedModel ? 'bg-primary/5' : ''}>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  {row.model}
                                  {row.model === selectedModel && (
                                    <Badge className="bg-emerald-100 text-emerald-800 hover:bg-emerald-100 border-emerald-200 text-[10px]">Auto-Selected</Badge>
                                  )}
                                </div>
                              </TableCell>
                              <TableCell>
                                <Badge variant="outline" className={modelStatusClass(row.status)}>{row.status}</Badge>
                              </TableCell>
                              <TableCell>{formatIndianNumber(row.mae)}</TableCell>
                              <TableCell>{formatIndianNumber(row.rmse)}</TableCell>
                              <TableCell>{row.mape != null ? row.mape + '%' : 'N/A'}</TableCell>
                              <TableCell>{row.smape != null ? row.smape + '%' : 'N/A'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      {selectionReason && (
                        <p className="mt-3 text-xs text-muted-foreground">{selectionReason}</p>
                      )}
                    </CardContent>
                  </Card>
                )}

                {/* Results from legacy flow */}
                {result && !modelComparison.length && (
                  <>
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Model</CardDescription><CardTitle className="text-2xl">{result.training_summary.model_name}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Statistical time-series model family selected for the series.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Split</CardDescription><CardTitle className="text-2xl">{result.training_summary.train_percentage}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Training share with the remaining {result.training_summary.test_percentage}% held out for backtesting.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>MAE</CardDescription><CardTitle className="text-2xl">{result.metrics.mae.toLocaleString()}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average absolute forecast error on the backtest window.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>MAPE</CardDescription><CardTitle className="text-2xl">{result.metrics.mape}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average percentage error over held-out periods.</CardContent></Card>
                    </div>

                    <Card>
                      <CardHeader>
                        <CardTitle>Production Validation</CardTitle>
                        <CardDescription>Auto-selection, data quality gate, naive baseline, and calculation assumptions.</CardDescription>
                      </CardHeader>
                      <CardContent className="grid gap-4 lg:grid-cols-3">
                        <div className="rounded-xl border p-4">
                          <p className="text-xs uppercase tracking-wide text-muted-foreground">Data Quality</p>
                          <p className="mt-2 text-2xl font-bold">{result.data_quality?.score ?? 'N/A'}</p>
                          <p className="mt-1 text-sm text-muted-foreground">{result.data_quality?.status ?? 'Not scored'}</p>
                        </div>
                        <div className="rounded-xl border p-4">
                          <p className="text-xs uppercase tracking-wide text-muted-foreground">Naive Baseline MAE</p>
                          <p className="mt-2 text-2xl font-bold">{result.naive_baseline?.metrics.mae ?? 'N/A'}</p>
                          <p className="mt-1 text-sm text-muted-foreground">{result.naive_baseline?.mae_improvement_pct ?? 0}% MAE improvement</p>
                        </div>
                        <div className="rounded-xl border p-4">
                          <p className="text-xs uppercase tracking-wide text-muted-foreground">Audit Trail</p>
                          <p className="mt-2 text-sm text-muted-foreground">{result.assumptions_audit?.slice(0, 2).join(' ') ?? 'No assumptions captured.'}</p>
                        </div>
                      </CardContent>
                    </Card>

                    {result.model_comparison?.length ? (
                      <Card>
                        <CardHeader>
                          <CardTitle>Model Comparison</CardTitle>
                          <CardDescription>SARIMA, ARIMA, and Prophet are compared with walk-forward validation; the lowest-error completed candidate is selected automatically.</CardDescription>
                        </CardHeader>
                        <CardContent>
                          {result.validation_warnings?.length ? (
                            <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                              {result.validation_warnings.join(' ')}
                            </div>
                          ) : null}
                          <Table>
                            <TableHeader><TableRow><TableHead>Model</TableHead><TableHead>Status</TableHead><TableHead>MAE</TableHead><TableHead>RMSE</TableHead><TableHead>MAPE</TableHead><TableHead>Availability / Training Note</TableHead></TableRow></TableHeader>
                            <TableBody>
                              {result.model_comparison.map((model) => (
                                <TableRow key={model.model_type}>
                                  <TableCell>{model.model_name}</TableCell>
                                  <TableCell><Badge variant="outline" className={modelStatusClass(model.status)}>{model.status}</Badge></TableCell>
                                  <TableCell>{model.metrics?.mae ?? 'N/A'}</TableCell>
                                  <TableCell>{model.metrics?.rmse ?? 'N/A'}</TableCell>
                                  <TableCell>{model.metrics?.mape ?? 'N/A'}</TableCell>
                                  <TableCell className="max-w-md text-sm text-muted-foreground">
                                    {[model.availability_note, model.skip_reason, model.tuning?.note].filter(Boolean).join(' ')}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </CardContent>
                      </Card>
                    ) : null}
                  </>
                )}

                {/* Chart */}
                {(result && (chartData.length > 0 || multiChartData.length > 0)) && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Historical Vs Forecast</CardTitle>
                      <CardDescription>The shaded band shows the 95% confidence interval.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={chartData.length > 0 ? chartData : multiChartData}>
                            <CartesianGrid stroke={TS_CHART_COLORS.grid} strokeDasharray="3 3" opacity={0.35} />
                            <XAxis dataKey="period" tickLine={false} axisLine={false} tickMargin={10} tick={{ fill: '#64748b', fontSize: 12 }} />
                            <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b', fontSize: 12 }} />
                            <RechartsTooltip content={<ForecastTooltip />} />
                            <Legend />
                            <Area type="monotone" dataKey="lowerBand" name="Lower 95%" stackId="confidence" stroke="transparent" fill={TS_CHART_COLORS.bandBase} fillOpacity={0.12} isAnimationActive={false} />
                            <Area type="monotone" dataKey="confidenceRange" name="95% Confidence Band" stackId="confidence" stroke="transparent" fill={TS_CHART_COLORS.band} fillOpacity={0.18} isAnimationActive={false} />
                            <Line type="monotone" connectNulls dataKey="actual" name="Actual" stroke={TS_CHART_COLORS.actual} strokeWidth={3} dot={{ r: 4, fill: '#ffffff', stroke: TS_CHART_COLORS.actual, strokeWidth: 2.5 }} activeDot={{ r: 6, fill: TS_CHART_COLORS.actual, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                            <Line type="monotone" connectNulls dataKey="backtest" name="Backtest" stroke={TS_CHART_COLORS.backtest} strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3.5, fill: '#ffffff', stroke: TS_CHART_COLORS.backtest, strokeWidth: 2 }} activeDot={{ r: 5, fill: TS_CHART_COLORS.backtest, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                            <Line type="monotone" connectNulls dataKey="forecast" name="Forecast" stroke={TS_CHART_COLORS.forecast} strokeWidth={3} dot={{ r: 4, fill: '#ffffff', stroke: TS_CHART_COLORS.forecast, strokeWidth: 2.5 }} activeDot={{ r: 6, fill: TS_CHART_COLORS.forecast, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Future Forecast Table */}
                {forecastResults.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Future Forecast Table</CardTitle>
                      <CardDescription>Future forecast values with 95% confidence intervals.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Period</TableHead>
                            <TableHead>Forecast</TableHead>
                            <TableHead>Lower 95%</TableHead>
                            <TableHead>Upper 95%</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {forecastResults.map((point) => (
                            <TableRow key={point.period}>
                              <TableCell>{point.period}</TableCell>
                              <TableCell>{formatIndianNumber(point.forecast)}</TableCell>
                              <TableCell>{formatIndianNumber(point.lower)}</TableCell>
                              <TableCell>{formatIndianNumber(point.upper)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                )}

                {/* Legacy future forecast table */}
                {result && result.future_forecast.length > 0 && forecastResults.length === 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Future Forecast Table</CardTitle>
                      <CardDescription>Future months forecast values with the model's projected horizon.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Period</TableHead>
                            <TableHead>Forecast</TableHead>
                            <TableHead>Lower 95%</TableHead>
                            <TableHead>Upper 95%</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.future_forecast.map((point) => (
                            <TableRow key={point.period}>
                              <TableCell>{point.period}</TableCell>
                              <TableCell>{point.predicted.toLocaleString()}</TableCell>
                              <TableCell>{point.lower != null ? point.lower.toLocaleString() : 'N/A'}</TableCell>
                              <TableCell>{point.upper != null ? point.upper.toLocaleString() : 'N/A'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                )}

                {/* Insight Panel */}
                {insight && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Forecast Insight</CardTitle>
                      <CardDescription>Programmatic insight generated from metrics — no LLM call.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm leading-6 text-muted-foreground">{insight.insight_text}</p>
                      {insight.risk_flag && (
                        <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                          <span>{insight.risk_flag}</span>
                        </div>
                      )}
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="secondary" className="text-xs">
                          Confidence: {insight.confidence}
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          Selection: {insight.selection_metric}
                        </Badge>
                      </div>
                      <div className="flex justify-end gap-2">
                        <Button variant="outline" onClick={() => setCurrentStep(2)}>Try Another TS Model</Button>
                        <Button onClick={() => setActiveTab(modelTrained ? 'prediction' : 'forecast_ml')} className="gap-2">
                          {modelTrained ? 'Continue To Prediction' : 'Continue To ML Forecast'}<ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Legacy insight */}
                {result && result.analysis && !insight && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Forecast Insight</CardTitle>
                      <CardDescription>Time-series summary for business review.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm leading-6 text-muted-foreground">{result.analysis}</p>
                      <div className="flex justify-end gap-2">
                        <Button variant="outline" onClick={() => setCurrentStep(2)}>Try Another TS Model</Button>
                        <Button onClick={() => setActiveTab(modelTrained ? 'prediction' : 'forecast_ml')} className="gap-2">
                          {modelTrained ? 'Continue To Prediction' : 'Continue To ML Forecast'}<ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
