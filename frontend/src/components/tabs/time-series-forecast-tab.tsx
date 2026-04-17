'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAppStore, type ColumnInfo, type DataRow, type TimeSeriesForecastResult } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertCircle, ArrowRight, CalendarDays, CheckCircle2, ChevronLeft, Loader2, RadioTower, ShieldCheck, TrendingUp, Waves, Zap } from 'lucide-react';
import { Area, ComposedChart, CartesianGrid, Legend, Line, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';

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

const transition = { duration: 0.28, ease: 'easeOut' } as const;

function getPreferredSalesColumn(columns: ColumnInfo[]) {
  const numeric = columns.filter((column) => column.role === 'numeric');
  return numeric.find((column) => /sales|revenue|amount|profit|price|total|qty/i.test(column.name))?.name ?? numeric[0]?.name ?? '';
}

function getPreferredDateColumn(columns: ColumnInfo[]) {
  return columns.find((column) => column.role === 'datetime' && /doc_date|invoice_date|order_date|date/i.test(column.name))?.name
    ?? columns.find((column) => column.role === 'datetime')?.name
    ?? columns.find((column) => /date|month|time|period/i.test(column.name))?.name
    ?? '';
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
  const detected_frequency = medianDays <= 2 ? 'day' : medianDays <= 10 ? 'week' : medianDays <= 45 ? 'month' : medianDays <= 120 ? 'quarter' : 'year';

  return {
    detected_frequency,
    usable_periods: points.length,
    volatility: mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean),
    zero_value_share: values.filter((value) => value === 0).length / values.length,
  };
}

function buildModelRecommendations(profile: ReturnType<typeof inferSeriesProfile>) {
  const hasSeasonality = profile.usable_periods >= (profile.detected_frequency === 'month' ? 12 : profile.detected_frequency === 'quarter' ? 8 : 10);
  return [
    {
      model_type: 'sarima',
      model_name: 'SARIMA',
      recommended: hasSeasonality,
      recommendation_reason: hasSeasonality ? 'Recommended due to recurring seasonality and enough historical periods.' : 'Useful when you want an explicit seasonal statistical baseline.',
    },
    {
      model_type: 'prophet',
      model_name: 'Prophet',
      recommended: !hasSeasonality,
      recommendation_reason: !hasSeasonality ? 'Recommended when the series shows smoother trend movement and flexible seasonality.' : 'Alternative when you want trend-focused decomposition.',
    },
    {
      model_type: 'arima',
      model_name: 'ARIMA',
      recommended: false,
      recommendation_reason: 'Good fallback for short or relatively stable series without strong seasonal structure.',
    },
  ];
}

export default function TimeSeriesForecastTab() {
  const { toast } = useToast();
  const rawData = useAppStore((state) => state.rawData);
  const cleanedData = useAppStore((state) => state.cleanedData);
  const columns = useAppStore((state) => state.columns);
  const datasetId = useAppStore((state) => state.datasetId);
  const modelTrained = useAppStore((state) => state.modelTrained);
  const storedResult = useAppStore((state) => state.timeSeriesForecastResult);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const data = cleanedData ?? rawData ?? [];

  const numericColumns = useMemo(() => columns.filter((column) => column.role === 'numeric'), [columns]);
  const dateColumns = useMemo(() => columns.filter((column) => column.role === 'datetime' || /date|month|time|period/i.test(column.name)), [columns]);

  const [currentStep, setCurrentStep] = useState(1);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(3);
  const [trainSplitPercent, setTrainSplitPercent] = useState(80);
  const [selectedModelType, setSelectedModelType] = useState('sarima');
  const [result, setResult] = useState<TimeSeriesForecastResult | null>(storedResult);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    if (!dateColumn) setDateColumn(getPreferredDateColumn(columns));
    if (!targetColumn) setTargetColumn(getPreferredSalesColumn(columns));
  }, [columns, dateColumn, targetColumn]);

  useEffect(() => {
    if (storedResult) {
      setResult(storedResult);
      setCurrentStep(3);
    }
  }, [storedResult]);

  const profile = useMemo(() => inferSeriesProfile(data as DataRow[], dateColumn, targetColumn), [data, dateColumn, targetColumn]);
  const recommendations = useMemo(() => buildModelRecommendations(profile), [profile]);

  useEffect(() => {
    if (!recommendations.some((item) => item.model_type === selectedModelType)) {
      setSelectedModelType(recommendations[0]?.model_type ?? 'sarima');
    }
  }, [recommendations, selectedModelType]);

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
          confidenceRange:
            testPoint?.lower != null && testPoint?.upper != null
              ? Math.max(testPoint.upper - testPoint.lower, 0)
              : null,
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
        confidenceRange:
          item.lower != null && item.upper != null
            ? Math.max(item.upper - item.lower, 0)
            : null,
      })),
    ];
  }, [result]);

  const handleRun = async () => {
    if (!dateColumn || !targetColumn) {
      toast({ title: 'Configuration incomplete', description: 'Choose both a date column and a sales target.', variant: 'destructive' });
      return;
    }

    setIsTraining(true);
    try {
      const payload = {
        dataset_id: datasetId ?? null,
        data: datasetId ? [] : data,
        date_column: dateColumn,
        target_column: targetColumn,
        forecast_periods: forecastPeriods,
        test_percentage: 100 - trainSplitPercent,
        model_type: selectedModelType,
      };

      const response = await apiClient.post('/forecast/ts/run', payload);
      const nextResult = response.data as TimeSeriesForecastResult;
      setResult(nextResult);
      useAppStore.setState({ timeSeriesForecastResult: nextResult });
      toast({ title: 'Time-series forecast ready', description: `Projected ${forecastPeriods} future ${nextResult.period_label ?? 'period'}${forecastPeriods === 1 ? '' : 's'}.` });
    } catch (error) {
      toast({ title: 'Forecast failed', description: getApiErrorMessage(error, 'Time-series forecast failed.'), variant: 'destructive' });
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
                Trend and seasonality live inside the model here, so the series itself carries the forecasting signal instead of manually engineered features.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge variant="secondary">{dateColumn || 'Pick a date column'}</Badge>
                <Badge variant="secondary">{targetColumn || 'Pick a target column'}</Badge>
                <Badge variant="secondary">{forecastPeriods} future periods</Badge>
                <Badge variant="secondary">{recommendations.find((item) => item.model_type === selectedModelType)?.model_name ?? 'TS model selected'}</Badge>
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

      <Card className="border border-primary/20 bg-gradient-to-r from-primary/6 via-background to-secondary/70">
        <CardContent className="p-5">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-10 w-10 items-center justify-center rounded-xl border border-primary/20 bg-background text-primary shadow-sm">
              <Waves className="h-4 w-4" />
            </div>
            <div className="max-w-4xl">
              <p className="text-sm font-semibold">Stationarity Note</p>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                {result?.stationarity_check.note ?? 'Run the time-series model once to capture the stationarity note and recommended model behavior for this series.'}
              </p>
            </div>
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
                  <CardDescription>Choose the date axis, sales target, and future horizon. Training parameters stay hidden until step 3.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Date Column</Label>
                      <Select value={dateColumn} onValueChange={setDateColumn}>
                        <SelectTrigger><SelectValue placeholder="Select date column" /></SelectTrigger>
                        <SelectContent>{dateColumns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}</SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Sales Target</Label>
                      <Select value={targetColumn} onValueChange={setTargetColumn}>
                        <SelectTrigger><SelectValue placeholder="Select sales target" /></SelectTrigger>
                        <SelectContent>{numericColumns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}</SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Future Periods</Label>
                    <Input type="number" min={1} max={24} value={forecastPeriods} onChange={(event) => setForecastPeriods(Math.max(1, Math.min(24, Number(event.target.value) || 1)))} />
                  </div>
                  <Card className="border border-primary/20 bg-primary/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Stationarity Check</CardTitle>
                      <CardDescription>Read-only ADF-style diagnostic from the historical sales series.</CardDescription>
                    </CardHeader>
                    <CardContent className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-xl border bg-background p-4 dark:border-slate-800 dark:bg-slate-900/80"><p className="text-xs uppercase tracking-wide text-muted-foreground">Test</p><p className="mt-2 font-semibold">Dickey-Fuller</p></div>
                      <div className="rounded-xl border bg-background p-4 dark:border-slate-800 dark:bg-slate-900/80"><p className="text-xs uppercase tracking-wide text-muted-foreground">p-value</p><p className="mt-2 font-semibold">{result?.stationarity_check.p_value?.toFixed(3) ?? 'Pending'}</p></div>
                      <div className="rounded-xl border bg-background p-4 dark:border-slate-800 dark:bg-slate-900/80"><p className="text-xs uppercase tracking-wide text-muted-foreground">Verdict</p><p className="mt-2 font-semibold">{result?.stationarity_check.verdict ?? 'Will evaluate after model run'}</p></div>
                    </CardContent>
                  </Card>
                  <div className="flex justify-end"><Button onClick={() => setCurrentStep(2)} className="gap-2">Next: Statistical Models <ArrowRight className="h-4 w-4" /></Button></div>
                </CardContent>
              </Card>
            )}

            {currentStep === 2 && (
              <Card>
                <CardHeader>
                  <CardTitle>Step 2: Statistical Model Selection</CardTitle>
                  <CardDescription>Pick the time-series family that best matches the stationarity and seasonality profile.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {recommendations.map((model) => {
                    const isSelected = selectedModelType === model.model_type;
                    return (
                      <button key={model.model_type} type="button" onClick={() => setSelectedModelType(model.model_type)} className={`w-full rounded-2xl border p-5 text-left transition-all ${isSelected ? 'border-primary bg-primary/5 shadow-sm' : 'bg-background hover:border-primary/30 dark:bg-slate-950/70 dark:hover:border-primary/40'}`}>
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="flex flex-wrap items-center gap-2">
                              <p className="text-base font-semibold">{model.model_name}</p>
                              {model.recommended && <Badge className="bg-primary text-primary-foreground">Recommended</Badge>}
                            </div>
                            <p className="mt-2 text-sm text-muted-foreground">{model.recommendation_reason}</p>
                          </div>
                          <ShieldCheck className="h-5 w-5 text-primary" />
                        </div>
                      </button>
                    );
                  })}
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
                    <CardDescription>Training parameters appear here only. Post-training output focuses on forecast behavior and confidence bounds.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>Training Split (%)</Label>
                        <Input type="number" min={50} max={90} value={trainSplitPercent} onChange={(event) => setTrainSplitPercent(Math.max(50, Math.min(90, Number(event.target.value) || 80)))} />
                      </div>
                      <div className="rounded-xl border bg-muted/20 p-4 dark:border-slate-800 dark:bg-slate-900/70"><p className="text-xs uppercase tracking-wide text-muted-foreground">Selected Model</p><p className="mt-2 font-semibold">{recommendations.find((item) => item.model_type === selectedModelType)?.model_name ?? 'SARIMA'}</p></div>
                    </div>
                    <Card className="border-dashed">
                      <CardContent className="p-4 text-sm text-muted-foreground">
                        The stationarity note above helps explain whether this series behaves more like a stable statistical process or needs a more flexible trend-oriented model.
                      </CardContent>
                    </Card>
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setCurrentStep(2)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                      <Button onClick={handleRun} disabled={isTraining} className="gap-2">{isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}{isTraining ? 'Training Forecast...' : 'Train And Forecast'}</Button>
                    </div>
                  </CardContent>
                </Card>

                {result && (
                  <>
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Model</CardDescription><CardTitle className="text-2xl">{result.training_summary.model_name}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Statistical time-series model family selected for the series.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Split</CardDescription><CardTitle className="text-2xl">{result.training_summary.train_percentage}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Training share with the remaining {result.training_summary.test_percentage}% held out for backtesting.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>MAE</CardDescription><CardTitle className="text-2xl">{result.metrics.mae.toLocaleString()}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average absolute forecast error on the backtest window.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>MAPE</CardDescription><CardTitle className="text-2xl">{result.metrics.mape}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average percentage error over held-out periods.</CardContent></Card>
                    </div>

                    <Card>
                      <CardHeader>
                        <CardTitle>Historical Vs Forecast</CardTitle>
                        <CardDescription>The shaded band shows the 95% confidence interval. SHAP and feature-importance visuals are intentionally excluded from the time-series paradigm.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-80 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={chartData}>
                              <CartesianGrid stroke={TS_CHART_COLORS.grid} strokeDasharray="3 3" opacity={0.35} />
                              <XAxis dataKey="period" tickLine={false} axisLine={false} tickMargin={10} tick={{ fill: '#64748b', fontSize: 12 }} />
                              <YAxis tickLine={false} axisLine={false} tick={{ fill: '#64748b', fontSize: 12 }} />
                              <RechartsTooltip
                                contentStyle={{
                                  borderRadius: '14px',
                                  border: '1px solid rgba(148, 163, 184, 0.25)',
                                  backgroundColor: 'rgba(255,255,255,0.96)',
                                  boxShadow: '0 18px 45px rgba(15, 23, 42, 0.12)',
                                }}
                              />
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

                    <Card>
                      <CardHeader>
                        <CardTitle>Forecast Insight</CardTitle>
                        <CardDescription>Time-series summary for business review.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <p className="text-sm leading-6 text-muted-foreground">{result.analysis}</p>
                        <div className="flex justify-end gap-2">
                          <Button variant="outline" onClick={() => setCurrentStep(2)}>Try Another TS Model</Button>
                          <Button onClick={() => setActiveTab(modelTrained ? 'prediction' : 'forecast_ml')} className="gap-2">{modelTrained ? 'Continue To Prediction' : 'Continue To ML Forecast'}<ArrowRight className="h-4 w-4" /></Button>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
