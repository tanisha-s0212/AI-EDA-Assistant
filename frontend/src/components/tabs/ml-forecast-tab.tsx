'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAppStore, type ColumnInfo, type DataRow, type MlForecastResult } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { AlertCircle, ArrowRight, CheckCircle2, ChevronLeft, Cpu, Loader2, Settings2, Table2, TrendingUp, Zap } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

type FeatureGroupId = 'trend' | 'calendar' | 'lags' | 'rolling';

const FEATURE_GROUPS: { id: FeatureGroupId; label: string; description: string }[] = [
  { id: 'trend', label: 'Trend index', description: 'Numeric step counter that lets the model learn growth or decay.' },
  { id: 'calendar', label: 'Calendar markers', description: 'Month, quarter, weekday, and related date breakdowns.' },
  { id: 'lags', label: 'Lag features', description: 'Recent observed sales values transformed into explicit input columns.' },
  { id: 'rolling', label: 'Rolling windows', description: 'Moving averages and short-term momentum summaries.' },
];

const STEP_ITEMS = [
  { step: 1, label: 'Feature Setup', icon: Settings2 },
  { step: 2, label: 'ML Models', icon: Cpu },
  { step: 3, label: 'Train & Explain', icon: TrendingUp },
];

const ML_FORECAST_CHART_COLORS = {
  actual: '#2563eb',
  backtest: '#f59e0b',
  forecast: '#8b5cf6',
  shap: ['#2563eb', '#7c3aed', '#14b8a6', '#f59e0b', '#ef4444', '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'],
  grid: '#cbd5e1',
} as const;

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

function buildGeneratedFeaturePreview(featureGroups: FeatureGroupId[], lagPeriods: number) {
  const preview: string[] = [];
  if (featureGroups.includes('trend')) preview.push('trend_index');
  if (featureGroups.includes('calendar')) preview.push('month_number', 'quarter_number', 'weekday_number', 'is_month_end');
  if (featureGroups.includes('lags')) for (let lag = 1; lag <= lagPeriods; lag += 1) preview.push(`lag_${lag}`);
  if (featureGroups.includes('rolling')) preview.push('rolling_mean_3', 'rolling_mean_6', 'rolling_std_3');
  return preview;
}

function buildModelRecommendations(featureCount: number) {
  return [
    {
      model_type: 'gradient_boosting',
      model_name: 'Gradient Boosting',
      recommended: featureCount >= 6,
      recommendation_reason: `Excellent for capturing non-linear patterns across the ${featureCount} generated features.`,
    },
    {
      model_type: 'random_forest',
      model_name: 'Random Forest',
      recommended: featureCount < 6,
      recommendation_reason: 'Robust baseline when generated features are fewer and interactions still matter.',
    },
    {
      model_type: 'ridge_regression',
      model_name: 'Ridge Regression',
      recommended: false,
      recommendation_reason: 'Useful when you want a simpler, more stable relationship across engineered forecast features.',
    },
  ];
}

function formatForecastValue(value: number) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(value);
}

export default function MlForecastTab() {
  const { toast } = useToast();
  const rawData = useAppStore((state) => state.rawData);
  const cleanedData = useAppStore((state) => state.cleanedData);
  const columns = useAppStore((state) => state.columns);
  const datasetId = useAppStore((state) => state.datasetId);
  const modelTrained = useAppStore((state) => state.modelTrained);
  const storedResult = useAppStore((state) => state.mlForecastResult);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const data = cleanedData ?? rawData ?? [];

  const numericColumns = useMemo(() => columns.filter((column) => column.role === 'numeric'), [columns]);
  const dateColumns = useMemo(() => columns.filter((column) => column.role === 'datetime' || /date|month|time|period/i.test(column.name)), [columns]);

  const [currentStep, setCurrentStep] = useState(1);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(3);
  const [trainSplitPercent, setTrainSplitPercent] = useState(80);
  const [lagPeriods, setLagPeriods] = useState(3);
  const [featureGroups, setFeatureGroups] = useState<FeatureGroupId[]>(['trend', 'calendar', 'lags', 'rolling']);
  const [selectedModelType, setSelectedModelType] = useState('gradient_boosting');
  const [result, setResult] = useState<MlForecastResult | null>(storedResult);
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
  const generatedFeatures = useMemo(() => buildGeneratedFeaturePreview(featureGroups, lagPeriods), [featureGroups, lagPeriods]);
  const recommendations = useMemo(() => buildModelRecommendations(generatedFeatures.length), [generatedFeatures.length]);

  useEffect(() => {
    if (!recommendations.some((item) => item.model_type === selectedModelType)) {
      setSelectedModelType(recommendations[0]?.model_type ?? 'gradient_boosting');
    }
  }, [recommendations, selectedModelType]);

  const chartData = useMemo(() => {
    if (!result) return [];
    return [
      ...result.history.map((item) => {
        const backtest = result.test_forecast.find((forecast) => forecast.period === item.period);
        return { period: item.period, actual: item.actual, backtest: backtest?.predicted ?? null, forecast: null as number | null };
      }),
      ...result.future_forecast.map((item) => ({ period: item.period, actual: null, backtest: null, forecast: item.predicted })),
    ];
  }, [result]);

  const shapData = useMemo(() => (result?.shap_feature_importance ?? []).slice(0, 10).map((item) => ({ ...item, display: Number(item.importance.toFixed(3)) })), [result]);
  const forecastSummary = useMemo(() => {
    if (!result?.future_forecast.length) return null;

    const values = result.future_forecast.map((item) => item.predicted);
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    const maxPoint = result.future_forecast.reduce((highest, current) => current.predicted > highest.predicted ? current : highest, result.future_forecast[0]);

    return {
      average,
      maxPoint,
    };
  }, [result]);

  const toggleFeatureGroup = (feature: FeatureGroupId) => {
    setFeatureGroups((previous) => previous.includes(feature) ? previous.filter((item) => item !== feature) : [...previous, feature]);
  };

  const handleRun = async () => {
    if (!dateColumn || !targetColumn || featureGroups.length === 0) {
      toast({ title: 'Configuration incomplete', description: 'Choose the core columns and at least one feature engineering group.', variant: 'destructive' });
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
        lag_periods: lagPeriods,
        model_type: selectedModelType,
        feature_groups: featureGroups,
      };

      const response = await apiClient.post('/forecast/ml/run', payload);
      const nextResult = response.data as MlForecastResult;
      setResult(nextResult);
      useAppStore.setState({ mlForecastResult: nextResult });
      toast({ title: 'ML forecast ready', description: `Projected ${forecastPeriods} future ${nextResult.period_label ?? 'period'}${forecastPeriods === 1 ? '' : 's'}.` });
    } catch (error) {
      toast({ title: 'Forecast failed', description: getApiErrorMessage(error, 'ML forecasting failed.'), variant: 'destructive' });
    } finally {
      setIsTraining(false);
    }
  };

  if (!data.length || !columns.length) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Sales Forecast Using Machine Learning</h2>
          <p className="mt-1 text-muted-foreground">This workflow expects the cleaned dataset cached after Step 3 and turns time into explicit engineered features.</p>
        </div>
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <div>
              <p className="font-medium">Upload and clean a dataset first</p>
              <p className="mt-1 text-sm text-muted-foreground">Step 6 depends on the cleaned cached dataset before it can build forecast features.</p>
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
                <TrendingUp className="h-3.5 w-3.5" />
                Forecast ML
              </div>
              <h2 className="mt-3 text-2xl font-bold tracking-tight">Sales Forecast Using Machine Learning</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Time is treated as just another input here, so feature engineering drives forecast quality, explainability, and the shape of the final projection.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge variant="secondary">{dateColumn || 'Pick a date column'}</Badge>
                <Badge variant="secondary">{targetColumn || 'Pick a target column'}</Badge>
                <Badge variant="secondary">{generatedFeatures.length} generated features</Badge>
                <Badge variant="secondary">{forecastPeriods} future periods</Badge>
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
          <motion.div key={`mlf-step-${currentStep}`} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.22, ease: 'easeOut' }}>
            {currentStep === 1 && (
              <Card>
                <CardHeader>
                  <CardTitle>Step 1: Target And Feature Engineering</CardTitle>
                  <CardDescription>Choose the forecasting columns and explicitly tell the model which time-derived features to generate.</CardDescription>
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
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Future Periods</Label>
                      <Input type="number" min={1} max={24} value={forecastPeriods} onChange={(event) => setForecastPeriods(Math.max(1, Math.min(24, Number(event.target.value) || 1)))} />
                    </div>
                    <div className="space-y-2">
                      <Label>Lag Depth</Label>
                      <Input type="number" min={1} max={12} value={lagPeriods} onChange={(event) => setLagPeriods(Math.max(1, Math.min(12, Number(event.target.value) || 1)))} />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {FEATURE_GROUPS.map((group) => (
                      <label key={group.id} className="flex cursor-pointer items-start gap-3 rounded-xl border p-4 transition-colors hover:border-primary/30 dark:border-slate-800 dark:bg-slate-950/60 dark:hover:border-primary/40">
                        <Checkbox checked={featureGroups.includes(group.id)} onCheckedChange={() => toggleFeatureGroup(group.id)} />
                        <div>
                          <p className="text-sm font-medium">{group.label}</p>
                          <p className="mt-1 text-sm text-muted-foreground">{group.description}</p>
                        </div>
                      </label>
                    ))}
                  </div>
                  <Card className="border-dashed dark:border-slate-800 dark:bg-slate-950/65">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <CardTitle className="text-base">Generated Forecast Features</CardTitle>
                          <CardDescription>These are the engineered inputs the ML forecast will rely on.</CardDescription>
                        </div>
                        <Badge variant="secondary">{generatedFeatures.length} features</Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex flex-wrap gap-2">
                        {generatedFeatures.map((feature) => <Badge key={feature} variant="outline">{feature}</Badge>)}
                      </div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Feature</TableHead>
                            <TableHead>Source</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {generatedFeatures.map((feature) => (
                            <TableRow key={feature}>
                              <TableCell className="font-medium">{feature}</TableCell>
                              <TableCell className="text-muted-foreground">{feature.startsWith('lag_') ? 'Lag history' : feature.startsWith('rolling_') ? 'Rolling windows' : feature.includes('trend') ? 'Trend index' : 'Calendar markers'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                  <div className="flex justify-end"><Button onClick={() => setCurrentStep(2)} className="gap-2">Next: ML Model Selection <ArrowRight className="h-4 w-4" /></Button></div>
                </CardContent>
              </Card>
            )}

            {currentStep === 2 && (
              <Card>
                <CardHeader>
                  <CardTitle>Step 2: ML Model Selection</CardTitle>
                  <CardDescription>Choose the learner best suited for the engineered forecast feature set.</CardDescription>
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
                          <Cpu className="h-5 w-5 text-primary" />
                        </div>
                      </button>
                    );
                  })}
                  <div className="flex justify-between">
                    <Button variant="outline" onClick={() => setCurrentStep(1)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                    <Button onClick={() => setCurrentStep(3)} className="gap-2">Next: Train & Explain <ArrowRight className="h-4 w-4" /></Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {currentStep === 3 && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Step 3: Training, Explainability, And Forecasting</CardTitle>
                    <CardDescription>Training parameters appear here only. The post-training view emphasizes the forecast line plus SHAP-style importance for generated features.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>Training Split (%)</Label>
                        <Input type="number" min={50} max={90} value={trainSplitPercent} onChange={(event) => setTrainSplitPercent(Math.max(50, Math.min(90, Number(event.target.value) || 80)))} />
                      </div>
                      <div className="rounded-xl border bg-muted/20 p-4 dark:border-slate-800 dark:bg-slate-900/70"><p className="text-xs uppercase tracking-wide text-muted-foreground">Selected Model</p><p className="mt-2 font-semibold">{recommendations.find((item) => item.model_type === selectedModelType)?.model_name ?? 'Gradient Boosting'}</p></div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {generatedFeatures.map((feature) => <Badge key={feature} variant="outline">{feature}</Badge>)}
                    </div>
                    <div className="flex justify-between">
                      <Button variant="outline" onClick={() => setCurrentStep(2)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                      <Button onClick={handleRun} disabled={isTraining} className="gap-2">{isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}{isTraining ? 'Training Forecast...' : 'Train And Forecast'}</Button>
                    </div>
                  </CardContent>
                </Card>

                {result && (
                  <>
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Model</CardDescription><CardTitle className="text-2xl">{result.training_summary.model_name}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">ML learner selected for the engineered forecast features.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>Features</CardDescription><CardTitle className="text-2xl">{result.generated_features.length}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Generated features driving the final forecast.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>MAE</CardDescription><CardTitle className="text-2xl">{result.metrics.mae.toLocaleString()}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average absolute forecast error on the holdout window.</CardContent></Card>
                      <Card className="dark:border-slate-800 dark:bg-slate-950/75"><CardHeader className="pb-2"><CardDescription>RMSE</CardDescription><CardTitle className="text-2xl">{result.metrics.rmse.toLocaleString()}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Penalty on larger miss distances during backtesting.</CardContent></Card>
                    </div>

                    <div className="space-y-6">
                      <Card className="overflow-hidden border border-primary/20 shadow-sm dark:bg-slate-950/80 dark:shadow-none">
                        <CardHeader>
                          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                            <div>
                              <CardTitle>Forecast Line</CardTitle>
                              <CardDescription>Historical actuals, backtest predictions, and future ML forecast. Confidence intervals are intentionally excluded here.</CardDescription>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">Actuals + Backtest + Future</Badge>
                              <Badge variant="outline" className="border-secondary bg-secondary text-secondary-foreground">{result.future_forecast.length} projected periods</Badge>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-5">
                          <div className="grid gap-3 sm:grid-cols-3">
                            <div className="rounded-2xl border bg-gradient-to-br from-primary/8 to-background p-4">
                              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Latest Actual</p>
                              <p className="mt-2 text-xl font-semibold text-primary">
                                {result.history.length ? formatForecastValue(result.history[result.history.length - 1].actual) : 'N/A'}
                              </p>
                            </div>
                            <div className="rounded-2xl border bg-gradient-to-br from-secondary to-background p-4">
                              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Avg Forecast</p>
                              <p className="mt-2 text-xl font-semibold text-foreground">
                                {forecastSummary ? formatForecastValue(forecastSummary.average) : 'N/A'}
                              </p>
                            </div>
                            <div className="rounded-2xl border bg-gradient-to-br from-muted/30 to-background p-4">
                              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Peak Projection</p>
                              <p className="mt-2 text-xl font-semibold text-foreground">
                                {forecastSummary ? formatForecastValue(forecastSummary.maxPoint.predicted) : 'N/A'}
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">{forecastSummary?.maxPoint.period ?? 'No horizon'}</p>
                            </div>
                          </div>
                          <div className="h-[500px] w-full rounded-2xl border bg-gradient-to-br from-muted/20 via-background to-primary/5 p-5 dark:border-slate-800 dark:from-slate-950 dark:via-slate-950 dark:to-primary/10">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={chartData} margin={{ top: 10, right: 18, left: 0, bottom: 6 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={ML_FORECAST_CHART_COLORS.grid} vertical={false} opacity={0.35} />
                                <XAxis dataKey="period" tickLine={false} axisLine={false} tickMargin={10} tick={{ fill: '#64748b', fontSize: 12 }} />
                                <YAxis tickLine={false} axisLine={false} width={72} tick={{ fill: '#64748b', fontSize: 12 }} tickFormatter={(value) => formatForecastValue(Number(value))} />
                                <RechartsTooltip
                                  formatter={(value: number | string) => typeof value === 'number' ? formatForecastValue(value) : value}
                                  contentStyle={{
                                    borderRadius: '14px',
                                    border: '1px solid rgba(148, 163, 184, 0.25)',
                                    backgroundColor: 'rgba(255,255,255,0.96)',
                                    boxShadow: '0 18px 45px rgba(15, 23, 42, 0.12)',
                                  }}
                                />
                                <Legend />
                                <Line type="monotone" connectNulls dataKey="actual" name="Actual" stroke={ML_FORECAST_CHART_COLORS.actual} strokeWidth={3} dot={{ r: 4, strokeWidth: 2.5, fill: '#ffffff', stroke: ML_FORECAST_CHART_COLORS.actual }} activeDot={{ r: 6, fill: ML_FORECAST_CHART_COLORS.actual, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                                <Line type="monotone" connectNulls dataKey="backtest" name="Backtest" stroke={ML_FORECAST_CHART_COLORS.backtest} strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 3.5, strokeWidth: 2, fill: '#ffffff', stroke: ML_FORECAST_CHART_COLORS.backtest }} activeDot={{ r: 5, fill: ML_FORECAST_CHART_COLORS.backtest, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                                <Line type="monotone" connectNulls dataKey="forecast" name="Forecast" stroke={ML_FORECAST_CHART_COLORS.forecast} strokeWidth={3} dot={{ r: 4, strokeWidth: 2.5, fill: '#ffffff', stroke: ML_FORECAST_CHART_COLORS.forecast }} activeDot={{ r: 6, fill: ML_FORECAST_CHART_COLORS.forecast, stroke: '#ffffff', strokeWidth: 2 }} isAnimationActive={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>

                      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
                        <Card className="border-slate-200/80 shadow-sm dark:border-slate-800 dark:bg-slate-950/80 dark:shadow-none">
                          <CardHeader>
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <CardTitle>SHAP Feature Importance</CardTitle>
                                <CardDescription>Which generated forecast features most influenced the model output.</CardDescription>
                              </div>
                              <Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">Top {shapData.length}</Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="grid gap-3 sm:grid-cols-2">
                              <div className="rounded-2xl border bg-gradient-to-br from-primary/8 to-background p-4">
                                <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Top Driver</p>
                                <p className="mt-2 text-lg font-semibold text-primary">{shapData[0]?.name ?? 'N/A'}</p>
                              </div>
                              <div className="rounded-2xl border bg-gradient-to-br from-muted/20 to-background p-4 dark:border-slate-800 dark:from-slate-900 dark:to-slate-950">
                                <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Importance</p>
                                <p className="mt-2 text-lg font-semibold text-foreground">{shapData[0] ? shapData[0].display.toFixed(3) : 'N/A'}</p>
                              </div>
                            </div>
                            <div className="h-[380px] w-full rounded-2xl border bg-gradient-to-br from-muted/20 to-background p-3 dark:border-slate-800 dark:from-slate-950 dark:to-slate-900">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={shapData} layout="vertical" margin={{ left: 28, right: 16, top: 10, bottom: 10 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke={ML_FORECAST_CHART_COLORS.grid} horizontal={false} opacity={0.25} />
                                  <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: '#64748b', fontSize: 12 }} />
                                  <YAxis type="category" dataKey="name" width={140} tickLine={false} axisLine={false} tick={{ fill: '#475569', fontSize: 12 }} />
                                  <RechartsTooltip
                                    formatter={(value: number | string) => typeof value === 'number' ? value.toFixed(3) : value}
                                    contentStyle={{
                                      borderRadius: '14px',
                                      border: '1px solid rgba(148, 163, 184, 0.25)',
                                      backgroundColor: 'rgba(255,255,255,0.96)',
                                      boxShadow: '0 18px 45px rgba(15, 23, 42, 0.12)',
                                    }}
                                  />
                                  <Bar dataKey="display" name="Importance" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                                    {shapData.map((_, index) => (
                                      <Cell key={`shap-bar-${index}`} fill={ML_FORECAST_CHART_COLORS.shap[index % ML_FORECAST_CHART_COLORS.shap.length]} />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </CardContent>
                        </Card>

                        <Card className="border-slate-200/80 shadow-sm dark:border-slate-800 dark:bg-slate-950/80 dark:shadow-none">
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <CardTitle>Future Forecast Table</CardTitle>
                                <CardDescription>Projected values for the ML forecast horizon.</CardDescription>
                              </div>
                              <Badge variant="secondary">{result.period_label ?? 'Period'} horizon</Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="grid gap-3 sm:grid-cols-2">
                              <div className="rounded-2xl border bg-gradient-to-br from-secondary to-background p-4">
                                <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">First Period</p>
                                <p className="mt-2 text-base font-semibold">{result.future_forecast[0]?.period ?? 'N/A'}</p>
                              </div>
                              <div className="rounded-2xl border bg-gradient-to-br from-muted/30 to-background p-4">
                                <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">Last Period</p>
                                <p className="mt-2 text-base font-semibold">{result.future_forecast[result.future_forecast.length - 1]?.period ?? 'N/A'}</p>
                              </div>
                            </div>
                            <div className="rounded-2xl border bg-muted/20 p-4 dark:border-slate-800 dark:bg-slate-900/70">
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Horizon Avg</p>
                                  <p className="mt-1 font-semibold">{forecastSummary ? formatForecastValue(forecastSummary.average) : 'N/A'}</p>
                                </div>
                                <div>
                                  <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Peak Value</p>
                                  <p className="mt-1 font-semibold">{forecastSummary ? formatForecastValue(forecastSummary.maxPoint.predicted) : 'N/A'}</p>
                                </div>
                              </div>
                            </div>
                            <div className="max-h-[380px] overflow-auto rounded-2xl border">
                              <Table>
                                <TableHeader className="sticky top-0 bg-background">
                                  <TableRow>
                                    <TableHead>Period</TableHead>
                                    <TableHead className="text-right">Forecast</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {result.future_forecast.map((point) => (
                                    <TableRow key={point.period}>
                                      <TableCell className="font-medium">{point.period}</TableCell>
                                      <TableCell className="text-right">{formatForecastValue(point.predicted)}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>

                    <Card>
                      <CardHeader>
                        <div className="flex items-center gap-2">
                          <Table2 className="h-4 w-4 text-primary" />
                          <CardTitle>Generated Forecast Features Table</CardTitle>
                        </div>
                        <CardDescription>Sample rows of the engineered inputs used for model fitting and explainability.</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              {Object.keys(result.feature_preview_rows[0] ?? {}).map((key) => <TableHead key={key}>{key}</TableHead>)}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.feature_preview_rows.map((row, rowIndex) => (
                              <TableRow key={`feature-row-${rowIndex}`}>
                                {Object.entries(row).map(([key, value]) => <TableCell key={`${rowIndex}-${key}`}>{value === null ? 'N/A' : String(value)}</TableCell>)}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Forecast Insight</CardTitle>
                        <CardDescription>How the generated features shaped the prediction.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <p className="text-sm leading-6 text-muted-foreground">{result.analysis}</p>
                        <div className="flex justify-end gap-2">
                          <Button variant="outline" onClick={() => setCurrentStep(2)}>Try Another ML Forecast Model</Button>
                          <Button onClick={() => setActiveTab(modelTrained ? 'prediction' : 'ml')} className="gap-2">{modelTrained ? 'Continue To Prediction' : 'Continue To ML Assistant'}<ArrowRight className="h-4 w-4" /></Button>
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
