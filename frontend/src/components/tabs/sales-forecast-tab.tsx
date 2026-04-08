'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { type ColumnInfo, type DataRow, type SalesForecastResult, useAppStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertCircle, ArrowRight, BrainCircuit, CalendarDays, CheckCircle2, ChevronLeft, Gauge, LineChart as LineChartIcon, Loader2, ShieldCheck, Sparkles, TrendingUp, Zap } from 'lucide-react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, YAxis } from 'recharts';
import { apiClient, getApiErrorMessage } from '@/lib/api';

type ForecastFeatureGroupId = 'trend' | 'calendar' | 'seasonality' | 'lags' | 'rolling';
type ForecastModelCard = { model_type: string; model_name: string; explainability_level: 'High' | 'Medium'; fitLabel: string; reason: string; explainability: string; recommended: boolean };
type ForecastResultView = SalesForecastResult & { model_details?: { model_type: string; model_name: string; recommendation_reason?: string; rationale?: string; explainability?: string; explainability_level?: string; feature_groups?: string[] }; feature_importance?: { name: string; importance: number }[]; recommended_models?: { model_type: string; model_name: string; recommended?: boolean; recommendation_reason?: string; explainability?: string; explainability_level?: string; rationale?: string }[] };

const FEATURE_GROUPS: { id: ForecastFeatureGroupId; label: string; description: string }[] = [
  { id: 'trend', label: 'Trend index', description: 'Tracks long-run growth or decline.' },
  { id: 'calendar', label: 'Calendar markers', description: 'Uses month, quarter, and day timing.' },
  { id: 'seasonality', label: 'Seasonality cycles', description: 'Adds cyclical seasonal signals.' },
  { id: 'lags', label: 'Recent lag values', description: 'Uses the latest observed sales periods directly.' },
  { id: 'rolling', label: 'Rolling averages', description: 'Summarizes recent momentum with moving averages.' },
];

const STEPS = [
  { step: 1, label: 'Target & Features', icon: Sparkles },
  { step: 2, label: 'Model Choice', icon: BrainCircuit },
  { step: 3, label: 'Train Forecast', icon: Zap },
  { step: 4, label: 'Forecast Output', icon: TrendingUp },
];

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

  if (points.length < 2) return { periods: points.length, periodLabel: 'period', volatility: 0, zeroShare: 0 };

  const values = points.map((item) => item.value);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const diffs = points.slice(1).map((item, index) => (item.date.getTime() - points[index].date.getTime()) / 86400000).sort((a, b) => a - b);
  const medianDays = diffs[Math.floor(diffs.length / 2)] ?? 30;
  const periodLabel = medianDays <= 2 ? 'day' : medianDays <= 10 ? 'week' : medianDays <= 45 ? 'month' : medianDays <= 120 ? 'quarter' : 'year';

  return {
    periods: points.length,
    periodLabel,
    volatility: mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean),
    zeroShare: values.filter((value) => value === 0).length / values.length,
  };
}

function buildFeaturePreview(featureGroups: ForecastFeatureGroupId[], lagPeriods: number) {
  const preview: string[] = [];
  if (featureGroups.includes('trend')) preview.push('trend_index');
  if (featureGroups.includes('calendar')) preview.push('month_number', 'quarter_number', 'day_of_month', 'day_of_week');
  if (featureGroups.includes('seasonality')) preview.push('month_sin', 'month_cos', 'quarter_sin', 'quarter_cos');
  if (featureGroups.includes('rolling')) preview.push('lag_mean', 'lag_last_3_mean');
  if (featureGroups.includes('lags')) for (let lag = 1; lag <= lagPeriods; lag += 1) preview.push(`lag_${lag}`);
  return preview;
}

function buildModelRecommendations(profile: ReturnType<typeof inferSeriesProfile>, featureGroups: ForecastFeatureGroupId[]): ForecastModelCard[] {
  const withLags = featureGroups.includes('lags');
  const models: Record<string, Omit<ForecastModelCard, 'recommended'>> = {
    linear_regression: { model_type: 'linear_regression', model_name: 'Time-Series Linear Regression', explainability_level: 'High', fitLabel: 'Stable baseline', reason: `Good when ${profile.periodLabel}-level sales are smooth and interpretable.${withLags ? ' Lag features are enabled, so it can lean on recent momentum.' : ''}`, explainability: 'Most transparent option with direct trend and seasonality interpretation.' },
    ridge_regression: { model_type: 'ridge_regression', model_name: 'Time-Series Ridge Regression', explainability_level: 'High', fitLabel: 'Robust with correlated lags', reason: 'Useful when multiple lag and calendar signals overlap and need a steadier fit.', explainability: 'Still highly explainable, but more stable when features are correlated.' },
    elasticnet: { model_type: 'elasticnet', model_name: 'Time-Series Elastic Net', explainability_level: 'Medium', fitLabel: 'Sparse signal finder', reason: 'Helps when only a subset of lag and seasonal signals should stay active.', explainability: 'Explains the forecast through the strongest retained coefficients.' },
    random_forest: { model_type: 'random_forest', model_name: 'Time-Series Random Forest', explainability_level: 'Medium', fitLabel: 'Nonlinear pattern learner', reason: 'Helpful when recent sales shifts and calendar effects interact in nonlinear ways.', explainability: 'Explained mainly through feature importance, not direct coefficients.' },
    gradient_boosting: { model_type: 'gradient_boosting', model_name: 'Time-Series Gradient Boosting', explainability_level: 'Medium', fitLabel: 'Changing momentum', reason: 'Strong option when momentum and seasonality combine in a more complex way.', explainability: 'Explains forecast behavior through feature importance and sequential sensitivity.' },
  };

  let order = ['ridge_regression', 'linear_regression', 'gradient_boosting', 'random_forest', 'elasticnet'];
  if (profile.periods >= 24 && profile.volatility > 0.35) order = ['gradient_boosting', 'random_forest', 'ridge_regression', 'linear_regression', 'elasticnet'];
  else if (profile.zeroShare > 0.15) order = ['random_forest', 'gradient_boosting', 'ridge_regression', 'linear_regression', 'elasticnet'];
  else if (profile.periods < 12) order = ['ridge_regression', 'linear_regression', 'elasticnet', 'gradient_boosting', 'random_forest'];

  return order.map((modelType, index) => ({ ...models[modelType], recommended: index === 0 }));
}

export default function SalesForecastTab() {
  const { toast } = useToast();
  const cleanedData = useAppStore((state) => state.cleanedData);
  const rawData = useAppStore((state) => state.rawData);
  const columns = useAppStore((state) => state.columns);
  const fileName = useAppStore((state) => state.fileName);
  const modelTrained = useAppStore((state) => state.modelTrained);
  const setActiveTab = useAppStore((state) => state.setActiveTab);
  const storedResult = useAppStore((state) => state.salesForecastResult);
  const data = cleanedData ?? rawData ?? [];
  const isParquetDataset = (fileName ?? '').toLowerCase().endsWith('.parquet');

  const numericColumns = useMemo(() => columns.filter((column) => column.role === 'numeric'), [columns]);
  const dateColumns = useMemo(() => columns.filter((column) => column.role === 'datetime' || /date|month|time|period/i.test(column.name)), [columns]);

  const [currentStep, setCurrentStep] = useState(1);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(3);
  const [trainSplitPercent, setTrainSplitPercent] = useState(80);
  const [lagPeriods, setLagPeriods] = useState(3);
  const [featureGroups, setFeatureGroups] = useState<ForecastFeatureGroupId[]>(['trend', 'calendar', 'seasonality', 'lags', 'rolling']);
  const [selectedModelType, setSelectedModelType] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [result, setResult] = useState<ForecastResultView | null>((storedResult as ForecastResultView | null) ?? null);

  useEffect(() => {
    if (!dateColumn) setDateColumn(getPreferredDateColumn(columns));
    if (!targetColumn) setTargetColumn(getPreferredSalesColumn(columns));
  }, [columns, dateColumn, targetColumn]);

  useEffect(() => {
    if (storedResult) {
      setResult(storedResult as ForecastResultView);
      setCurrentStep(4);
    }
  }, [storedResult]);

  const profile = useMemo(() => inferSeriesProfile(data as DataRow[], dateColumn, targetColumn), [data, dateColumn, targetColumn]);
  const featurePreview = useMemo(() => buildFeaturePreview(featureGroups, lagPeriods), [featureGroups, lagPeriods]);
  const modelRecommendations = useMemo(() => buildModelRecommendations(profile, featureGroups), [profile, featureGroups]);
  const selectedModel = modelRecommendations.find((model) => model.model_type === selectedModelType) ?? modelRecommendations[0] ?? null;

  useEffect(() => {
    if (!selectedModelType && modelRecommendations.length > 0) setSelectedModelType(modelRecommendations[0].model_type);
  }, [modelRecommendations, selectedModelType]);

  const chartData = useMemo(() => {
    if (!result) return [];
    const testMap = new Map(result.test_forecast.map((item) => [item.period, item]));
    const futureMap = new Map(result.future_forecast.map((item) => [item.period, item]));
    return [...result.history.map((item) => ({ period: item.period, actual: item.actual, backtest: testMap.get(item.period)?.predicted ?? null, forecast: null as number | null })), ...result.future_forecast.map((item) => ({ period: item.period, actual: null, backtest: null, forecast: item.predicted }))].map((item) => ({ ...item, forecast: futureMap.get(item.period)?.predicted ?? item.forecast }));
  }, [result]);

  const toggleFeatureGroup = (featureGroup: ForecastFeatureGroupId) => {
    setFeatureGroups((previous) => previous.includes(featureGroup) ? (previous.length === 1 ? previous : previous.filter((item) => item !== featureGroup)) : [...previous, featureGroup]);
  };

  const goToModelChoice = () => {
    if (!dateColumn || !targetColumn) {
      toast({ title: 'Missing columns', description: 'Choose both a date column and a sales target first.', variant: 'destructive' });
      return;
    }
    setCurrentStep(2);
  };

  const handleTrainForecast = async () => {
    if (!dateColumn || !targetColumn || !selectedModel) {
      toast({ title: 'Forecast setup incomplete', description: 'Complete the target, feature, and model steps first.', variant: 'destructive' });
      return;
    }

    setIsTraining(true);
    setResult(null);
    setCurrentStep(3);

    try {
      const payload = {
        data,
        date_column: dateColumn,
        target_column: targetColumn,
        forecast_periods: forecastPeriods,
        test_percentage: 100 - trainSplitPercent,
        lag_periods: lagPeriods,
        model_type: selectedModel.model_type,
        feature_groups: featureGroups,
      };

      const response = await apiClient.post('/sales-forecast', payload);
      const forecastResult = response.data as ForecastResultView;
      const enrichedResult: ForecastResultView = {
        ...forecastResult,
        model_details: forecastResult.model_details ?? {
          model_type: selectedModel.model_type,
          model_name: selectedModel.model_name,
          recommendation_reason: selectedModel.recommended ? 'Recommended from the current dataset profile.' : 'Selected by the user from the recommended models.',
          rationale: selectedModel.reason,
          explainability: selectedModel.explainability,
          explainability_level: selectedModel.explainability_level,
          feature_groups: featureGroups,
        },
        recommended_models: forecastResult.recommended_models ?? modelRecommendations.map((model) => ({
          model_type: model.model_type,
          model_name: model.model_name,
          recommended: model.recommended,
          recommendation_reason: model.recommended ? 'Recommended from the current dataset profile.' : 'Alternative recommended model.',
          explainability: model.explainability,
          explainability_level: model.explainability_level,
          rationale: model.reason,
        })),
      };
      setResult(enrichedResult);
      useAppStore.setState({ salesForecastResult: enrichedResult });
      setCurrentStep(4);
      toast({ title: 'Forecast ready', description: `Projected ${forecastPeriods} future ${forecastResult.period_label ?? 'period'}${forecastPeriods === 1 ? '' : 's'}.` });
    } catch (error) {
      const message = getApiErrorMessage(error, 'Sales forecasting failed');
      toast({ title: 'Forecast failed', description: message, variant: 'destructive' });
      setCurrentStep(2);
    } finally {
      setIsTraining(false);
    }
  };

  if (!data.length || !columns.length) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Sales Forecast</h2>
          <p className="text-muted-foreground mt-1">Project future sales from your historical time-based sales data.</p>
        </div>
        <Card className="border-dashed">
          <CardContent className="py-14 flex flex-col items-center text-center gap-3">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <div>
              <p className="font-medium">Upload and clean a dataset first</p>
              <p className="text-sm text-muted-foreground mt-1">This tab needs historical data before it can build a forecast.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Sales Forecast</h2>
        <p className="text-muted-foreground mt-1">Follow a guided forecasting workflow: choose the target and forecast features, review recommended models, train the selected strategy, and project future sales.</p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center gap-4">
            {STEPS.map((step) => {
              const Icon = step.icon;
              const isActive = currentStep === step.step;
              const isDone = currentStep > step.step;
              return (
                <div key={step.step} className="flex items-center gap-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold ${isActive ? 'bg-emerald-500 text-white' : isDone ? 'bg-emerald-100 text-emerald-700' : 'bg-muted text-muted-foreground'}`}>
                    {isDone ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
                  </div>
                  <span className={`text-sm font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>{step.label}</span>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {currentStep === 1 && (
        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Step 1: Target And Features</CardTitle>
              <CardDescription>Pick the time column and sales target, then choose the engineered forecast features the model should use.</CardDescription>
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
                    <SelectTrigger><SelectValue placeholder="Select sales column" /></SelectTrigger>
                    <SelectContent>{numericColumns.map((column) => <SelectItem key={column.name} value={column.name}>{column.name}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2"><Label>Future Periods</Label><Input type="number" min={1} max={24} value={forecastPeriods} onChange={(event) => setForecastPeriods(Math.max(1, Math.min(24, Number(event.target.value) || 1)))} /></div>
                <div className="space-y-2"><Label>Training Split (%)</Label><Input type="number" min={50} max={90} value={trainSplitPercent} onChange={(event) => setTrainSplitPercent(Math.max(50, Math.min(90, Number(event.target.value) || 80)))} /></div>
                <div className="space-y-2"><Label>Lag Periods</Label><Input type="number" min={1} max={12} value={lagPeriods} onChange={(event) => setLagPeriods(Math.max(1, Math.min(12, Number(event.target.value) || 1)))} /></div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {FEATURE_GROUPS.map((featureGroup) => (
                  <label key={featureGroup.id} className="flex items-start gap-3 rounded-xl border p-4 cursor-pointer hover:border-emerald-300 transition-colors">
                    <Checkbox checked={featureGroups.includes(featureGroup.id)} onCheckedChange={() => toggleFeatureGroup(featureGroup.id)} />
                    <div>
                      <p className="text-sm font-medium">{featureGroup.label}</p>
                      <p className="text-sm text-muted-foreground mt-1">{featureGroup.description}</p>
                    </div>
                  </label>
                ))}
              </div>
              <div className="rounded-xl border bg-muted/20 p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium">Generated Forecast Features</p>
                    <p className="text-sm text-muted-foreground mt-1">These are the engineered inputs for forecasting.</p>
                  </div>
                  <Badge variant="secondary">{featurePreview.length} features</Badge>
                </div>
                <div className="flex flex-wrap gap-2">{featurePreview.map((feature) => <Badge key={feature} variant="outline">{feature}</Badge>)}</div>
              </div>
              <div className="flex justify-end"><Button onClick={goToModelChoice} className="gap-2">Next: Recommended Models <ArrowRight className="h-4 w-4" /></Button></div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Dataset Profile</CardTitle><CardDescription>Quick signals that shape the recommended forecast models.</CardDescription></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Detected Frequency</p><p className="mt-2 text-lg font-semibold capitalize">{profile.periodLabel}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Usable Periods</p><p className="mt-2 text-lg font-semibold">{profile.periods}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Volatility</p><p className="mt-2 text-lg font-semibold">{profile.volatility.toFixed(2)}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Zero-Value Share</p><p className="mt-2 text-lg font-semibold">{(profile.zeroShare * 100).toFixed(1)}%</p></div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      {currentStep === 2 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold">Step 2: Pick A Forecast Model</h3>
              <p className="text-sm text-muted-foreground mt-1">These models are ranked from the current dataset profile, forecast horizon, and selected feature groups.</p>
            </div>
            <Button variant="outline" onClick={() => setCurrentStep(1)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            {modelRecommendations.map((model) => {
              const isSelected = selectedModelType === model.model_type;
              return (
                <button key={model.model_type} type="button" onClick={() => setSelectedModelType(model.model_type)} className={`rounded-2xl border p-5 text-left transition-all ${isSelected ? 'border-emerald-500 bg-emerald-50/70 shadow-md' : 'hover:border-emerald-300 bg-background'}`}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-base font-semibold">{model.model_name}</p>
                        {model.recommended && <Badge className="bg-emerald-600 text-white">Best Match</Badge>}
                        <Badge variant="outline">{model.fitLabel}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">{model.reason}</p>
                    </div>
                    <Badge variant={model.explainability_level === 'High' ? 'secondary' : 'outline'}>{model.explainability_level} explainability</Badge>
                  </div>
                  <div className="mt-4 rounded-xl border bg-muted/20 p-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2 font-medium text-foreground"><ShieldCheck className="h-4 w-4 text-emerald-600" />Explainability</div>
                    <p className="mt-2 leading-6">{model.explainability}</p>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="flex justify-end"><Button onClick={() => setCurrentStep(3)} className="gap-2" disabled={!selectedModelType}>Next: Train Forecast <ArrowRight className="h-4 w-4" /></Button></div>
        </div>
      )}

      {currentStep === 3 && (
        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <Card>
            <CardHeader>
              <CardTitle>Step 3: Train The Selected Forecast Model</CardTitle>
              <CardDescription>Review the chosen setup, train the model, then generate future sales forecasts from the learned time-series pattern.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Date Column</p><p className="mt-2 text-base font-semibold">{dateColumn}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Sales Target</p><p className="mt-2 text-base font-semibold">{targetColumn}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Selected Model</p><p className="mt-2 text-base font-semibold">{selectedModel?.model_name ?? 'Not selected'}</p></div>
                <div className="rounded-xl border p-4"><p className="text-xs uppercase tracking-wide text-muted-foreground">Forecast Horizon</p><p className="mt-2 text-base font-semibold">{forecastPeriods} future {profile.periodLabel}{forecastPeriods === 1 ? '' : 's'}</p></div>
              </div>
              <div className="rounded-xl border bg-muted/20 p-4">
                <div className="flex items-center gap-2 font-medium"><Gauge className="h-4 w-4 text-emerald-600" />Why this model is selected</div>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">{selectedModel?.reason}</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{selectedModel?.explainability}</p>
              </div>
              <div className="flex flex-wrap gap-2">{featurePreview.map((feature) => <Badge key={feature} variant="outline">{feature}</Badge>)}</div>
              {isTraining && <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">{isParquetDataset ? 'Preparing the cached parquet history and generating the future sales forecast. This can take a little longer on large files.' : 'Preparing the time-series history and generating the future sales forecast.'}</div>}
              <div className="flex justify-between gap-3">
                <Button variant="outline" onClick={() => setCurrentStep(2)} className="gap-2"><ChevronLeft className="h-4 w-4" />Previous</Button>
                <Button onClick={handleTrainForecast} disabled={isTraining || !selectedModel} className="gap-2">{isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}{isTraining ? 'Training Forecast...' : 'Train And Forecast'}</Button>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Forecast Explainability Preview</CardTitle><CardDescription>How the trained forecast will be easiest to explain once the run completes.</CardDescription></CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <div className="rounded-xl border p-4"><p className="font-medium text-foreground">Primary explanation style</p><p className="mt-2 leading-6">{selectedModel?.explainability}</p></div>
              <div className="rounded-xl border p-4"><p className="font-medium text-foreground">Expected forecast drivers</p><p className="mt-2 leading-6">The trained model will explain forecasts mainly through the selected lag, trend, and seasonal features shown in the setup step.</p></div>
            </CardContent>
          </Card>
        </div>
      )}

      {currentStep === 4 && result && (
        <>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">Model: {result.model_details?.model_name ?? result.training_summary.model_name}</Badge>
              <Badge variant="outline">Detected frequency: {result.period_label ?? 'period'}</Badge>
              <Badge variant="outline">Split: {result.training_summary.train_percentage}% train / {result.training_summary.test_percentage}% test</Badge>
              <Badge variant="outline">Forecast horizon: {result.training_summary.forecast_periods} {result.period_label ?? 'period'}{result.training_summary.forecast_periods === 1 ? '' : 's'}</Badge>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setCurrentStep(2)}>Train Another Model</Button>
              <Button onClick={() => setActiveTab(modelTrained ? 'prediction' : 'ml')} className="gap-2">{modelTrained ? 'Proceed To Prediction' : 'Continue To ML Assistant'}<ArrowRight className="h-4 w-4" /></Button>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card><CardHeader className="pb-2"><CardDescription>Training Window</CardDescription><CardTitle className="text-2xl">{result.training_summary.train_percentage}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground"><p>{result.training_summary.train_periods} {result.period_label ?? 'period'}{result.training_summary.train_periods === 1 ? '' : 's'}</p><p className="mt-1 font-medium text-foreground">{result.training_summary.train_start} to {result.training_summary.train_end}</p></CardContent></Card>
            <Card><CardHeader className="pb-2"><CardDescription>Backtest Window</CardDescription><CardTitle className="text-2xl">{result.training_summary.test_percentage}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground"><p>{result.training_summary.test_periods} {result.period_label ?? 'period'}{result.training_summary.test_periods === 1 ? '' : 's'}</p><p className="mt-1 font-medium text-foreground">{result.training_summary.test_start} to {result.training_summary.test_end}</p></CardContent></Card>
            <Card><CardHeader className="pb-2"><CardDescription>MAE</CardDescription><CardTitle className="text-2xl">{result.metrics.mae.toLocaleString()}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average difference between actual and predicted sales during backtesting.</CardContent></Card>
            <Card><CardHeader className="pb-2"><CardDescription>MAPE</CardDescription><CardTitle className="text-2xl">{result.metrics.mape}%</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">Average percentage gap during the backtest periods.</CardContent></Card>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>Selected Model Explainability</CardTitle><CardDescription>How to explain this forecast run to business users.</CardDescription></CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p className="leading-6">{result.model_details?.rationale ?? selectedModel?.reason}</p>
                <p className="leading-6">{result.model_details?.explainability ?? selectedModel?.explainability}</p>
                <div className="flex flex-wrap gap-2">{(result.model_details?.feature_groups ?? featureGroups).map((featureGroup) => <Badge key={featureGroup} variant="secondary">{featureGroup}</Badge>)}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Top Forecast Drivers</CardTitle><CardDescription>The most important engineered inputs for this forecast configuration.</CardDescription></CardHeader>
              <CardContent className="space-y-3">{(result.feature_importance && result.feature_importance.length > 0 ? result.feature_importance.slice(0, 6) : featurePreview.slice(0, 6).map((feature) => ({ name: feature, importance: 0 }))).map((item, index) => <div key={item.name} className="rounded-xl border p-3"><div className="flex items-center justify-between gap-3 text-sm"><span className="font-medium">{index + 1}. {item.name}</span><Badge variant="outline">{typeof item.importance === 'number' ? item.importance.toFixed(3) : 'Guided'}</Badge></div></div>)}</CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Sales Forecast View</CardTitle><CardDescription>Actual history, backtest predictions for recent periods, and future forecast in one chart.</CardDescription></CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                    <XAxis dataKey="period" label={{ value: result.period_label ? `${result.period_label[0].toUpperCase()}${result.period_label.slice(1)}` : 'Period', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Sales', angle: -90, position: 'insideLeft' }} />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="actual" name="Actual Sales" stroke="#0f766e" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
                    <Line type="monotone" dataKey="backtest" name="Backtest Prediction" stroke="#f59e0b" strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3 }} isAnimationActive={false} />
                    <Line type="monotone" dataKey="forecast" name="Future Forecast" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader><div className="flex items-center gap-2"><CalendarDays className="h-4 w-4 text-emerald-600" /><CardTitle>Future Forecast</CardTitle></div><CardDescription>Projected sales for the upcoming periods.</CardDescription></CardHeader>
              <CardContent>
                <Table>
                  <TableHeader><TableRow><TableHead>{result.period_label ? `${result.period_label[0].toUpperCase()}${result.period_label.slice(1)}` : 'Period'}</TableHead><TableHead>Forecasted Sales</TableHead></TableRow></TableHeader>
                  <TableBody>{result.future_forecast.map((item) => <TableRow key={item.period}><TableCell>{item.period}</TableCell><TableCell className="font-medium">{item.predicted.toLocaleString()}</TableCell></TableRow>)}</TableBody>
                </Table>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Backtest Check</CardTitle><CardDescription>How the model performed on the most recent held-out periods.</CardDescription></CardHeader>
              <CardContent>
                <Table>
                  <TableHeader><TableRow><TableHead>{result.period_label ? `${result.period_label[0].toUpperCase()}${result.period_label.slice(1)}` : 'Period'}</TableHead><TableHead>Actual</TableHead><TableHead>Predicted</TableHead></TableRow></TableHeader>
                  <TableBody>{result.test_forecast.map((item) => <TableRow key={item.period}><TableCell>{item.period}</TableCell><TableCell>{item.actual.toLocaleString()}</TableCell><TableCell className="font-medium">{item.predicted.toLocaleString()}</TableCell></TableRow>)}</TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Forecast Insight</CardTitle><CardDescription>Plain-language summary of the time-series run.</CardDescription></CardHeader>
            <CardContent><p className="text-sm leading-6 text-muted-foreground">{result.analysis}</p></CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
