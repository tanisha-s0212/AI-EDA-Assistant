'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAppStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  BrainCircuit,
  Target,
  Sparkles,
  Send,
  CheckCircle2,
  AlertCircle,
  Loader2,
  BarChart3,
  TrendingUp,
  Hash,
  Tag,
  Gauge,
  Clock,
  ArrowRight,
  Trash2,
  Download,
  Upload,
  CalendarDays,
  LineChart as LineChartIcon,
  History,
  Zap,
  FileSpreadsheet,
  CircleDot,
  Activity,
  Bot,
  Layers,
  RefreshCw,
} from 'lucide-react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  Legend,
  ReferenceLine,
} from 'recharts';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { cn } from '@/lib/utils';

// ─── Types ─────────────────────────────────────────────────────────────────────

interface PredictResponse {
  prediction: number | string;
  prediction_label?: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  top_class?: string;
}

interface UploadModelResponse {
  model_id: string;
  model_name: string;
  model_type: string;
  problem_type: string;
  target_column: string;
  feature_columns: string[];
  trained_at: string;
  source_filename: string;
}

interface FeatureInput {
  name: string;
  role: string;
  dtype: string;
  value: string;
  uniqueValues: string[];
  min?: number;
  max?: number;
  mean?: number;
}

// ─── Animation Variants ───────────────────────────────────────────────────────

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: 'spring', stiffness: 300, damping: 30 },
  },
};

const scaleVariants: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { type: 'spring', stiffness: 300, damping: 25 },
  },
};

// ─── Helpers ───────────────────────────────────────────────────────────────────

function generateUUID(): string {
  return 'xxxx-xxxx-xxxx'.replace(/x/g, () => Math.floor(Math.random() * 16).toString(16));
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return iso;
  }
}

function getModelDisplayName(modelName: string): string {
  return modelName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatConfidence(value: number | undefined | null): string {
  if (value == null) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

function getConfidenceColor(value: number | undefined | null): string {
  if (value == null) return 'text-muted-foreground';
  if (value >= 0.8) return 'text-emerald-500';
  if (value >= 0.6) return 'text-amber-500';
  return 'text-rose-500';
}

function getConfidenceBg(value: number | undefined | null): string {
  if (value == null) return 'bg-muted';
  if (value >= 0.8) return 'bg-emerald-500/10 border-emerald-500/20';
  if (value >= 0.6) return 'bg-amber-500/10 border-amber-500/20';
  return 'bg-rose-500/10 border-rose-500/20';
}

function stringifyFeatureValue(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return '';
  return String(value);
}

function getSuggestedFeatureValue(data: Record<string, string | number | boolean | null>[], featureName: string): string {
  for (let index = data.length - 1; index >= 0; index--) {
    const value = data[index]?.[featureName];
    if (value !== null && value !== undefined && value !== '') {
      return stringifyFeatureValue(value);
    }
  }
  return '';
}


function findFeatureName(featureNames: string[], patterns: string[]): string | null {
  const lowerNames = featureNames.map((name) => ({ original: name, lower: name.toLowerCase() }));
  for (const pattern of patterns) {
    const match = lowerNames.find((entry) => entry.lower.includes(pattern));
    if (match) return match.original;
  }
  return null;
}

function formatDateInputValue(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function findMatchingFeatureNames(featureNames: string[], patterns: string[]): string[] {
  const loweredPatterns = patterns.map((pattern) => pattern.toLowerCase());
  return featureNames.filter((name) => {
    const lowered = name.toLowerCase();
    return loweredPatterns.some((pattern) => lowered.includes(pattern));
  });
}

function getPredictionDateFieldNames(featureNames: string[]): string[] {
  return findMatchingFeatureNames(featureNames, ['year_month', 'date', 'invoice_date', 'order_date', 'period', 'ds']);
}

function buildFeaturesFromInputs(featureInputs: FeatureInput[]): Record<string, string | number> {
  const features: Record<string, string | number> = {};
  for (const fi of featureInputs) {
    if (fi.role === 'numeric') {
      const num = parseFloat(fi.value);
      features[fi.name] = Number.isNaN(num) ? 0 : num;
    } else {
      features[fi.name] = fi.value;
    }
  }
  return features;
}

function applySelectedDateToFeatures(
  features: Record<string, string | number>,
  featureNames: string[],
  selectedDate: Date,
): Record<string, string | number> {
  const next = { ...features };
  const isoDate = formatDateInputValue(selectedDate);
  const monthNumber = selectedDate.getMonth() + 1;
  const dayOfWeek = selectedDate.getDay();
  const quarter = Math.floor(selectedDate.getMonth() / 3) + 1;
  const year = selectedDate.getFullYear();
  const weekendFlag = [0, 6].includes(dayOfWeek) ? 1 : 0;

  getPredictionDateFieldNames(featureNames).forEach((fieldName) => {
    next[fieldName] = isoDate;
  });

  findMatchingFeatureNames(featureNames, ['month_number', 'year_month_month']).forEach((fieldName) => {
    next[fieldName] = monthNumber;
  });

  findMatchingFeatureNames(featureNames, ['day_of_week', 'weekday']).forEach((fieldName) => {
    next[fieldName] = dayOfWeek;
  });

  findMatchingFeatureNames(featureNames, ['quarter']).forEach((fieldName) => {
    next[fieldName] = quarter;
  });

  findMatchingFeatureNames(featureNames, ['is_weekend', 'weekend']).forEach((fieldName) => {
    next[fieldName] = weekendFlag;
  });

  findMatchingFeatureNames(featureNames, ['year_number', 'year']).forEach((fieldName) => {
    next[fieldName] = year;
  });

  return next;
}

function applyForecastDerivations(
  features: Record<string, string | number>,
  featureNames: string[],
  data: Record<string, string | number | boolean | null>[],
  targetColumn: string | null,
): Record<string, string | number> {
  const next = { ...features };
  const dateFieldName = findFeatureName(featureNames, ['year_month', 'date', 'invoice_date', 'order_date']);
  const dateValue = dateFieldName ? next[dateFieldName] : null;
  const parsedDate = typeof dateValue === 'string' || typeof dateValue === 'number' ? new Date(dateValue) : null;

  if (parsedDate && !Number.isNaN(parsedDate.getTime())) {
    const monthNumberName = findFeatureName(featureNames, ['month_number']);
    const dayOfWeekName = findFeatureName(featureNames, ['day_of_week']);
    const quarterName = findFeatureName(featureNames, ['quarter']);
    const weekendName = findFeatureName(featureNames, ['is_weekend', 'weekend']);
    const derivedMonthName = findFeatureName(featureNames, ['year_month_month']);

    if (monthNumberName && (next[monthNumberName] === '' || next[monthNumberName] === undefined)) next[monthNumberName] = parsedDate.getMonth() + 1;
    if (derivedMonthName && (next[derivedMonthName] === '' || next[derivedMonthName] === undefined)) next[derivedMonthName] = parsedDate.getMonth() + 1;
    if (dayOfWeekName && (next[dayOfWeekName] === '' || next[dayOfWeekName] === undefined)) next[dayOfWeekName] = parsedDate.getDay();
    if (quarterName && (next[quarterName] === '' || next[quarterName] === undefined)) next[quarterName] = Math.floor(parsedDate.getMonth() / 3) + 1;
    if (weekendName && (next[weekendName] === '' || next[weekendName] === undefined)) next[weekendName] = [0, 6].includes(parsedDate.getDay()) ? 1 : 0;
  }

  const previousMonthField = findFeatureName(featureNames, ['previous_month', 'prev_month']);
  if (previousMonthField && (next[previousMonthField] === '' || next[previousMonthField] === undefined) && targetColumn) {
    for (let index = data.length - 1; index >= 0; index--) {
      const value = data[index]?.[targetColumn];
      if (typeof value === 'number' && !Number.isNaN(value)) {
        next[previousMonthField] = value;
        break;
      }
    }
  }

  return next;
}

function buildPredictionExplanation(
  predictionValue: string | number,
  confidence: number | undefined,
  processedFeatures: Record<string, string | number>,
  targetColumn: string | null,
  featureImportance: { name: string; importance: number }[] | null,
  problemType: string,
): string {
  const lines: string[] = [];
  lines.push(`The model predicts ${predictionValue} for ${targetColumn ?? 'the selected target'}.`);

  if (confidence != null) {
    lines.push(`The confidence score is ${(confidence * 100).toFixed(1)}%, which indicates how strongly the model favors this output.`);
  }

  const topFeatures = (featureImportance ?? []).slice(0, 3);
  if (topFeatures.length > 0) {
    const summary = topFeatures
      .map((item) => {
        const value = processedFeatures[item.name];
        return value !== undefined ? `${item.name}=${value}` : item.name;
      })
      .join(', ');
    lines.push(`Top influential features in this model include ${summary}.`);
  }

  const dateFeature = Object.entries(processedFeatures).find(([name]) => name.toLowerCase().includes('year_month') || name.toLowerCase().includes('date'));
  if (dateFeature) {
    lines.push(`Temporal context was included through ${dateFeature[0]}=${dateFeature[1]}, so calendar effects can influence the result.`);
  }

  const previousMonthFeature = Object.entries(processedFeatures).find(([name]) => name.toLowerCase().includes('previous_month') || name.toLowerCase().includes('prev_month'));
  if (previousMonthFeature) {
    lines.push(`Historical carry-forward information was used via ${previousMonthFeature[0]}=${previousMonthFeature[1]}.`);
  }

  lines.push(
    problemType === 'classification'
      ? 'Interpret this as the most likely class under the current inputs; review class probabilities for alternative outcomes.'
      : 'Interpret this as the expected numeric outcome under the current inputs; adjust key drivers and rerun prediction to test sensitivity.'
  );

  return lines.join(' ');
}

function splitPredictionAnalysis(analysis: string | null) {
  if (!analysis) {
    return { lead: '', points: [] as string[] };
  }

  const sentences = analysis
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  return {
    lead: sentences[0] ?? '',
    points: sentences.slice(1),
  };
}

function EmptyState() {
  const { setActiveTab } = useAppStore();
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-20"
    >
      <div className="relative mb-6">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10">
          <BrainCircuit className="h-10 w-10 text-emerald-500" />
        </div>
        <motion.div
          className="absolute -top-1 -right-1 h-6 w-6 rounded-full bg-rose-500/20 flex items-center justify-center"
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <AlertCircle className="h-3.5 w-3.5 text-rose-500" />
        </motion.div>
      </div>
      <h3 className="text-xl font-semibold mb-2">No Model Available</h3>
      <p className="text-sm text-muted-foreground mb-2 text-center max-w-md">
        You need to train a machine learning model before making predictions.
        Go to the ML Assistant tab to configure and train your model.
      </p>
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-6">
        <Zap className="h-3.5 w-3.5 text-amber-500" />
        <span>Train a model in 6 simple steps with AI guidance</span>
      </div>
      <Button
        className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-lg shadow-emerald-500/20"
        onClick={() => setActiveTab('ml')}
      >
        Go to ML Assistant
        <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </motion.div>
  );
}

// ─── Model Status Card (Section 1) ─────────────────────────────────────────────

function UploadModelCard({
  onUploadClick,
  isUploading,
}: {
  onUploadClick: () => void;
  isUploading: boolean;
}) {
  return (
    <motion.div variants={itemVariants}>
      <Card className="border-dashed border-emerald-500/30 bg-gradient-to-br from-emerald-500/5 to-teal-500/5">
        <CardContent className="p-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 shrink-0">
              <Upload className="h-5 w-5 text-emerald-500" />
            </div>
            <div>
              <p className="text-sm font-medium">Upload a trained model</p>
              <p className="text-xs text-muted-foreground mt-1">
                Import a `.joblib`, `.pkl`, or `.pickle` sklearn model to predict without retraining in this app.
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            onClick={onUploadClick}
            disabled={isUploading}
            className="border-emerald-500/30 text-emerald-600 dark:text-emerald-400"
          >
            {isUploading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Upload className="mr-2 h-4 w-4" />
            )}
            {isUploading ? 'Uploading...' : 'Upload Model File'}
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function ModelStatusCard() {
  const { uploadedModel, modelMetrics, modelTrained, setActiveTab } = useAppStore();

  if (!uploadedModel) {
    return (
      <motion.div variants={itemVariants}>
        <Card className="border-rose-500/20 bg-gradient-to-r from-rose-500/5 to-transparent">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/10 shrink-0">
              <AlertCircle className="h-5 w-5 text-rose-500" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">No Model Loaded</p>
              <p className="text-xs text-muted-foreground">Train a model in the ML tab to get started</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-rose-500/30 text-rose-600 dark:text-rose-400 hover:bg-rose-500/10"
              onClick={() => setActiveTab('ml')}
            >
              <ArrowRight className="mr-1 h-3.5 w-3.5" />
              ML Tab
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  const primaryMetric = uploadedModel.problem === 'regression' ? 'R²' : 'Accuracy';
  const metricValue = modelMetrics?.[primaryMetric] ?? modelMetrics?.['R2'] ?? modelMetrics?.['r2'] ?? null;

  return (
    <motion.div variants={itemVariants}>
      <Card className="border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-teal-500/5 overflow-hidden">
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            {/* Model Info */}
            <div className="flex items-center gap-3 flex-1">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 shrink-0">
                <BrainCircuit className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-sm font-semibold truncate">
                    {getModelDisplayName(uploadedModel.name)}
                  </h3>
                  <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20">
                    <CheckCircle2 className="mr-1 h-3 w-3" />
                    Model Ready
                  </Badge>
                </div>
                <div className="flex items-center gap-3 mt-1 flex-wrap">
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Tag className="h-3 w-3" />
                    {uploadedModel.problem === 'regression' ? 'Regression' : 'Classification'}
                  </span>
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Target className="h-3 w-3" />
                    {uploadedModel.target}
                  </span>
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatTime(uploadedModel.trainedAt)}
                  </span>
                </div>
              </div>
            </div>

            {/* Key Metric */}
            {metricValue != null && (
              <div className={cn('flex items-center gap-2 px-4 py-2 rounded-xl border', getConfidenceBg(metricValue))}>
                <Gauge className={cn('h-4 w-4', getConfidenceColor(metricValue))} />
                <div>
                  <p className="text-xs text-muted-foreground">{primaryMetric}</p>
                  <p className={cn('text-lg font-bold leading-tight', getConfidenceColor(metricValue))}>
                    {metricValue >= 1 ? metricValue.toFixed(2) : (metricValue * 100).toFixed(1) + '%'}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Features Summary */}
          {uploadedModel.features && uploadedModel.features.length > 0 && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Layers className="h-3.5 w-3.5 text-emerald-500" />
                <span className="text-xs font-medium text-muted-foreground">
                  {uploadedModel.features.length} Feature{uploadedModel.features.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {uploadedModel.features.slice(0, 8).map((f) => (
                  <Badge key={f} variant="outline" className="text-[10px] px-1.5 py-0">
                    {f}
                  </Badge>
                ))}
                {uploadedModel.features.length > 8 && (
                  <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                    +{uploadedModel.features.length - 8} more
                  </Badge>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Prediction Input Form (Section 2) ─────────────────────────────────────────

function PredictionForm({
  featureInputs,
  setFeatureInputs,
  onPredict,
  onAutoFill,
  isPredicting,
}: {
  featureInputs: FeatureInput[];
  setFeatureInputs: React.Dispatch<React.SetStateAction<FeatureInput[]>>;
  onPredict: () => void;
  onAutoFill: () => void;
  isPredicting: boolean;
}) {
  const handleInputChange = (idx: number, value: string) => {
    setFeatureInputs((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], value };
      return next;
    });
  };

  return (
    <motion.div variants={itemVariants}>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10">
              <Target className="h-4 w-4 text-emerald-500" />
            </div>
            <div>
              <CardTitle className="text-base">Prediction Input</CardTitle>
              <CardDescription>Inputs are auto-filled from the latest available dataset row. Adjust any values before predicting.</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {featureInputs.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No feature columns available for prediction.
            </p>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {featureInputs.map((fi, idx) => (
                  <motion.div
                    key={fi.name}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.03 }}
                    className="space-y-1.5"
                  >
                    <Label className="text-xs font-medium flex items-center gap-1.5">
                      {fi.role === 'numeric' ? (
                        <Hash className="h-3 w-3 text-emerald-500" />
                      ) : (
                        <Tag className="h-3 w-3 text-teal-500" />
                      )}
                      {fi.name}
                    </Label>
                    {fi.role === 'numeric' ? (
                      <Input
                        type="number"
                        step="any"
                        placeholder={`e.g. ${fi.mean != null ? fi.mean.toFixed(1) : '0'}`}
                        value={fi.value}
                        onChange={(e) => handleInputChange(idx, e.target.value)}
                        className="h-9 text-sm"
                      />
                    ) : fi.uniqueValues.length > 0 ? (
                      <Select value={fi.value} onValueChange={(v) => handleInputChange(idx, v)}>
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue placeholder="Select..." />
                        </SelectTrigger>
                        <SelectContent className="max-h-48">
                          {fi.uniqueValues.map((uv) => (
                            <SelectItem key={uv} value={uv}>
                              {uv}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <Input
                        type="text"
                        placeholder="Enter value"
                        value={fi.value}
                        onChange={(e) => handleInputChange(idx, e.target.value)}
                        className="h-9 text-sm"
                      />
                    )}
                    {fi.role === 'numeric' && fi.min != null && fi.max != null && (
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <BarChart3 className="h-2.5 w-2.5" />
                        Range: {fi.min} — {fi.max} {fi.mean != null && `| Avg: ${fi.mean.toFixed(1)}`}
                      </p>
                    )}
                    {fi.role !== 'numeric' && fi.uniqueValues.length > 0 && (
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Layers className="h-2.5 w-2.5" />
                        {fi.uniqueValues.length} unique values
                      </p>
                    )}
                  </motion.div>
                ))}
              </div>

              <div className="flex flex-col items-center gap-3 pt-2 sm:flex-row sm:justify-center">
                <Button
                  variant="outline"
                  onClick={onAutoFill}
                  disabled={isPredicting}
                  className="border-emerald-500/30 text-emerald-600 dark:text-emerald-400"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Auto-Fill From Dataset
                </Button>
                <Button
                  size="lg"
                  className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white shadow-lg shadow-emerald-500/20 px-8"
                  onClick={onPredict}
                  disabled={isPredicting}
                >
                  <AnimatePresence mode="wait">
                    {isPredicting ? (
                      <motion.span
                        key="loading"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        className="flex items-center gap-2"
                      >
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Predicting...
                      </motion.span>
                    ) : (
                      <motion.span
                        key="idle"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        className="flex items-center gap-2"
                      >
                        <Send className="h-4 w-4" />
                        Predict
                      </motion.span>
                    )}
                  </AnimatePresence>
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Prediction Result Display (Section 3) ─────────────────────────────────────

function PredictionResult({
  result,
  analysis,
  probabilities,
  problemType,
}: {
  result: number | string | null;
  analysis: string | null;
  probabilities: Record<string, number> | null;
  problemType: string;
}) {
  if (result == null) {
    return (
      <motion.div variants={itemVariants}>
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-10">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted/50 mb-3">
              <Activity className="h-7 w-7 text-muted-foreground/40" />
            </div>
            <p className="text-sm font-medium text-muted-foreground">No Prediction Yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Enter feature values and click Predict to see results</p>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  const probsArray = probabilities
    ? Object.entries(probabilities).sort(([, a], [, b]) => b - a).map(([label, prob]) => ({ label, prob }))
    : [];

  const maxProb = probsArray.length > 0 ? probsArray[0].prob : null;
  const confidencePct = maxProb != null ? maxProb * 100 : null;
  const parsedAnalysis = splitPredictionAnalysis(analysis);
  const topProbability = probsArray[0] ?? null;

  return (
    <motion.div variants={itemVariants}>
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        >
          <Card className="border-emerald-500/20 overflow-hidden bg-gradient-to-br from-emerald-500/5 via-background to-teal-500/5">
            <div className="relative bg-gradient-to-r from-emerald-500/10 via-teal-500/10 to-transparent p-6">
              <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative text-center">
                <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-background/70 px-3 py-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                  <Sparkles className="h-3.5 w-3.5" />
                  Prediction Ready
                </div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Predicted Result</p>
                <motion.p
                  className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.1, type: 'spring' }}
                >
                  {typeof result === 'number' ? result.toLocaleString(undefined, { maximumFractionDigits: 4 }) : result}
                </motion.p>
              </div>
            </div>

            <CardContent className="space-y-5 p-6">
              <div className="grid gap-3 sm:grid-cols-3">
                <div className={cn('rounded-xl border p-4', getConfidenceBg(maxProb))}>
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Confidence</p>
                  <div className="mt-2 flex items-center gap-2">
                    <Gauge className={cn('h-4 w-4', getConfidenceColor(maxProb))} />
                    <p className={cn('text-xl font-bold', getConfidenceColor(maxProb))}>
                      {confidencePct != null ? `${confidencePct.toFixed(1)}%` : 'N/A'}
                    </p>
                  </div>
                </div>

                <div className="rounded-xl border bg-background/70 p-4">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Prediction Type</p>
                  <p className="mt-2 text-sm font-semibold capitalize">{problemType || 'Unknown'}</p>
                </div>

                <div className="rounded-xl border bg-background/70 p-4">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Top Outcome</p>
                  <p className="mt-2 text-sm font-semibold">
                    {topProbability ? `${topProbability.label} (${(topProbability.prob * 100).toFixed(1)}%)` : String(result)}
                  </p>
                </div>
              </div>

              {analysis && (
                <div className="rounded-xl border bg-background/80 p-4">
                  <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                    <Bot className="h-4 w-4" />
                    <p className="text-sm font-semibold">Prediction Analysis</p>
                  </div>
                  {parsedAnalysis.lead && (
                    <p className="mt-2 text-sm text-foreground leading-relaxed">{parsedAnalysis.lead}</p>
                  )}
                  {parsedAnalysis.points.length > 0 && (
                    <div className="mt-3 grid gap-3">
                      {parsedAnalysis.points.map((point, index) => (
                        <div key={index} className="flex items-start gap-2 rounded-lg bg-muted/40 p-3">
                          <div className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-500">
                            <Sparkles className="h-3.5 w-3.5" />
                          </div>
                          <p className="text-sm text-muted-foreground leading-relaxed">{point}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {problemType === 'classification' && probsArray.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <BarChart3 className="h-3.5 w-3.5" />
                    Class Probabilities
                  </p>
                  <div className="space-y-2.5">
                    {probsArray.map((item, idx) => (
                      <motion.div
                        key={item.label}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 + idx * 0.08 }}
                        className="space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{item.label}</span>
                          <span className={cn('text-sm font-semibold', getConfidenceColor(item.prob))}>
                            {(item.prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                          <motion.div
                            className={cn(
                              'h-full rounded-full',
                              item.prob === maxProb
                                ? 'bg-gradient-to-r from-emerald-500 to-teal-500'
                                : 'bg-emerald-500/40',
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.max(item.prob * 100, 1)}%` }}
                            transition={{ delay: 0.3 + idx * 0.08, duration: 0.6, ease: 'easeOut' }}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}

function FuturePrediction({
  featureInputs,
  setFeatureInputs,
  isPredicting,
}: {
  featureInputs: FeatureInput[];
  setFeatureInputs: React.Dispatch<React.SetStateAction<FeatureInput[]>>;
  isPredicting: boolean;
}) {
  const { toast } = useToast();
  const store = useAppStore();
  const data = store.cleanedData || store.rawData;
  const columns = store.columns;
  const modelId = store.modelId;
  const [numPeriods, setNumPeriods] = useState(5);
  const [forecastData, setForecastData] = useState<{ date: string; predicted: number | string; confidence: number | null }[]>([]);
  const [isForecasting, setIsForecasting] = useState(false);

  // Detect if dataset has a date column
  const dateColumn = useMemo(() => {
    return columns.find((c) => c.role === 'datetime');
  }, [columns]);

  const handleForecast = useCallback(async () => {
    if (!modelId || !data) return;
    setIsForecasting(true);
    setForecastData([]);

    try {
      const now = new Date();
      const results: { date: string; predicted: number | string; confidence: number | null }[] = [];

      for (let i = 1; i <= numPeriods; i++) {
        const futureDate = new Date(now);
        futureDate.setDate(futureDate.getDate() + i * 30);

        // Use rolling averages for numeric features, mode for categorical
        const features: Record<string, string | number> = {};
        for (const fi of featureInputs) {
          if (fi.role === 'numeric') {
            const numericValues = data
              .map((r) => r[fi.name])
              .filter((v): v is number => typeof v === 'number' && !isNaN(v));
            const avg = numericValues.length > 0
              ? numericValues.reduce((a, b) => a + b, 0) / numericValues.length
              : parseFloat(fi.value) || 0;
            features[fi.name] = parseFloat(avg.toFixed(4));
          } else {
            const stringValues = data.map((r) => String(r[fi.name] ?? ''));
            const freq: Record<string, number> = {};
            for (const sv of stringValues) {
              freq[sv] = (freq[sv] || 0) + 1;
            }
            const mode = Object.entries(freq).sort(([, a], [, b]) => b - a)[0]?.[0] || fi.value;
            features[fi.name] = mode;
          }
        }

        try {
          const response = await apiClient.post('/predict', { model_id: modelId, features });
          const pred: PredictResponse = response.data;
          results.push({
            date: futureDate.toISOString().split('T')[0],
            predicted: pred.prediction_label ?? pred.prediction,
            confidence: pred.confidence ?? null,
          });
        } catch {
          // Skip failed predictions
        }
      }

      setForecastData(results);
      toast({ title: 'Forecast Complete', description: `Generated ${results.length} period predictions.` });
    } catch (error) {
      const msg = getApiErrorMessage(error, 'Forecast failed');
      toast({ title: 'Forecast Error', description: msg, variant: 'destructive' });
    } finally {
      setIsForecasting(false);
    }
  }, [modelId, data, featureInputs, numPeriods, toast]);

  // Chart data
  const chartData = forecastData.map((item) => ({
    date: item.date,
    predicted: typeof item.predicted === 'number' ? item.predicted : null,
    label: typeof item.predicted === 'string' ? item.predicted : null,
  }));

  if (!dateColumn) return null;

  return (
    <motion.div variants={itemVariants}>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-500/10">
              <LineChartIcon className="h-4 w-4 text-teal-500" />
            </div>
            <div>
              <CardTitle className="text-base">Predict Future</CardTitle>
              <CardDescription>Project future values using rolling averages from historical data</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Controls */}
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1.5">
              <Label className="text-xs">Number of Periods</Label>
              <Select value={String(numPeriods)} onValueChange={(v) => setNumPeriods(parseInt(v, 10))}>
                <SelectTrigger className="w-32 h-9 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[3, 5, 10, 15, 20].map((n) => (
                    <SelectItem key={n} value={String(n)}>{n} periods</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              className="bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 text-white"
              onClick={handleForecast}
              disabled={isForecasting || !modelId}
            >
              {isForecasting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="mr-2 h-4 w-4" />
              )}
              Generate Forecast
            </Button>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CalendarDays className="h-3.5 w-3.5" />
            <span>Date column detected: <strong>{dateColumn.name}</strong></span>
            <span className="text-muted-foreground/40">|</span>
            <span>Each period ≈ 1 month from today</span>
          </div>

          {/* Chart (Regression) */}
          {store.problemType === 'regression' && chartData.length > 0 && (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <RechartsTooltip
                    contentStyle={{
                      borderRadius: '8px',
                      border: '1px solid rgba(0,0,0,0.1)',
                      fontSize: '12px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#059669"
                    strokeWidth={2.5}
                    dot={{ fill: '#059669', r: 4 }}
                    activeDot={{ r: 6, fill: '#0d9488' }}
                    name="Predicted"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Forecast Table */}
          {forecastData.length > 0 && (
            <div className="max-h-64 overflow-y-auto rounded-lg border custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    <TableHead className="text-xs font-semibold">Date</TableHead>
                    <TableHead className="text-xs font-semibold">Predicted Value</TableHead>
                    <TableHead className="text-xs font-semibold">Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {forecastData.map((item, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="text-xs font-medium">{item.date}</TableCell>
                      <TableCell className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                        {typeof item.predicted === 'number'
                          ? item.predicted.toLocaleString(undefined, { maximumFractionDigits: 4 })
                          : item.predicted}
                      </TableCell>
                      <TableCell>
                        {item.confidence != null ? (
                          <Badge
                            variant="outline"
                            className={cn(
                              'text-[10px]',
                              item.confidence >= 0.8
                                ? 'border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                                : item.confidence >= 0.6
                                ? 'border-amber-500/30 text-amber-600 dark:text-amber-400'
                                : 'border-rose-500/30 text-rose-600 dark:text-rose-400',
                            )}
                          >
                            {(item.confidence * 100).toFixed(1)}%
                          </Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {forecastData.length === 0 && !isForecasting && (
            <div className="text-center py-6 text-sm text-muted-foreground">
              Click "Generate Forecast" to predict future periods.
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Batch Prediction (Section 5) ──────────────────────────────────────────────

function BatchPrediction() {
  const { toast } = useToast();
  const store = useAppStore();
  const modelId = store.modelId;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [batchResults, setBatchResults] = useState<
    { row: number; features: Record<string, string | number>; predicted: number | string; confidence: number | null }[]
  >([]);
  const [isBatching, setIsBatching] = useState(false);

  const handleFileUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setIsBatching(true);
      setBatchResults([]);

      try {
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length < 2) {
          toast({ title: 'Invalid CSV', description: 'CSV must have a header row and at least one data row.', variant: 'destructive' });
          return;
        }

        const headers = lines[0].split(',').map((h) => h.trim());
        const results: typeof batchResults = [];

        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map((v) => v.trim());
          const features: Record<string, string | number> = {};
          headers.forEach((h, j) => {
            const val = values[j] ?? '';
            const num = parseFloat(val);
            features[h] = isNaN(num) || val === '' ? val : num;
          });

          try {
            const response = await apiClient.post('/predict', { model_id: modelId, features });
            const pred: PredictResponse = response.data;
            results.push({
              row: i,
              features,
              predicted: pred.prediction_label ?? pred.prediction,
              confidence: pred.confidence ?? null,
            });
          } catch {
            // skip failed rows
          }
        }

        setBatchResults(results);
        toast({ title: 'Batch Complete', description: `Processed ${results.length} of ${lines.length - 1} rows.` });
      } catch (error) {
        const msg = getApiErrorMessage(error, 'Batch prediction failed');
        toast({ title: 'Batch Error', description: msg, variant: 'destructive' });
      } finally {
        setIsBatching(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    },
    [modelId, toast],
  );

  return (
    <motion.div variants={itemVariants}>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10">
              <FileSpreadsheet className="h-4 w-4 text-emerald-500" />
            </div>
            <div>
              <CardTitle className="text-base">Batch Prediction</CardTitle>
              <CardDescription>Upload a CSV file with feature columns for bulk predictions</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileUpload}
            />
            <Button
              variant="outline"
              className="border-emerald-500/30 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10"
              onClick={() => fileInputRef.current?.click()}
              disabled={isBatching || !modelId}
            >
              {isBatching ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Upload className="mr-2 h-4 w-4" />
              )}
              Upload Features CSV
            </Button>
          </div>

          <p className="text-xs text-muted-foreground">
            CSV must have column headers matching the model&apos;s feature names. Each row will be predicted individually.
          </p>

          {/* Results Table */}
          {batchResults.length > 0 && (
            <div className="max-h-64 overflow-y-auto rounded-lg border custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    <TableHead className="text-xs font-semibold w-16">Row</TableHead>
                    <TableHead className="text-xs font-semibold">Features</TableHead>
                    <TableHead className="text-xs font-semibold">Predicted</TableHead>
                    <TableHead className="text-xs font-semibold w-24">Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {batchResults.map((item, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="text-xs font-medium">#{item.row}</TableCell>
                      <TableCell className="text-xs text-muted-foreground max-w-xs truncate">
                        {Object.entries(item.features)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(', ')}
                      </TableCell>
                      <TableCell className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                        {typeof item.predicted === 'number'
                          ? item.predicted.toLocaleString(undefined, { maximumFractionDigits: 4 })
                          : String(item.predicted)}
                      </TableCell>
                      <TableCell>
                        {item.confidence != null ? (
                          <Badge
                            variant="outline"
                            className={cn(
                              'text-[10px]',
                              item.confidence >= 0.8
                                ? 'border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                                : item.confidence >= 0.6
                                ? 'border-amber-500/30 text-amber-600 dark:text-amber-400'
                                : 'border-rose-500/30 text-rose-600 dark:text-rose-400',
                            )}
                          >
                            {(item.confidence * 100).toFixed(1)}%
                          </Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {batchResults.length === 0 && !isBatching && (
            <div className="text-center py-4 text-sm text-muted-foreground">
              No batch predictions yet. Upload a CSV to get started.
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ─── Prediction History (Section 6) ────────────────────────────────────────────

function PredictionHistory() {
  const { toast } = useToast();
  const store = useAppStore();
  const history = store.predictionHistory;

  const clearHistory = useCallback(() => {
    useAppStore.setState({
      predictionHistory: [],
      predictionResult: null,
      predictionAnalysis: null,
      predictionProbabilities: null,
    });
    toast({ title: 'History Cleared', description: 'All prediction records have been removed.' });
  }, [toast]);

  const exportCSV = useCallback(() => {
    if (history.length === 0) return;

    const headers = ['Timestamp', 'Prediction', 'Confidence', 'Features'];
    const rows = history.map((h) => [
      h.timestamp,
      String(h.prediction),
      h.confidence != null ? (h.confidence * 100).toFixed(1) + '%' : '',
      Object.entries(h.features)
        .map(([k, v]) => `${k}=${v}`)
        .join('; '),
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.map((v) => `"${v}"`).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    toast({ title: 'Exported', description: `${history.length} predictions exported as CSV.` });
  }, [history, toast]);

  return (
    <motion.div variants={itemVariants}>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10">
                <History className="h-4 w-4 text-emerald-500" />
              </div>
              <div>
                <CardTitle className="text-base flex items-center gap-2">
                  Prediction History
                  {history.length > 0 && (
                    <Badge variant="secondary" className="text-[10px]">{history.length}</Badge>
                  )}
                </CardTitle>
                <CardDescription>All previous predictions and analyses</CardDescription>
              </div>
            </div>
            {history.length > 0 && (
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8 text-xs" onClick={exportCSV}>
                      <Download className="mr-1.5 h-3.5 w-3.5" />
                      Export CSV
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Download all predictions as CSV</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs text-rose-600 dark:text-rose-400 border-rose-500/30 hover:bg-rose-500/10"
                      onClick={clearHistory}
                    >
                      <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                      Clear
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Clear all prediction history</TooltipContent>
                </Tooltip>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-center py-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted/50 mx-auto mb-3">
                <Clock className="h-6 w-6 text-muted-foreground/30" />
              </div>
              <p className="text-sm text-muted-foreground">No predictions made yet.</p>
              <p className="text-xs text-muted-foreground/60 mt-1">Your prediction history will appear here</p>
            </div>
          ) : (
            <div className="max-h-96 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              {[...history].reverse().map((item) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-start gap-3 p-3 rounded-lg border bg-card hover:bg-muted/30 transition-colors"
                >
                  <div className={cn(
                    'flex h-8 w-8 items-center justify-center rounded-lg shrink-0 mt-0.5',
                    item.confidence != null && item.confidence >= 0.8
                      ? 'bg-emerald-500/10'
                      : item.confidence != null && item.confidence >= 0.6
                      ? 'bg-amber-500/10'
                      : 'bg-muted',
                  )}>
                    {item.confidence != null && item.confidence >= 0.6 ? (
                      <CheckCircle2 className={cn('h-4 w-4', getConfidenceColor(item.confidence))} />
                    ) : (
                      <CircleDot className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                        {typeof item.prediction === 'number'
                          ? item.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
                          : String(item.prediction)}
                      </span>
                      {item.confidence != null && (
                        <Badge
                          variant="outline"
                          className={cn(
                            'text-[10px]',
                            item.confidence >= 0.8
                              ? 'border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                              : item.confidence >= 0.6
                              ? 'border-amber-500/30 text-amber-600 dark:text-amber-400'
                              : 'border-rose-500/30 text-rose-600 dark:text-rose-400',
                          )}
                        >
                          {(item.confidence * 100).toFixed(1)}%
                        </Badge>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-0.5 truncate">
                      {Object.entries(item.features)
                        .slice(0, 5)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(', ')}
                      {Object.keys(item.features).length > 5 && ` +${Object.keys(item.features).length - 5}`}
                    </p>
                  </div>
                  <span className="text-[10px] text-muted-foreground shrink-0 mt-0.5">
                    {formatTime(item.timestamp)}
                  </span>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function PredictionTab() {
  const { toast } = useToast();
  const store = useAppStore();
  const data = store.cleanedData || store.rawData;
  const columns = store.columns;
  const modelId = store.modelId;
  const modelTrained = store.modelTrained;
  const uploadedModel = store.uploadedModel;
  const featureImportance = store.featureImportance;
  const targetColumn = store.targetColumn;

  // ─── Local state ─────────────────────────────────────────────────────────
  const [isPredicting, setIsPredicting] = useState(false);
  const [isUploadingModel, setIsUploadingModel] = useState(false);
  const [featureInputs, setFeatureInputs] = useState<FeatureInput[]>([]);
  const [processedPreview, setProcessedPreview] = useState<Record<string, string | number> | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);

  // ─── Build feature inputs from model ─────────────────────────────────────
  const autoFillFeatureInputs = useCallback(() => {
    if (!uploadedModel?.features || uploadedModel.features.length === 0) {
      setFeatureInputs([]);
      return;
    }

    const dataset = data ?? [];
    const inputs: FeatureInput[] = uploadedModel.features.map((featName) => {
      const col = columns.find((c) => c.name === featName);
      const values = dataset.map((r) => r[featName]).filter((v) => v != null);
      const numericValues = values.filter((v): v is number => typeof v === 'number' && !isNaN(v));
      const inferredNumeric = !col && numericValues.length > 0;
      const role = col?.role || (inferredNumeric ? 'numeric' : 'categorical');
      const dtype = col?.dtype || (inferredNumeric ? 'number' : 'string');

      let min: number | undefined;
      let max: number | undefined;
      let mean: number | undefined;
      const uniqueValues: string[] = [];

      if (role === 'numeric') {
        if (numericValues.length > 0) {
          min = Math.min(...numericValues);
          max = Math.max(...numericValues);
          mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
        }
      } else {
        const uniqueSet = new Set(values.map((v) => String(v)));
        uniqueValues.push(...Array.from(uniqueSet).sort());
      }

      return {
        name: featName,
        role,
        dtype,
        value: dataset.length > 0 ? getSuggestedFeatureValue(dataset, featName) : '',
        uniqueValues,
        min,
        max,
        mean,
      };
    });

    setFeatureInputs(inputs);
  }, [uploadedModel?.features, data, columns]);

  useEffect(() => {
    autoFillFeatureInputs();
  }, [autoFillFeatureInputs]);

  // ─── Single Prediction ───────────────────────────────────────────────────
  const handleUploadModelFile = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setIsUploadingModel(true);

    try {
      const response = await apiClient.post('/upload-model', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const result: UploadModelResponse = response.data;

      useAppStore.setState({
        modelId: result.model_id,
        modelTrained: true,
        selectedModel: result.model_type,
        problemType: result.problem_type === 'classification' ? 'classification' : 'regression',
        targetColumn: result.target_column || 'prediction_target',
        selectedFeatures: result.feature_columns || [],
        modelMetrics: null,
        featureImportance: null,
        uploadedModel: {
          name: result.model_name,
          type: result.model_type,
          target: result.target_column || 'prediction_target',
          problem: result.problem_type,
          trainedAt: result.trained_at,
          metrics: {},
          features: result.feature_columns || [],
        },
      });
      setProcessedPreview(null);
      toast({
        title: 'Model Uploaded',
        description: `${result.model_name} is ready for prediction inputs.`,
      });
    } catch (error) {
      toast({
        title: 'Upload Failed',
        description: getApiErrorMessage(error, 'Could not load the uploaded model file.'),
        variant: 'destructive',
      });
    } finally {
      setIsUploadingModel(false);
    }
  }, [toast]);

  const openModelUpload = useCallback(() => {
    uploadInputRef.current?.click();
  }, []);

  const handlePredict = useCallback(async () => {
    if (!modelId) {
      toast({
        title: 'No Model ID',
        description: 'Model ID is missing. Please retrain the model in the ML tab.',
        variant: 'destructive',
      });
      return;
    }

    // Validate inputs
    const hasEmpty = featureInputs.some((fi) => fi.value === '');
    if (hasEmpty) {
      toast({
        title: 'Missing Values',
        description: 'Please fill in all feature values before predicting.',
        variant: 'destructive',
      });
      return;
    }

    // Build features object
    const features: Record<string, string | number> = {};
    for (const fi of featureInputs) {
      if (fi.role === 'numeric') {
        const num = parseFloat(fi.value);
        features[fi.name] = Number.isNaN(num) ? 0 : num;
      } else {
        features[fi.name] = fi.value;
      }
    }

    const processedFeatures = applyForecastDerivations(
      features,
      featureInputs.map((fi) => fi.name),
      data ?? [],
      targetColumn,
    );
    setProcessedPreview(processedFeatures);

    setIsPredicting(true);

    try {
      const response = await apiClient.post('/predict', { model_id: modelId, features: processedFeatures });
      const result: PredictResponse = response.data;

      const predictionValue = result.prediction_label ?? result.prediction;
      const analysis = buildPredictionExplanation(
        predictionValue,
        result.confidence,
        processedFeatures,
        targetColumn,
        featureImportance,
        store.problemType,
      );

      // Update store
      const existing = useAppStore.getState().predictionHistory;
      useAppStore.setState({
        predictionResult: predictionValue,
        predictionAnalysis: analysis,
        predictionProbabilities: result.probabilities ?? null,
        predictionHistory: [
          ...existing,
          {
            id: generateUUID(),
            prediction: predictionValue,
            confidence: result.confidence,
            probabilities: result.probabilities,
            features: processedFeatures,
            timestamp: new Date().toISOString(),
          },
        ],
      });

      toast({
        title: 'Prediction Complete',
        description: `Result: ${predictionValue}`,
      });
    } catch (error) {
      const msg = getApiErrorMessage(error, 'Prediction failed');
      toast({ title: 'Prediction Error', description: msg, variant: 'destructive' });
    } finally {
      setIsPredicting(false);
    }
  }, [modelId, featureInputs, toast, data, targetColumn, featureImportance, store.problemType]);

  // ─── Early return if no model ────────────────────────────────────────────
  if (!modelTrained) {
    return (
      <div className="space-y-6">
        <input
          ref={uploadInputRef}
          type="file"
          accept=".joblib,.pkl,.pickle"
          className="hidden"
          onChange={handleUploadModelFile}
        />
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BrainCircuit className="h-6 w-6 text-emerald-500" />
            Prediction
          </h2>
          <p className="text-muted-foreground mt-1">Make predictions using your trained machine learning model.</p>
        </motion.div>
        <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
          <UploadModelCard onUploadClick={openModelUpload} isUploading={isUploadingModel} />
          <EmptyState />
        </motion.div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BrainCircuit className="h-6 w-6 text-emerald-500" />
            Prediction
          </h2>
          <p className="text-muted-foreground mt-1">Make predictions and generate forecasts using your trained model.</p>
        </div>
        <div className="flex items-center gap-2">
          {uploadedModel && (
            <Badge variant="outline" className="border-emerald-500/20 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400">
              <Bot className="mr-1 h-3 w-3" />
              {getModelDisplayName(uploadedModel.name)}
            </Badge>
          )}
          {store.problemType && (
            <Badge variant="outline" className="border-teal-500/20 bg-teal-500/5 text-teal-600 dark:text-teal-400">
              {store.problemType === 'regression' ? (
                <TrendingUp className="mr-1 h-3 w-3" />
              ) : (
                <Tag className="mr-1 h-3 w-3" />
              )}
              {store.problemType === 'regression' ? 'Regression' : 'Classification'}
            </Badge>
          )}
        </div>
      </motion.div>

      <input
        ref={uploadInputRef}
        type="file"
        accept=".joblib,.pkl,.pickle"
        className="hidden"
        onChange={handleUploadModelFile}
      />

      {/* Main Content */}
      <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
        <UploadModelCard onUploadClick={openModelUpload} isUploading={isUploadingModel} />

        {/* Section 1: Model Status Card */}
        <ModelStatusCard />

        {processedPreview && (
          <motion.div variants={itemVariants}>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-500/10">
                    <Sparkles className="h-4 w-4 text-teal-500" />
                  </div>
                  <div>
                    <CardTitle className="text-base">Processed Input Data</CardTitle>
                    <CardDescription>Final values sent to the prediction engine after auto-fill and date derivations.</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(processedPreview).map(([key, value]) => (
                    <div key={key} className="rounded-lg border bg-background/60 p-3">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">{key}</p>
                      <p className="mt-1 text-sm font-medium break-all">{String(value)}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Section 2 & 3: Input Form + Result (side by side on large screens) */}
        <div className="grid gap-6 lg:grid-cols-2">
          <PredictionForm
            featureInputs={featureInputs}
            setFeatureInputs={setFeatureInputs}
            onPredict={handlePredict}
            onAutoFill={autoFillFeatureInputs}
            isPredicting={isPredicting}
          />
          <PredictionResult
            result={store.predictionResult}
            analysis={store.predictionAnalysis}
            probabilities={store.predictionProbabilities}
            problemType={store.problemType}
          />
        </div>

        {/* Section 4: Future Prediction (only if date column exists) */}
        <FuturePrediction
          featureInputs={featureInputs}
          setFeatureInputs={setFeatureInputs}
          isPredicting={isPredicting}
        />

        <Separator />

        <PredictionHistory />
      </motion.div>
    </div>
  );
}


