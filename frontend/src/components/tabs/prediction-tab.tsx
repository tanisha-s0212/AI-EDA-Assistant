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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
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
  History,
  Zap,
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
} from 'recharts';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { clearStaleModelState } from '@/lib/model-lifecycle';
import { predictionInputs, downloadPredictionCsv, type PredictionSchema } from '@/lib/prediction-contract';
import { PredictionBatch } from '@/components/prediction-batch';
import { cn } from '@/lib/utils';

// ─── Types ─────────────────────────────────────────────────────────────────────

interface PredictResponse {
  status?: 'success' | 'failed';
  error?: string;
  detail?: string;
  prediction: number | string;
  prediction_label?: string | number;
  warnings?: string[];
  probability_note?: string;
  readiness?: PredictionSchema['readiness'];
  model_id?: string;
  dataset_version?: number;
  dataset_fingerprint?: string;
  inputs?: Record<string, string | number>;
  scored_at?: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  top_class?: string;
  error_band?: { lower: number; upper: number; margin: number };
  error_band_note?: string;
  confidence_interval?: { lower: number; upper: number; margin: number };
  cv_score_mean?: number | null;
  cv_score_std?: number | null;
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
    transition: { type: 'spring', stiffness: 240, damping: 28 },
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

function getModelDisplayName(modelName: string): string {
  return modelName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getConfidenceColor(value: number | undefined | null): string {
  if (value == null) return 'text-muted-foreground';
  if (value >= 0.8) return 'text-primary';
  if (value >= 0.6) return 'text-amber-500';
  return 'text-rose-500';
}

function getConfidenceBg(value: number | undefined | null): string {
  if (value == null) return 'bg-muted';
  if (value >= 0.8) return 'bg-primary/10 border-primary/20';
  if (value >= 0.6) return 'bg-amber-500/10 border-amber-500/20';
  return 'bg-rose-500/10 border-rose-500/20';
}

function stringifyFeatureValue(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return '';
  return String(value);
}

function getSuggestedFeatureValue(data: Record<string, string | number | boolean | null>[], featureName: string): string {
  return stringifyFeatureValue(data[data.length - 1]?.[featureName]);
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
    lines.push(`The model assigns ${(confidence * 100).toFixed(1)}% probability to its top outcome. Calibration has not been measured; this score does not guarantee correctness.`);
  }

  const topFeatures = (featureImportance ?? []).slice(0, 3);
  if (topFeatures.length > 0) {
    const summary = topFeatures
      .map((item) => {
        const value = processedFeatures[item.name];
        return value !== undefined ? `${item.name}=${value}` : item.name;
      })
      .join(', ');
    lines.push(`Global feature importance includes ${summary}. These are model-wide associations and do not explain the cause of this individual prediction.`);
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
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10">
          <BrainCircuit className="h-10 w-10 text-primary" />
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
        No trained model found. Please complete ML Assistant training first.
      </p>
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-6">
        <Zap className="h-3.5 w-3.5 text-amber-500" />
        <span>Train a model in 6 simple steps with AI guidance</span>
      </div>
      <Button
        className="bg-primary text-primary-foreground shadow-lg shadow-primary/20 hover:bg-primary/90"
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
      <Card className="border-dashed border-primary/20 bg-primary/5">
        <CardContent className="p-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 shrink-0">
              <Upload className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium">Upload a trained model</p>
              <p className="text-xs text-muted-foreground mt-1">
                Import a trusted `.joblib`, `.pkl`, or `.pickle` sklearn model with named inputs. Only use files from a trusted source: Python model files can execute code when loaded.
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            onClick={onUploadClick}
            disabled={isUploading}
            className="border-primary/20 text-primary hover:bg-primary/5"
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

function ModelStatusCard({ schema }: { schema: PredictionSchema | null }) {
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
              <p className="text-xs text-muted-foreground">Train a model in the ML Assistant workflow to get started</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-rose-500/30 text-rose-600 dark:text-rose-400 hover:bg-rose-500/10"
              onClick={() => setActiveTab('ml')}
            >
              <ArrowRight className="mr-1 h-3.5 w-3.5" />
              Open ML Assistant
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  const primaryMetric = uploadedModel.problem === 'regression' ? 'R2' : 'Accuracy';
  const metricValue = schema?.readiness.metrics?.[primaryMetric] ?? modelMetrics?.[primaryMetric] ?? null;
  const rawFeatures = schema?.fields.map(field => field.name) ?? [];

  return (
    <motion.div variants={itemVariants}>
      <Card className="overflow-hidden border border-primary/20 bg-gradient-to-br from-primary/8 via-background to-secondary/70">
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            {/* Model Info */}
            <div className="flex items-center gap-3 flex-1">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/12 shrink-0">
                <BrainCircuit className="h-6 w-6 text-primary" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-sm font-semibold truncate">
                    {getModelDisplayName(uploadedModel.name)}
                  </h3>
                  <Badge className="border-primary/20 bg-primary/10 text-primary hover:bg-primary/15">
                    <CheckCircle2 className="mr-1 h-3 w-3" />
                    Model Loaded
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
                  <p className="text-xs text-muted-foreground">Validation {primaryMetric}</p>
                  <p className={cn('text-lg font-bold leading-tight', getConfidenceColor(metricValue))}>
                    {uploadedModel.problem === 'regression' ? metricValue.toFixed(4) : `${(metricValue * 100).toFixed(1)}%`}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Features Summary */}
          {rawFeatures.length > 0 && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Layers className="h-3.5 w-3.5 text-primary" />
                <span className="text-xs font-medium text-muted-foreground">
                  {rawFeatures.length} Feature{rawFeatures.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {rawFeatures.slice(0, 8).map((f) => (
                  <Badge key={f} variant="outline" className="text-[10px] px-1.5 py-0">
                    {f}
                  </Badge>
                ))}
                {rawFeatures.length > 8 && (
                  <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                    +{rawFeatures.length - 8} more
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
    useAppStore.setState({ predictionResult: null, predictionAnalysis: null, predictionProbabilities: null });
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
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <Target className="h-4 w-4 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">Prediction Input</CardTitle>
              <CardDescription>Inputs start with the last dataset row. Review the values and fill any missing fields before predicting.</CardDescription>
              <p className="mt-2 text-xs text-muted-foreground">This estimates the model target for one record. Use ML Assistant for totals and rankings, and forecasting for future periods.</p>
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
                        <Hash className="h-3 w-3 text-primary" />
                      ) : (
                        <Tag className="h-3 w-3 text-secondary-foreground" />
                      )}
                      {fi.name}
                    </Label>
                    {fi.role === 'numeric' ? (
                      <Input
                        type="number"
                        disabled={isPredicting}
                        step="any"
                        placeholder={`e.g. ${fi.mean != null ? fi.mean.toFixed(1) : '0'}`}
                        value={fi.value}
                        onChange={(e) => handleInputChange(idx, e.target.value)}
                        className="h-9 text-sm"
                      />
                    ) : fi.uniqueValues.length > 0 ? (
                      <Select value={fi.value} onValueChange={(v) => handleInputChange(idx, v)} disabled={isPredicting}>
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
                        disabled={isPredicting}
                        placeholder="Enter value"
                        value={fi.value}
                        onChange={(e) => handleInputChange(idx, e.target.value)}
                        className="h-9 text-sm"
                      />
                    )}
                    {fi.role === 'numeric' && fi.min != null && fi.max != null && (
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <BarChart3 className="h-2.5 w-2.5" />
                        Training range: {fi.min} — {fi.max} {fi.mean != null && `| Preview avg: ${fi.mean.toFixed(1)}`}
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
                  className="border-primary/20 text-primary hover:bg-primary/5"
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Auto-Fill From Dataset
                </Button>
                <Button
                  size="lg"
                  className="bg-primary px-8 text-primary-foreground shadow-lg shadow-primary/20 hover:bg-primary/90"
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
          <Card className="overflow-hidden border border-primary/20 bg-gradient-to-br from-primary/6 via-background to-secondary/70">
            <div className="relative bg-gradient-to-r from-primary/12 via-secondary/60 to-transparent p-6">
              <div className="absolute top-0 right-0 h-32 w-32 rounded-full bg-primary/5 -translate-y-1/2 translate-x-1/2" />
              <div className="relative text-center">
                <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-background/70 px-3 py-1 text-[11px] font-medium text-primary">
                  <Sparkles className="h-3.5 w-3.5" />
                  Prediction result
                </div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Predicted Result</p>
                <motion.p
                  className="bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-4xl font-bold text-transparent sm:text-5xl"
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
                  <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Model probability (uncalibrated)</p>
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
                  <div className="flex items-center gap-2 text-primary">
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
                          <div className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary">
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
                                ? 'bg-gradient-to-r from-primary to-primary/70'
                                : 'bg-primary/30',
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

    const headers = ['Timestamp', 'Prediction', 'Model probability (uncalibrated)', 'Features', 'Model ID', 'Dataset version', 'Dataset fingerprint', 'Warnings', 'Approximate error range', 'Validation status'];
    const rows = history.map((h) => [
      h.timestamp,
      String(h.prediction),
      h.confidence != null ? (h.confidence * 100).toFixed(1) + '%' : '',
      Object.entries(h.features)
        .map(([k, v]) => `${k}=${v}`)
        .join('; '),
      h.audit?.modelId ?? '',
      h.audit?.datasetVersion ?? '',
      h.audit?.datasetFingerprint ?? '',
      [...(h.audit?.warnings ?? []), ...(h.audit?.readiness?.reasons ?? []), h.audit?.probabilityNote ?? ''].join(' '),
      h.audit?.errorBand ? `${h.audit.errorBand.lower} to ${h.audit.errorBand.upper}` : '',
      h.audit?.readiness?.status ?? 'unverified',
    ]);

    downloadPredictionCsv(`predictions_${new Date().toISOString().split('T')[0]}.csv`, [headers, ...rows]);

    toast({ title: 'Exported', description: `${history.length} predictions exported as CSV.` });
  }, [history, toast]);

  return (
    <motion.div variants={itemVariants}>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <History className="h-4 w-4 text-primary" />
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
                      ? 'bg-primary/10'
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
                      <span className="text-sm font-semibold text-primary">
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
                              ? 'border-primary/20 text-primary'
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
                    {item.audit && <p className="text-[10px] text-muted-foreground">Model: {item.audit.modelId} · Dataset version: {item.audit.datasetVersion ?? 'Unverified'}{item.audit.warnings.length > 0 ? ' · Review warnings in export' : ''}</p>}
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
  const [schema, setSchema] = useState<PredictionSchema | null>(null);
  const [schemaError, setSchemaError] = useState('');
  const [predictionWarnings, setPredictionWarnings] = useState<string[]>([]);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let active = true;
    setSchema(null); setSchemaError(''); setProcessedPreview(null);
    useAppStore.setState({ predictionResult: null, predictionAnalysis: null, predictionProbabilities: null });
    if (modelId) {
      apiClient.get(`/prediction/schema/${encodeURIComponent(modelId)}`, { params: { dataset_id: store.datasetId ?? undefined } })
        .then(response => { if (active) setSchema(response.data); })
        .catch(error => {
          if (active) {
            clearStaleModelState(error);
            setSchemaError(getApiErrorMessage(error, 'Unable to verify the model input schema.'));
          }
        });
    }
    return () => { active = false; };
  }, [modelId, store.datasetId]);

  // ─── Build feature inputs from model ─────────────────────────────────────
  const autoFillFeatureInputs = useCallback(() => {
    if (!schema?.fields.length) {
      setFeatureInputs([]);
      return;
    }

    const dataset = data ?? [];
    const inputs: FeatureInput[] = schema.fields.map((field) => {
      const featName = field.name;
      const col = columns.find((c) => c.name === featName);
      const values = dataset.map((r) => r[featName]).filter((v) => v != null);
      const numericValues = values.filter((v): v is number => typeof v === 'number' && !isNaN(v));
      const inferredNumeric = !col && numericValues.length > 0;
      const role = field.type === 'numeric' || field.type === 'boolean' ? 'numeric' : field.type === 'unknown' ? (col?.role || (inferredNumeric ? 'numeric' : 'categorical')) : 'categorical';
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
        uniqueValues.push(...(field.categories.length ? field.categories : Array.from(uniqueSet).filter(Boolean).sort()));
      }

      return {
        name: featName,
        role,
        dtype,
        value: field.type === 'boolean' && dataset.length > 0
          ? ({ true: '1', false: '0' }[getSuggestedFeatureValue(dataset, featName)] ?? getSuggestedFeatureValue(dataset, featName))
          : dataset.length > 0 ? getSuggestedFeatureValue(dataset, featName) : '',
        uniqueValues,
        min: field.range?.min,
        max: field.range?.max,
        mean,
      };
    });

    setFeatureInputs(inputs);
  }, [schema, data, columns]);

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
    useAppStore.setState({ predictionResult: null, predictionAnalysis: null, predictionProbabilities: null });
    if (!modelId || !schema || featureInputs.length === 0) {
      toast({
        title: 'Model inputs unavailable',
        description: 'Wait for the model schema check or review the readiness message above.',
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
    let processedFeatures: Record<string, string | number>;
    try { processedFeatures = predictionInputs(featureInputs); }
    catch (error) {
      toast({ title: 'Invalid input', description: (error as Error).message, variant: 'destructive' });
      return;
    }
    const requestContext = `${modelId}:${store.datasetId}`;
    const contextChanged = () => {
      const current = useAppStore.getState();
      return `${current.modelId}:${current.datasetId}` !== requestContext;
    };
    setProcessedPreview(processedFeatures);

    setIsPredicting(true);

    try {
      const response = await apiClient.post('/predict', { model_id: modelId, dataset_id: store.datasetId, features: processedFeatures });
      const result: PredictResponse = response.data;
      if (contextChanged()) return;
      if (result.status === 'failed') {
        toast({
          title: 'Prediction Error',
          description: result.error || 'Prediction failed',
          variant: 'destructive',
        });
        return;
      }

      const predictionValue = result.prediction_label ?? result.prediction;
      setPredictionWarnings([...(result.warnings ?? []), ...(result.readiness?.reasons ?? [])]);
      let analysis = buildPredictionExplanation(
        predictionValue,
        result.confidence,
        processedFeatures,
        targetColumn,
        featureImportance,
        store.problemType,
      );
      const errorBand = result.error_band ?? result.confidence_interval;
      if (errorBand) {
        analysis += ` Approximate validation error band: ${errorBand.lower.toLocaleString()} to ${errorBand.upper.toLocaleString()} (plus/minus ${errorBand.margin.toLocaleString()}). ${result.error_band_note || 'This range is not a calibrated 95% prediction interval.'}`;
      }
      if (result.cv_score_mean != null) {
        analysis += ` Validation score: mean ${Number(result.cv_score_mean).toFixed(4)}${result.cv_score_std != null ? `, std ${Number(result.cv_score_std).toFixed(4)}` : ''}.`;
      }
      analysis += ` ${[...(result.warnings ?? []), ...(result.readiness?.reasons ?? []), result.probability_note ?? ''].join(' ')}`;

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
            audit: { modelId, datasetVersion: result.dataset_version, datasetFingerprint: result.dataset_fingerprint,
              warnings: result.warnings ?? [], errorBand: result.error_band, readiness: result.readiness,
              scoredAt: result.scored_at, probabilityNote: result.probability_note },
          },
        ],
      });

      toast({
        title: 'Prediction Complete',
        description: `Result: ${predictionValue}`,
      });
    } catch (error) {
      if (contextChanged()) return;
      if (clearStaleModelState(error)) {
        toast({ title: 'Model expired', description: 'The dataset changed. Train the model again before predicting.', variant: 'destructive' });
        return;
      }
      const msg = getApiErrorMessage(error, 'Prediction failed');
      toast({ title: 'Prediction Error', description: msg, variant: 'destructive' });
    } finally {
      setIsPredicting(false);
    }
  }, [modelId, schema, featureInputs, toast, targetColumn, featureImportance, store.problemType, store.datasetId]);

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
            <BrainCircuit className="h-6 w-6 text-primary" />
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
            <BrainCircuit className="h-6 w-6 text-primary" />
            Prediction
          </h2>
          <p className="text-muted-foreground mt-1">Make predictions and generate forecasts using your trained model.</p>
        </div>
        <div className="flex items-center gap-2">
          {uploadedModel && (
            <Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">
              <Bot className="mr-1 h-3 w-3" />
              {getModelDisplayName(uploadedModel.name)}
            </Badge>
          )}
          {store.problemType && (
            <Badge variant="outline" className="border-secondary bg-secondary text-secondary-foreground">
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
        <ModelStatusCard schema={schema} />

        <Card>
          <CardHeader><CardTitle className="text-base">Prediction readiness</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            {schemaError ? <p role="alert" className="text-destructive">{schemaError}</p> : !schema ? <p>Checking the saved model and required inputs…</p> : <>
              <p>{schema.readiness.status === 'available' ? 'Validation evidence available' : 'Review required before using results'}</p>
              <p>Target: {schema.target}. Trained: {schema.trained_at || 'Unavailable'}. Dataset version: {schema.dataset_version ?? 'Unverified'}.</p>
              {schema.readiness.method && <p>Validation: {schema.readiness.method}, {schema.readiness.rows} evaluated rows.</p>}
              {schema.readiness.metrics && <p>{Object.entries(schema.readiness.metrics).map(([name, value]) => `${name}: ${typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : 'unavailable'}`).join(' · ')}</p>}
              {schema.readiness.reasons.map((reason, index) => <p key={index} className="text-amber-700 dark:text-amber-400">{reason}</p>)}
              <p className="text-muted-foreground">Review all auto-filled inputs. Values outside the training range and new categories receive warnings. Regression ranges are approximate; class probabilities are uncalibrated.</p>
            </>}
          </CardContent>
        </Card>

        {processedPreview && (
          <motion.div variants={itemVariants}>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-secondary text-secondary-foreground">
                    <Sparkles className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-base">Processed Input Data</CardTitle>
                    <CardDescription>The input values submitted to the model.</CardDescription>
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
        {store.predictionResult !== null && predictionWarnings.length > 0 && <Card className="border-amber-500/40">
          <CardHeader><CardTitle className="text-base">Review this prediction</CardTitle></CardHeader>
          <CardContent role="status" className="space-y-2 text-sm">{predictionWarnings.map((warning, index) => <p key={index}>{warning}</p>)}</CardContent>
        </Card>}
        <div className="grid gap-6 lg:grid-cols-2">
          <PredictionForm
            featureInputs={featureInputs}
            setFeatureInputs={setFeatureInputs}
            onPredict={handlePredict}
            onAutoFill={autoFillFeatureInputs}
            isPredicting={isPredicting || !schema}
          />
          <PredictionResult
            result={store.predictionResult}
            analysis={store.predictionAnalysis}
            probabilities={store.predictionProbabilities}
            problemType={store.problemType}
          />
        </div>

        <Separator />

        {modelId && schema && <PredictionBatch key={`${modelId}:${store.datasetId}`} modelId={modelId} datasetId={store.datasetId} />}

        <Card><CardContent className="pt-6 text-sm">For forecasts over future dates, use Sales Forecast Using Time Series Analysis or Sales Forecast Using Machine Learning. Those tabs validate time order and generate future features.</CardContent></Card>

        <PredictionHistory />
      </motion.div>
    </div>
  );
}


