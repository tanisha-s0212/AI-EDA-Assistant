'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useAppStore, ColumnInfo, DataRow } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { Textarea } from '@/components/ui/textarea';
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
  Zap,
  Sparkles,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  BarChart3,
  Trophy,
  ArrowRight,
  Lightbulb,
  TrendingUp,
  Shield,
  Gauge,
  Hash,
  Tag,
  Activity,
  Info,
  Star,
  Award,
  Bot,
  Swords,
  GitCompareArrows,
  Eye,
  CircleDot,
  Play,
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
  ScatterChart,
  Scatter,
  ReferenceLine,
} from 'recharts';
import { cn } from '@/lib/utils';
import { apiClient, getApiErrorMessage } from '@/lib/api';

// ─── Types ─────────────────────────────────────────────────────────────────────

type ProblemType = 'regression' | 'classification';
type TrainingMode = 'fast' | 'balanced';

interface ModelDef {
  id: string;
  name: string;
  description: string;
  recommended: boolean;
  whyRecommended: string;
  problemType: ProblemType;
}

interface TrainResponse {
  model_id: string;
  metrics: Record<string, number>;
  feature_importance: { name: string; importance: number }[];
  analysis: string;
  sample_predictions?: { actual: string | number; predicted: string | number }[];
  cv_scores?: number[];
  overfitting_detected?: boolean;
  overfitting_status?: 'healthy' | 'watch' | 'detected';
  overfitting_explanation?: string;
  generalization_gap?: number;
  cv_gap?: number | null;
  training_time?: number;
  optimization?: {
    training_rows_available?: number;
    training_rows_used?: number;
    training_sampled?: boolean;
    cv_rows_evaluated?: number;
    cv_folds_used?: number;
    cv_sampled?: boolean;
    training_mode?: TrainingMode;
    importance_rows_evaluated?: number;
    importance_repeats?: number;
  };
}

interface ComparisonResult {
  modelId: string;
  modelUuid: string;
  modelName: string;
  metrics: Record<string, number>;
  featureImportance: { name: string; importance: number }[];
  analysis: string;
  cvScores: number[];
  trainingTime: number;
  samplePredictions?: { actual: string | number; predicted: string | number }[];
  overfittingDetected?: boolean;
  overfittingStatus?: 'healthy' | 'watch' | 'detected';
  overfittingExplanation?: string;
  generalizationGap?: number;
  cvGap?: number | null;
  optimization?: TrainResponse['optimization'];
}

// ─── Model Definitions ────────────────────────────────────────────────────────

const REGRESSION_MODELS: ModelDef[] = [
  { id: 'random_forest', name: 'Random Forest', description: 'Ensemble of decision trees with bagging for robust predictions', recommended: true, whyRecommended: 'Excellent general-purpose model with built-in feature importance. Handles non-linear relationships well and resists overfitting.', problemType: 'regression' },
  { id: 'gradient_boosting', name: 'Gradient Boosting', description: 'Sequential ensemble that corrects errors of previous trees', recommended: true, whyRecommended: 'Often achieves highest accuracy. Excellent for structured/tabular data with complex patterns.', problemType: 'regression' },
  { id: 'ridge_regression', name: 'Ridge Regression', description: 'Linear regression with L2 regularization for coefficient shrinkage', recommended: true, whyRecommended: 'Great baseline model. Fast training, interpretable, and handles multicollinearity well.', problemType: 'regression' },
  { id: 'elasticnet', name: 'Elastic Net', description: 'Combines L1 and L2 regularization for feature selection + shrinkage', recommended: false, whyRecommended: 'Useful when you have correlated features and want automatic feature selection.', problemType: 'regression' },
  { id: 'lasso_regression', name: 'Lasso Regression', description: 'Linear regression with L1 regularization for feature selection', recommended: false, whyRecommended: 'Good for feature selection when you have many features. Can zero out irrelevant features.', problemType: 'regression' },
  { id: 'svr', name: 'Support Vector Regression', description: 'Kernel-based method that finds optimal margin around predictions', recommended: false, whyRecommended: 'Works well with high-dimensional data but requires careful tuning of hyperparameters.', problemType: 'regression' },
  { id: 'decision_tree', name: 'Decision Tree', description: 'Single tree that splits data based on feature thresholds', recommended: false, whyRecommended: 'Highly interpretable but prone to overfitting. Good for understanding feature relationships.', problemType: 'regression' },
  { id: 'knn_regressor', name: 'KNN Regressor', description: 'Predicts based on average of k nearest neighbors', recommended: false, whyRecommended: 'Simple and non-parametric but sensitive to feature scaling and k value.', problemType: 'regression' },
];

const CLASSIFICATION_MODELS: ModelDef[] = [
  { id: 'random_forest', name: 'Random Forest', description: 'Ensemble of decision trees with majority voting', recommended: true, whyRecommended: 'Excellent general-purpose classifier with built-in feature importance. Handles imbalanced data well.', problemType: 'classification' },
  { id: 'gradient_boosting', name: 'Gradient Boosting', description: 'Sequential ensemble that corrects misclassifications', recommended: true, whyRecommended: 'Often achieves highest accuracy on structured data. Excellent for complex classification boundaries.', problemType: 'classification' },
  { id: 'logistic_regression', name: 'Logistic Regression', description: 'Linear model with sigmoid activation for probability estimation', recommended: true, whyRecommended: 'Great baseline model. Fast, interpretable, provides probability scores. Works well for linearly separable data.', problemType: 'classification' },
  { id: 'svm', name: 'Support Vector Machine', description: 'Kernel-based method that finds optimal decision boundary', recommended: false, whyRecommended: 'Powerful for high-dimensional data with non-linear boundaries using kernel trick.', problemType: 'classification' },
  { id: 'knn', name: 'KNN Classifier', description: 'Classifies based on majority vote of k nearest neighbors', recommended: false, whyRecommended: 'Simple and effective for small datasets. Distance-based so requires feature scaling.', problemType: 'classification' },
  { id: 'decision_tree', name: 'Decision Tree', description: 'Single tree splitting data with information gain criteria', recommended: false, whyRecommended: 'Highly interpretable but prone to overfitting. Good for understanding classification rules.', problemType: 'classification' },
];

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

const scaleVariants: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { type: 'spring', stiffness: 220, damping: 24 },
  },
};

const ML_CHART_COLORS = {
  primary: '#2563eb',
  secondary: '#7c3aed',
  accent: '#14b8a6',
  warning: '#f59e0b',
  danger: '#ef4444',
  grid: '#cbd5e1',
  bars: ['#2563eb', '#7c3aed', '#14b8a6', '#f59e0b', '#ef4444', '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'],
} as const;

// ─── Helper Functions ─────────────────────────────────────────────────────────

function isIdLikeColumn(col: ColumnInfo): boolean {
  const name = col.name.toLowerCase();
  const idPatterns = ['id', 'uuid', 'index', '_id', 'number', 'no.', 'seq', 'key', 'code', 'identifier'];
  if (idPatterns.some(p => name === p || name.endsWith(p))) return true;
  if (col.uniqueCount === col.nonNull && col.uniqueCount > 10 && col.nonNull > 20) return true;
  if (col.role === 'identifier') return true;
  return false;
}

function detectProblemType(col: ColumnInfo, data: DataRow[]): { type: ProblemType; confidence: number } {
  if (col.role === 'numeric') {
    const values = data
      .map(r => r[col.name])
      .filter((v): v is number => typeof v === 'number' && !isNaN(v));
    const uniqueRatio = col.uniqueCount / Math.max(col.nonNull, 1);
    const numericRatio = values.length / Math.max(data.length, 1);

    if (uniqueRatio > 0.9 && col.uniqueCount > 10) {
      return { type: 'regression', confidence: 70 };
    }
    if (numericRatio > 0.8) {
      return { type: 'regression', confidence: 90 };
    }
    return { type: 'regression', confidence: 60 };
  }

  const uniqueRatio = col.uniqueCount / Math.max(col.nonNull, 1);
  if (col.role === 'categorical' || col.role === 'boolean') {
    return { type: 'classification', confidence: 85 };
  }
  if (uniqueRatio < 0.3 && col.uniqueCount <= 20) {
    return { type: 'classification', confidence: 75 };
  }
  return { type: 'classification', confidence: 60 };
}

function getAutoTarget(columns: ColumnInfo[]): { target: string; reason: string } | null {
  const priorityPatterns = ['deposit', 'subscribed', 'subscribe', 'subscription', 'term_deposit', 'target', 'label', 'class', 'output', 'result', 'prediction', 'dependent', 'y'];
  const exactPriorityNames = ['deposit', 'y', 'subscribed', 'subscribe', 'subscription', 'term_deposit'];
  const numericPatterns = ['price', 'sales', 'revenue', 'profit', 'cost', 'amount', 'score', 'rating', 'value', 'quantity'];
  const calendarPatterns = ['day', 'month', 'year', 'week', 'weekday', 'quarter'];
  const names = columns.map((col) => col.name.toLowerCase());
  const bankMarketingSignature = ['job', 'marital', 'housing', 'loan', 'contact', 'campaign', 'poutcome'].every((name) => names.includes(name));

  for (const exactName of exactPriorityNames) {
    const exactMatch = columns.find((col) => col.name.toLowerCase() === exactName);
    if (exactMatch) {
      return { target: exactMatch.name, reason: `Column "${exactMatch.name}" is an exact high-priority target match.` };
    }
  }

  if (bankMarketingSignature) {
    const depositColumn = columns.find((col) => col.name.toLowerCase() === 'deposit');
    if (depositColumn) {
      return { target: depositColumn.name, reason: `Selected "${depositColumn.name}" as the bank marketing subscription target.` };
    }
    const yColumn = columns.find((col) => col.name.toLowerCase() === 'y');
    if (yColumn) {
      return { target: yColumn.name, reason: `Selected "${yColumn.name}" as the bank marketing subscription target.` };
    }
  }

  for (const col of columns) {
    const name = col.name.toLowerCase();
    if (priorityPatterns.some((p) => name === p || name.endsWith(p) || name.includes(`_${p}`))) {
      return { target: col.name, reason: `Column "${col.name}" matches strong target naming patterns.` };
    }
  }

  const binaryCategorical = columns.find((col) => {
    const name = col.name.toLowerCase();
    return !isIdLikeColumn(col) && !calendarPatterns.includes(name) && (col.role === 'categorical' || col.role === 'boolean') && col.uniqueCount <= 5;
  });
  if (binaryCategorical) {
    return { target: binaryCategorical.name, reason: `Selected "${binaryCategorical.name}" as a likely classification target because it is a low-cardinality categorical field.` };
  }

  for (const col of columns) {
    const name = col.name.toLowerCase();
    if (calendarPatterns.includes(name)) continue;
    if (numericPatterns.some((p) => name === p || name.endsWith(p) || name.includes(`_${p}`))) {
      return { target: col.name, reason: `Column "${col.name}" looks like a typical prediction target (${col.role}).` };
    }
  }

  const numericCols = columns.filter((c) => c.role === 'numeric' && !isIdLikeColumn(c) && !calendarPatterns.includes(c.name.toLowerCase()));
  if (numericCols.length > 0) {
    const lastNumeric = numericCols[numericCols.length - 1];
    return { target: lastNumeric.name, reason: `Selected "${lastNumeric.name}" as the best remaining numeric target candidate.` };
  }

  return null;
}

function getAutoFeatures(columns: ColumnInfo[], target: string): string[] {
  return columns
    .filter(c => c.name !== target && !isIdLikeColumn(c) && c.role !== 'datetime')
    .map(c => c.name);
}

function formatMetricValue(value: number): string {
  if (value >= 1) return value.toFixed(2);
  if (value >= 0.1) return value.toFixed(3);
  return value.toFixed(4);
}

function parseModelAnalysis(analysis: string) {
  const lines = analysis
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const titleLine = lines.find((line) => line.startsWith('###')) ?? lines[0] ?? 'Model Summary';
  const title = titleLine.replace(/^###\s*/, '').trim();
  const bullets = lines
    .filter((line) => line.startsWith('-'))
    .map((line) => line.replace(/^-\s*/, '').trim());
  const summary = lines.filter((line) => !line.startsWith('###') && !line.startsWith('-')).join(' ');

  return { title, bullets, summary };
}

function buildModelSuggestionExplanation(params: {
  selectedTarget: string;
  targetProblemType: ProblemType;
  confidence: number;
  selectedFeatures: string[];
  columns: ColumnInfo[];
  recommendedModel: ModelDef | null;
}) {
  const { selectedTarget, targetProblemType, confidence, selectedFeatures, columns, recommendedModel } = params;
  const targetColumn = columns.find((column) => column.name === selectedTarget);
  const numericFeatures = selectedFeatures.filter((feature) => columns.find((column) => column.name === feature)?.role === 'numeric').length;
  const categoricalFeatures = selectedFeatures.filter((feature) => {
    const role = columns.find((column) => column.name === feature)?.role;
    return role === 'categorical' || role === 'boolean';
  }).length;

  return [
    `The uploaded dataset is currently being treated as a ${targetProblemType} problem because "${selectedTarget}" looks like a ${targetColumn?.role ?? 'candidate'} target with ${confidence}% detection confidence.`,
    `${selectedFeatures.length} feature${selectedFeatures.length === 1 ? '' : 's'} are selected for training, including ${numericFeatures} numeric and ${categoricalFeatures} categorical/boolean field${categoricalFeatures === 1 ? '' : 's'}.`,
    recommendedModel
      ? `${recommendedModel.name} is recommended as the starting model because ${recommendedModel.whyRecommended.toLowerCase()}`
      : 'A default baseline model is suggested first so the workflow can establish a measurable starting point.',
  ];
}

function getRecommendedStarterModel(
  models: ModelDef[],
  problemType: ProblemType,
  selectedFeatures: string[],
  columns: ColumnInfo[],
) {
  const selectedFeatureColumns = selectedFeatures
    .map((feature) => columns.find((column) => column.name === feature))
    .filter((column): column is ColumnInfo => Boolean(column));
  const featureCount = selectedFeatureColumns.length;
  const numericCount = selectedFeatureColumns.filter((column) => column.role === 'numeric').length;
  const categoricalCount = selectedFeatureColumns.filter((column) => column.role === 'categorical' || column.role === 'boolean').length;

  const preferredOrder =
    problemType === 'regression'
      ? featureCount >= 6 || (numericCount >= 4 && numericCount >= categoricalCount)
        ? ['gradient_boosting', 'random_forest', 'ridge_regression']
        : ['random_forest', 'gradient_boosting', 'ridge_regression']
      : featureCount >= 6 || numericCount >= 4
        ? ['gradient_boosting', 'random_forest', 'logistic_regression']
        : ['random_forest', 'gradient_boosting', 'logistic_regression'];

  for (const modelId of preferredOrder) {
    const match = models.find((model) => model.id === modelId);
    if (match) return match;
  }

  return models.find((model) => model.recommended) ?? models[0] ?? null;
}

function buildModelResultsExplanation(params: {
  selectedTarget: string;
  targetProblemType: ProblemType;
  selectedFeatures: string[];
  selectedModelName: string;
  primaryMetrics: Record<string, number>;
  overfittingStatus: 'healthy' | 'watch' | 'detected';
  cvScores: number[];
  hasFeatureImportance: boolean;
}) {
  const {
    selectedTarget,
    targetProblemType,
    selectedFeatures,
    selectedModelName,
    primaryMetrics,
    overfittingStatus,
    cvScores,
    hasFeatureImportance,
  } = params;

  const primaryMetricEntries = Object.entries(primaryMetrics).slice(0, 2);
  const metricSummary = primaryMetricEntries.length
    ? primaryMetricEntries.map(([key, value]) => `${key}=${formatMetricValue(value)}`).join(', ')
    : 'core metrics were returned';

  return [
    `${selectedModelName} was trained to predict "${selectedTarget}" using ${selectedFeatures.length} selected feature${selectedFeatures.length === 1 ? '' : 's'}, so the reported scores reflect how well that feature set explains the target on this dataset.`,
    `The main validation readout indicates ${metricSummary}, which is the quickest summary of current ${targetProblemType === 'regression' ? 'prediction accuracy for continuous values' : 'classification quality across the detected classes'}.`,
    overfittingStatus === 'healthy'
      ? 'Model health looks stable, which means train, test, and cross-validation behavior are reasonably aligned.'
      : overfittingStatus === 'watch'
      ? 'Model health needs monitoring because the gap between train and validation behavior suggests the model may be learning some dataset-specific patterns.'
      : 'Overfitting has been detected, which means the current model is learning the training slice too specifically and may generalize less reliably to new rows.',
    cvScores.length > 0
      ? `Cross-validation was computed across ${cvScores.length} fold${cvScores.length === 1 ? '' : 's'}, so the result quality is supported by repeated validation rather than a single split.`
      : 'Cross-validation scores were not available for this run, so interpretation should rely more heavily on the main holdout metrics and model-health checks.',
    hasFeatureImportance
      ? 'Feature importance is available, so you can inspect which inputs are driving the model most strongly on this uploaded dataset.'
      : 'Feature importance is limited for this model, so interpretation should focus more on the returned metrics and textual analysis.',
  ];
}

// ─── Difficulty Badge ─────────────────────────────────────────────────────────

function StepIndicator({ currentStep, totalSteps = 6 }: { currentStep: number; totalSteps?: number }) {
  const steps = [
    { num: 1, label: 'Target', icon: Target },
    { num: 2, label: 'Model', icon: BrainCircuit },
    { num: 3, label: 'Config', icon: Gauge },
    { num: 4, label: 'Train', icon: Zap },
    { num: 5, label: 'Compare', icon: GitCompareArrows },
    { num: 6, label: 'Summary', icon: Award },
  ];

  return (
    <div className="flex items-center gap-1 sm:gap-2 overflow-x-auto pb-2">
      {steps.map((step, idx) => {
        const Icon = step.icon;
        const isActive = currentStep === step.num;
        const isDone = currentStep > step.num;
        return (
          <React.Fragment key={step.num}>
            <motion.div
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium whitespace-nowrap transition-colors',
                isActive && 'bg-primary/10 text-primary',
                isDone && 'text-primary',
                !isActive && !isDone && 'text-muted-foreground/50',
              )}
              animate={isActive ? { scale: 1.05 } : { scale: 1 }}
              transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            >
              <div className={cn(
                'flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold',
                isActive && 'bg-primary text-primary-foreground shadow-sm shadow-primary/30',
                isDone && 'bg-primary/10 text-primary',
                !isActive && !isDone && 'bg-muted text-muted-foreground/50',
              )}>
                {isDone ? <CheckCircle2 className="h-3 w-3" /> : step.num}
              </div>
              <span className="hidden sm:inline">{step.label}</span>
            </motion.div>
            {idx < steps.length - 1 && (
              <div className={cn(
                'h-px w-4 sm:w-8',
                isDone ? 'bg-primary/40' : 'bg-border',
              )} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ─── Empty State ──────────────────────────────────────────────────────────────

function EmptyState() {
  const { setActiveTab } = useAppStore();
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-20"
    >
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
        <BrainCircuit className="h-8 w-8 text-primary" />
      </div>
      <h3 className="text-lg font-semibold">No Data Loaded</h3>
      <p className="text-sm text-muted-foreground mt-1 mb-4">Upload a dataset first to start training ML models.</p>
      <Button
        variant="outline"
        className="border-primary/20 text-primary hover:bg-primary/5"
        onClick={() => setActiveTab('upload')}
      >
        Go to Upload
        <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </motion.div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function MlTab() {
  const { toast } = useToast();
  const cleanedData = useAppStore((state) => state.cleanedData);
  const rawData = useAppStore((state) => state.rawData);
  const columns = useAppStore((state) => state.columns);
  const datasetId = useAppStore((state) => state.datasetId);
  const appProblemType = useAppStore((state) => state.problemType);
  const mlWorkflowStep = useAppStore((state) => state.mlWorkflowStep);
  const setMlWorkflowStep = useAppStore((state) => state.setMlWorkflowStep);

  const data = cleanedData || rawData;

  // Wizard state
  const [currentStep, setCurrentStep] = useState(mlWorkflowStep);

  // Step 1 state
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [detectedType, setDetectedType] = useState<ProblemType>('regression');
  const [confidence, setConfidence] = useState(0);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [recommendationReason, setRecommendationReason] = useState('');
  const [targetProblemType, setTargetProblemType] = useState<ProblemType>('regression');

  // Step 2 state
  const [selectedModel, setSelectedModel] = useState<string>('');

  // Step 3 state
  const [testSize, setTestSize] = useState(20);
  const [cvFolds, setCvFolds] = useState(3);
  const [trainingMode, setTrainingMode] = useState<TrainingMode>('balanced');
  const [randomState, setRandomState] = useState(42);

  // Step 4 state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainResults, setTrainResults] = useState<TrainResponse | null>(null);
  const [modelAnalysis, setModelAnalysis] = useState('');
  const [cvScores, setCvScores] = useState<number[]>([]);
  const [overfittingDetected, setOverfittingDetected] = useState(false);
  const [overfittingStatus, setOverfittingStatus] = useState<'healthy' | 'watch' | 'detected'>('healthy');
  const [overfittingExplanation, setOverfittingExplanation] = useState('');
  const [generalizationGap, setGeneralizationGap] = useState<number | null>(null);
  const [cvGap, setCvGap] = useState<number | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Step 5 state
  const [isComparing, setIsComparing] = useState(false);
  const [compareProgress, setCompareProgress] = useState(0);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);
  const [bestModel, setBestModel] = useState<string>('');

  // ML assistant question box
  const [mlQuestion, setMlQuestion] = useState('');
  const [mlAssistantReply, setMlAssistantReply] = useState('');
  const datasetSignatureRef = useRef('');
  const userOverrodeTargetRef = useRef(false);

  const updateMlStep = useCallback((nextStep: number | ((prev: number) => number)) => {
    setCurrentStep((prev) => {
      const resolvedStep = typeof nextStep === 'function' ? nextStep(prev) : nextStep;
      const boundedStep = Math.max(1, Math.min(6, resolvedStep));
      setMlWorkflowStep(boundedStep);
      return boundedStep;
    });
  }, [setMlWorkflowStep]);

  useEffect(() => {
    if (mlWorkflowStep !== currentStep) {
      setCurrentStep(mlWorkflowStep);
    }
  }, [currentStep, mlWorkflowStep]);

  // ─── Step 1: Auto-detect on mount ─────────────────────────────────────────

  useEffect(() => {
    if (!data || columns.length === 0) return;

    const datasetSignature = `${datasetId ?? 'local'}::${columns.map((column) => column.name).join('|')}::${data.length}`;
    const isNewDataset = datasetSignatureRef.current !== datasetSignature;

    if (isNewDataset) {
      datasetSignatureRef.current = datasetSignature;
      userOverrodeTargetRef.current = false;
      setSelectedModel('');
      setTrainResults(null);
      setTrainingMode('balanced');
      updateMlStep(1);
    }

    const autoTarget = getAutoTarget(columns);
    if (!autoTarget) return;

    const hasValidSelectedTarget = columns.some((column) => column.name === selectedTarget);
    if (userOverrodeTargetRef.current && hasValidSelectedTarget && !isNewDataset) {
      return;
    }

    const targetCol = columns.find(c => c.name === autoTarget.target);
    const features = getAutoFeatures(columns, autoTarget.target);
    const featuresChanged =
      selectedFeatures.length !== features.length ||
      selectedFeatures.some((feature, index) => feature !== features[index]);

    let hasUpdates = false;

    if (selectedTarget !== autoTarget.target) {
      setSelectedTarget(autoTarget.target);
      hasUpdates = true;
    }

    if (recommendationReason !== autoTarget.reason) {
      setRecommendationReason(autoTarget.reason);
      hasUpdates = true;
    }

    if (featuresChanged) {
      setSelectedFeatures(features);
      hasUpdates = true;
    }

    if (targetCol) {
      const detection = detectProblemType(targetCol, data);
      if (targetProblemType !== detection.type) {
        setTargetProblemType(detection.type);
        hasUpdates = true;
      }
      if (detectedType !== detection.type) {
        setDetectedType(detection.type);
        hasUpdates = true;
      }
      if (confidence !== detection.confidence) {
        setConfidence(detection.confidence);
        hasUpdates = true;
      }
    }

    if (!hasUpdates) return;
  }, [data, columns, datasetId, selectedTarget, recommendationReason, selectedFeatures, targetProblemType, detectedType, confidence, setMlWorkflowStep]);

  // ─── Update when target changes ───────────────────────────────────────────

  const handleTargetChange = useCallback((newTarget: string) => {
    userOverrodeTargetRef.current = true;
    setSelectedTarget(newTarget);
    const targetCol = columns.find(c => c.name === newTarget);
    if (targetCol && data) {
      const detection = detectProblemType(targetCol, data);
      setTargetProblemType(detection.type);
      setDetectedType(detection.type);
      setConfidence(detection.confidence);
    }
    const features = getAutoFeatures(columns, newTarget);
    setSelectedFeatures(features);
    setRecommendationReason(`Target manually set to "${newTarget}" (${targetCol?.role || 'unknown'} type).`);
    setSelectedModel('');
    setTrainResults(null);
  }, [columns, data]);

  const toggleFeature = useCallback((feature: string) => {
    setSelectedFeatures(prev =>
      prev.includes(feature) ? prev.filter(f => f !== feature) : [...prev, feature]
    );
  }, []);

  // ─── Get models for detected type ─────────────────────────────────────────

  const models = useMemo(() => {
    return targetProblemType === 'regression' ? REGRESSION_MODELS : CLASSIFICATION_MODELS;
  }, [targetProblemType]);

  const recommendedCandidates = useMemo(() => models.filter(m => m.recommended), [models]);
  const recommendedModel = useMemo(() => {
    return getRecommendedStarterModel(models, targetProblemType, selectedFeatures, columns);
  }, [columns, models, selectedFeatures, targetProblemType]);
  const otherModels = useMemo(() => models.filter(m => m.id !== recommendedModel?.id), [models, recommendedModel]);
  const modelSuggestionExplanation = useMemo(() => buildModelSuggestionExplanation({
    selectedTarget,
    targetProblemType,
    confidence,
    selectedFeatures,
    columns,
    recommendedModel,
  }), [selectedTarget, targetProblemType, confidence, selectedFeatures, columns, recommendedModel]);

  const optimizationSummary = trainResults?.optimization;
  const cvWasSkipped = Boolean(optimizationSummary && (optimizationSummary.cv_rows_evaluated ?? 0) === 0);
  const cvDisplayValue = cvScores.length > 0
    ? (cvScores.reduce((a, b) => a + b, 0) / cvScores.length).toFixed(4)
    : cvWasSkipped
      ? 'Skipped'
      : 'N/A';
  const cvStdDisplayValue = cvScores.length > 1
    ? Math.sqrt(cvScores.reduce((sum, v) => sum + (v - cvScores.reduce((a, b) => a + b, 0) / cvScores.length) ** 2, 0) / (cvScores.length - 1)).toFixed(4)
    : cvWasSkipped
      ? 'Skipped'
      : 'N/A';
  const cvGapDisplayValue = cvGap !== null ? cvGap.toFixed(4) : cvWasSkipped ? 'Skipped' : 'N/A';
  const hasMeaningfulFeatureImportance = Boolean(
    trainResults?.feature_importance?.some((item) => Math.abs(item.importance) > 0)
  );

  // ─── Training function ────────────────────────────────────────────────────

  const trainModel = useCallback(async (modelId: string): Promise<TrainResponse | null> => {
    if (!data || !selectedTarget || selectedFeatures.length === 0) {
      toast({ title: 'Configuration Error', description: 'Please complete target and feature selection first.', variant: 'destructive' });
      return null;
    }

    const payload = {
      data: datasetId ? [] : data,
      dataset_id: datasetId ?? null,
      target_column: selectedTarget,
      feature_columns: selectedFeatures,
      problem_type: targetProblemType,
      model_type: modelId,
      test_size: testSize / 100,
      random_state: randomState,
      cv_folds: cvFolds,
      training_mode: trainingMode,
    };

    try {
      const response = await apiClient.post('/train', payload);
      const result: TrainResponse = response.data;
      return result;
    } catch (error: unknown) {
      const msg = getApiErrorMessage(error, 'Training failed');
      throw new Error(msg);
    }
  }, [data, datasetId, selectedTarget, selectedFeatures, targetProblemType, testSize, randomState, cvFolds, trainingMode, toast]);

  // ─── Train selected model (Step 4) ───────────────────────────────────────

  const handleTrain = useCallback(async (): Promise<boolean> => {
    const modelToTrain = selectedModel || recommendedModel?.id;
    if (!modelToTrain) {
      toast({ title: 'No Model Selected', description: 'Please select a model first.', variant: 'destructive' });
      return false;
    }

    setIsTraining(true);
    setTrainingError(null);
    setTrainResults(null);
    setTrainingProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 90) { clearInterval(progressInterval); return 90; }
        return prev + Math.random() * 15;
      });
    }, 300);

    try {
      const result = await trainModel(modelToTrain);
      clearInterval(progressInterval);
      setTrainingProgress(100);

      if (result) {
        setTrainResults(result);
        setModelAnalysis(result.analysis || '');
        setCvScores(result.cv_scores || []);
        setOverfittingDetected(result.overfitting_detected || false);
        setOverfittingStatus(result.overfitting_status || 'healthy');
        setOverfittingExplanation(result.overfitting_explanation || '');
        setGeneralizationGap(typeof result.generalization_gap === 'number' ? result.generalization_gap : null);
        setCvGap(typeof result.cv_gap === 'number' ? result.cv_gap : null);

        // Update store
        useAppStore.setState({
          targetColumn: selectedTarget,
          problemType: targetProblemType,
          selectedFeatures,
          selectedModel: modelToTrain,
          modelId: result.model_id || null,
          modelMetrics: result.metrics,
          modelTrained: true,
          featureImportance: result.feature_importance || [],
          uploadedModel: {
            name: modelToTrain,
            type: modelToTrain,
            target: selectedTarget,
            problem: targetProblemType,
            trainedAt: new Date().toISOString(),
            metrics: result.metrics,
            features: selectedFeatures,
          },
        });

        toast({
          title: 'Model Trained Successfully!',
          description: `${modelToTrain} is ready for predictions.`,
        });
        return true;
      }

      return false;
    } catch (error: unknown) {
      clearInterval(progressInterval);
      const msg = getApiErrorMessage(error, 'Training failed');
      setTrainingError(msg);
      toast({ title: 'Training Failed', description: msg, variant: 'destructive' });
      return false;
    } finally {
      setTimeout(() => setIsTraining(false), 500);
    }
  }, [selectedModel, recommendedModel, trainModel, selectedTarget, targetProblemType, selectedFeatures, toast]);

  const handleTrainAndGoPredict = useCallback(async () => {
    updateMlStep(4);
    const success = await handleTrain();
    if (success) {
      useAppStore.getState().setActiveTab('prediction');
    }
  }, [handleTrain, updateMlStep]);

  // ─── Compare models (Step 5) ─────────────────────────────────────────────

  const activateComparedModel = useCallback((result: ComparisonResult) => {
    setBestModel(result.modelId);
    setSelectedModel(result.modelId);
    setTrainResults({
      model_id: result.modelUuid || '',
      metrics: result.metrics,
      feature_importance: result.featureImportance,
      analysis: result.analysis || '',
      sample_predictions: result.samplePredictions || [],
      cv_scores: result.cvScores || [],
      overfitting_detected: result.overfittingDetected || false,
      overfitting_status: result.overfittingStatus || 'healthy',
      overfitting_explanation: result.overfittingExplanation || '',
      generalization_gap: typeof result.generalizationGap === 'number' ? result.generalizationGap : undefined,
      cv_gap: typeof result.cvGap === 'number' ? result.cvGap : result.cvGap ?? null,
      training_time: result.trainingTime || 0,
      optimization: result.optimization,
    });
    setModelAnalysis(result.analysis || '');
    setCvScores(result.cvScores || []);
    setOverfittingDetected(result.overfittingDetected || false);
    setOverfittingStatus(result.overfittingStatus || 'healthy');
    setOverfittingExplanation(result.overfittingExplanation || '');
    setGeneralizationGap(typeof result.generalizationGap === 'number' ? result.generalizationGap : null);
    setCvGap(typeof result.cvGap === 'number' ? result.cvGap : result.cvGap ?? null);

    useAppStore.setState({
      targetColumn: selectedTarget,
      problemType: targetProblemType,
      selectedFeatures,
      selectedModel: result.modelId,
      modelId: result.modelUuid || null,
      modelMetrics: result.metrics,
      modelTrained: true,
      featureImportance: result.featureImportance,
      uploadedModel: {
        name: result.modelId,
        type: result.modelId,
        target: selectedTarget,
        problem: targetProblemType,
        trainedAt: new Date().toISOString(),
        metrics: result.metrics,
        features: selectedFeatures,
      },
    });
  }, [selectedFeatures, selectedTarget, targetProblemType]);

  const handleCompare = useCallback(async () => {
    const modelsToCompare = recommendedCandidates.map(m => m.id);
    if (modelsToCompare.length === 0) return;

    setIsComparing(true);
    setCompareProgress(0);
    setComparisonResults([]);
    setBestModel('');

    const results: ComparisonResult[] = [];
    const totalModels = modelsToCompare.length;

    for (let i = 0; i < modelsToCompare.length; i++) {
      setCompareProgress(((i) / totalModels) * 100);
      try {
        const result = await trainModel(modelsToCompare[i]);
        if (result) {
          results.push({
            modelId: modelsToCompare[i],
            modelUuid: result.model_id || '',
            modelName: modelsToCompare[i].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            metrics: result.metrics,
            featureImportance: result.feature_importance || [],
            analysis: result.analysis || '',
            cvScores: result.cv_scores || [],
            trainingTime: result.training_time || 0,
            samplePredictions: result.sample_predictions || [],
            overfittingDetected: result.overfitting_detected || false,
            overfittingStatus: result.overfitting_status || 'healthy',
            overfittingExplanation: result.overfitting_explanation || '',
            generalizationGap: typeof result.generalization_gap === 'number' ? result.generalization_gap : undefined,
            cvGap: typeof result.cv_gap === 'number' ? result.cv_gap : result.cv_gap ?? null,
            optimization: result.optimization,
          });
        }
      } catch {
        // skip failed models
      }
    }

    setCompareProgress(100);
    setComparisonResults(results);

    if (results.length > 0) {
      // Find best model based on primary metric
      const primaryMetric = targetProblemType === 'regression' ? 'R2' : 'Accuracy';
      let best = results[0];
      for (const r of results) {
        if ((r.metrics[primaryMetric] || 0) > (best.metrics[primaryMetric] || 0)) {
          best = r;
        }
      }
      activateComparedModel(best);

      toast({ title: 'Comparison Complete', description: `${best.modelName} is the best performing model.` });
    }

    setTimeout(() => setIsComparing(false), 500);
  }, [activateComparedModel, recommendedCandidates, trainModel, targetProblemType, toast]);

  // ─── Navigation ───────────────────────────────────────────────────────────



  const goToPredict = useCallback(() => {
    useAppStore.getState().setActiveTab('prediction');
  }, []);

  
  const handleAskAssistant = useCallback(() => {
    const question = mlQuestion.trim();
    if (!question) {
      toast({ title: 'Enter a problem', description: 'Describe the modeling problem, training quality concern, or prediction behavior you want help with.', variant: 'destructive' });
      return;
    }

    const lower = question.toLowerCase();
    const response: string[] = [];
    response.push(`Problem: ${question}`);

    if (!selectedTarget) {
      response.push('Choose a target column first so the assistant can give dataset-specific ML guidance.');
      setMlAssistantReply(response.join('\n\n'));
      return;
    }

    response.push(`Current setup: predicting ${selectedTarget} as a ${targetProblemType} task using ${selectedFeatures.length} feature(s) from ${data?.length ?? 0} row(s).`);

    if (selectedModel) {
      response.push(`Selected model: ${selectedModel.replace(/_/g, ' ')}.`);
    }

    if (trainResults) {
      const metricsSummary = Object.entries(trainResults.metrics || {})
        .slice(0, 4)
        .map(([key, value]) => `${key}: ${typeof value === 'number' ? formatMetricValue(value) : value}`)
        .join(', ');
      if (metricsSummary) {
        response.push(`Latest training metrics: ${metricsSummary}.`);
      }
    } else {
      response.push('No trained model is available yet, so the solution is based on the current configuration rather than measured performance.');
    }

    if (/(feature|important|importance)/.test(lower)) {
      if (trainResults?.feature_importance?.length) {
        const top = trainResults.feature_importance.slice(0, 3).map((item) => `${item.name} (${item.importance.toFixed(3)})`).join(', ');
        response.push(`Top features right now: ${top}.`);
      } else {
        response.push('Train the model to generate feature importance rankings.');
      }
    }

    if (/(predict|prediction|forecast)/.test(lower)) {
      response.push(
        targetProblemType === 'classification'
          ? 'This model will predict class labels and, when supported, class probabilities for each record.'
          : 'This model will predict a continuous numeric value for each record.'
      );
    }

    if (/(improve|better|accuracy|overfit|underfit|quality)/.test(lower)) {
      response.push('To improve results, review feature quality, remove leakage and identifier columns, verify the target is correct, and start with the single best suggested model.');
    }

    if (/(which model|best model|recommend)/.test(lower)) {
      response.push(`Recommended next step: ${selectedModel || recommendedModel?.name || 'use the best suggested model'} and run training so performance can be compared with evidence.`);
    }

    if (response.length <= 3) {
      response.push('Describe model selection, feature importance, prediction behavior, or training quality to get a more targeted solution.');
    }

    setMlAssistantReply(response.join('\n\n'));
  }, [data, mlQuestion, selectedFeatures.length, selectedModel, selectedTarget, targetProblemType, toast, recommendedModel, trainResults]);

  const canAdvanceMlStep =
    currentStep === 1
      ? Boolean(selectedTarget && selectedFeatures.length > 0)
      : currentStep === 2
        ? Boolean(selectedModel)
        : currentStep === 3
          ? true
          : currentStep === 4
            ? Boolean(trainResults)
            : currentStep === 5
              ? Boolean(trainResults)
              : Boolean(trainResults);

  const nextMlStepLabel =
    currentStep === 1
      ? 'Review Target and Continue'
      : currentStep === 2
        ? 'Continue to Training Setup'
        : currentStep === 3
          ? 'Continue to Training'
          : currentStep === 4
            ? 'Continue to Comparison'
            : currentStep === 5
              ? 'Continue to Summary'
              : 'Open Prediction';

  const mlStepHint =
    currentStep === 1
      ? 'Review the suggested target and selected features first, then click continue when you are ready for model selection.'
      : currentStep === 2
        ? 'Pick one model to continue to the training configuration step.'
        : currentStep === 3
          ? 'Review the training setup, then move on to run the model.'
          : currentStep === 4
            ? 'Train the model first. After results appear, you can continue.'
            : currentStep === 5
              ? 'You can compare models here, then move to the final summary.'
              : 'The model summary is ready. Open Prediction when you want to score new rows.';

  const goToPrevMlStep = useCallback(() => {
    updateMlStep((step) => step - 1);
  }, [updateMlStep]);

  const goToNextMlStep = useCallback(() => {
    if (currentStep >= 6) {
      useAppStore.getState().setActiveTab('prediction');
      return;
    }

    if (!canAdvanceMlStep) return;
    updateMlStep((step) => step + 1);
  }, [canAdvanceMlStep, currentStep, updateMlStep]);


  const primaryMetrics = targetProblemType === 'regression'
    ? { 'R2': trainResults?.metrics?.['R2'] ?? trainResults?.metrics?.['r2'] ?? null, 'RMSE': trainResults?.metrics?.['RMSE'] ?? null, 'MAE': trainResults?.metrics?.['MAE'] ?? null }
    : { 'Accuracy': trainResults?.metrics?.['Accuracy'] ?? null, 'Precision': trainResults?.metrics?.['Precision'] ?? null, 'Recall': trainResults?.metrics?.['Recall'] ?? null, 'F1': trainResults?.metrics?.['F1'] ?? trainResults?.metrics?.['F1 Score'] ?? null };
  const modelResultsExplanation = useMemo(() => buildModelResultsExplanation({
    selectedTarget,
    targetProblemType,
    selectedFeatures,
    selectedModelName: (selectedModel || recommendedModel?.name || 'Selected model').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    primaryMetrics: Object.fromEntries(Object.entries(primaryMetrics).filter((entry): entry is [string, number] => entry[1] !== null)),
    overfittingStatus,
    cvScores,
    hasFeatureImportance: hasMeaningfulFeatureImportance,
  }), [selectedTarget, targetProblemType, selectedFeatures, selectedModel, recommendedModel, primaryMetrics, overfittingStatus, cvScores, hasMeaningfulFeatureImportance]);

  const metricIcons: Record<string, React.ElementType> = {
    'R2': TrendingUp, 'RMSE': Activity, 'MAE': BarChart3,
    'Accuracy': Target, 'Precision': Gauge, 'Recall': Eye, 'F1': Shield,
  };


  const actualVsPredictedData = useMemo(() => {
    const numericPoints = (trainResults?.sample_predictions ?? [])
      .map((point, index) => {
        const actual = typeof point.actual === 'number' ? point.actual : Number(point.actual);
        const predicted = typeof point.predicted === 'number' ? point.predicted : Number(point.predicted);
        if (Number.isNaN(actual) || Number.isNaN(predicted)) return null;
        return { index: index + 1, actual, predicted };
      })
      .filter((point): point is { index: number; actual: number; predicted: number } => point !== null);

    if (numericPoints.length === 0) return [];

    const values = numericPoints.flatMap((point) => [point.actual, point.predicted]);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    const standardDeviation = Math.sqrt(variance);

    return numericPoints.map((point) => ({
      ...point,
      actualStandardized: standardDeviation === 0 ? 0 : (point.actual - mean) / standardDeviation,
      predictedStandardized: standardDeviation === 0 ? 0 : (point.predicted - mean) / standardDeviation,
    }));
  }, [trainResults]);

  const actualVsPredictedDomain = useMemo(() => {
    if (actualVsPredictedData.length === 0) return [-1, 1] as [number, number];
    const values = actualVsPredictedData.flatMap((point) => [point.actualStandardized, point.predictedStandardized]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return [min - 1, max + 1] as [number, number];
    const padding = (max - min) * 0.05;
    return [min - padding, max + padding] as [number, number];
  }, [actualVsPredictedData]);


  const formatPredictionChartValue = (value: number) =>
    value.toLocaleString(undefined, { maximumFractionDigits: 3 });

  const getMetricColor = (key: string, value: number | null) => {
    if (value === null) return 'text-muted-foreground';
    const good = ['R²', 'Accuracy', 'Precision', 'Recall', 'F1'];
    if (good.includes(key)) {
      return value >= 0.9 ? 'text-primary' : value >= 0.7 ? 'text-amber-500' : 'text-rose-500';
    }
    // For RMSE, MAE - lower is better, we can't easily judge without context
    return 'text-primary';
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BrainCircuit className="h-6 w-6 text-primary" />
            ML Assistant
          </h2>
          <p className="text-muted-foreground mt-1">Train and evaluate machine learning models with AI-powered guidance.</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">
            <Bot className="mr-1 h-3 w-3" />
            {data?.length ?? 0} rows
          </Badge>
          <Badge variant="outline" className="border-secondary bg-secondary text-secondary-foreground">
            {columns.length} features
          </Badge>
        </div>
      </motion.div>

      
      {/* Step Indicator */}
      <StepIndicator currentStep={currentStep} />

      <Card className="border border-primary/20 bg-gradient-to-br from-primary/6 to-secondary/70">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Bot className="h-4 w-4 text-primary" />
            <CardTitle className="text-base">Ask The ML Assistant</CardTitle>
          </div>
          <CardDescription>Enter a problem about model analysis, training quality, or prediction behavior.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={mlQuestion}
            onChange={(e) => setMlQuestion(e.target.value)}
            placeholder="Example: Which model should I try first for this target, and how will predictions behave?"
            className="min-h-24 resize-y bg-background/80"
          />
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
            <Button
              onClick={handleAskAssistant}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <Sparkles className="mr-2 h-4 w-4" />
              Analyze Problem
            </Button>
            <p className="text-xs text-muted-foreground sm:pt-2">
              The solution uses your current target, selected features, chosen model, and latest training results when available.
            </p>
          </div>
          {mlAssistantReply && (
            <div className="rounded-lg border bg-background/70 p-4 text-sm text-muted-foreground whitespace-pre-wrap">
              {mlAssistantReply}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
        >
          {/* ─── STEP 1: Target & Problem Detection ──────────────────────── */}
          {currentStep === 1 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              {/* Auto Detection Card */}
              <motion.div variants={itemVariants}>
                <Card className="border border-primary/20 bg-gradient-to-br from-primary/6 to-secondary/70">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                        <Sparkles className="h-4 w-4 text-primary" />
                      </div>
                      <CardTitle className="text-lg">Smart Recommendation</CardTitle>
                    </div>
                    <CardDescription>AI-powered target detection and feature selection</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Problem Type Detection */}
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        'flex h-10 w-10 items-center justify-center rounded-xl',
                        detectedType === 'regression' ? 'bg-primary/10' : 'bg-secondary',
                      )}>
                        {detectedType === 'regression'
                          ? <TrendingUp className="h-5 w-5 text-primary" />
                          : <Tag className="h-5 w-5 text-secondary-foreground" />
                        }
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium capitalize">{detectedType}</p>
                          <Badge variant="outline" className={cn(
                            'text-xs',
                            confidence >= 80 ? 'border-primary/20 bg-primary/10 text-primary'
                              : confidence >= 60 ? 'border-amber-500/20 bg-amber-500/10 text-amber-600 dark:text-amber-400'
                              : 'border-rose-500/20 bg-rose-500/10 text-rose-600 dark:text-rose-400',
                          )}>
                            {confidence}% confidence
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {detectedType === 'regression'
                            ? 'Numeric target detected — model will predict continuous values'
                            : 'Categorical target detected — model will classify into categories'}
                        </p>
                      </div>
                    </div>

                    {recommendationReason && (
                      <div className="flex gap-2 p-3 rounded-lg bg-muted/50">
                        <Lightbulb className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                        <p className="text-xs text-muted-foreground">{recommendationReason}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Target & Features Selection */}
              <div className="grid gap-6 md:grid-cols-2">
                {/* Target Column */}
                <motion.div variants={itemVariants}>
                  <Card>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-primary" />
                        <CardTitle className="text-base">Target Column</CardTitle>
                      </div>
                      <CardDescription>Select the variable to predict</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Select value={selectedTarget} onValueChange={handleTargetChange}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select target column..." />
                        </SelectTrigger>
                        <SelectContent className="max-h-64">
                          {columns.map(col => (
                            <SelectItem key={col.name} value={col.name}>
                              <div className="flex items-center gap-2">
                                {col.role === 'numeric' ? <Hash className="h-3 w-3 text-primary" /> : <Tag className="h-3 w-3 text-secondary-foreground" />}
                                <span>{col.name}</span>
                                <span className="text-xs text-muted-foreground">({col.dtype})</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Problem Type */}
                <motion.div variants={itemVariants}>
                  <Card>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <BrainCircuit className="h-4 w-4 text-primary" />
                        <CardTitle className="text-base">Problem Type</CardTitle>
                      </div>
                      <CardDescription>Auto-detected based on target column</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex gap-3">
                        <button
                          onClick={() => setTargetProblemType('regression')}
                          className={cn(
                            'flex-1 rounded-lg border-2 p-3 text-center transition-all',
                            targetProblemType === 'regression'
                              ? 'border-primary bg-primary/5'
                              : 'border-transparent bg-muted/50 hover:bg-muted',
                          )}
                        >
                          <TrendingUp className={cn('mx-auto mb-1 h-5 w-5', targetProblemType === 'regression' ? 'text-primary' : 'text-muted-foreground')} />
                          <p className="text-sm font-medium">Regression</p>
                          <p className="text-xs text-muted-foreground">Predict numbers</p>
                        </button>
                        <button
                          onClick={() => setTargetProblemType('classification')}
                          className={cn(
                            'flex-1 rounded-lg border-2 p-3 text-center transition-all',
                            targetProblemType === 'classification'
                              ? 'border-secondary-foreground/20 bg-secondary'
                              : 'border-transparent bg-muted/50 hover:bg-muted',
                          )}
                        >
                          <Tag className={cn('mx-auto mb-1 h-5 w-5', targetProblemType === 'classification' ? 'text-secondary-foreground' : 'text-muted-foreground')} />
                          <p className="text-sm font-medium">Classification</p>
                          <p className="text-xs text-muted-foreground">Predict categories</p>
                        </button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </div>

              {/* Feature Selection */}
              <motion.div variants={itemVariants}>
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-primary" />
                        <CardTitle className="text-base">Feature Selection</CardTitle>
                        <Badge variant="secondary" className="text-xs">{selectedFeatures.length} selected</Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs text-muted-foreground"
                        onClick={() => {
                          const allFeatures = getAutoFeatures(columns, selectedTarget);
                          setSelectedFeatures(allFeatures);
                        }}
                      >
                        Reset All
                      </Button>
                    </div>
                    <CardDescription>Choose features for model training (excludes target & ID columns)</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-64 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
                      {getAutoFeatures(columns, selectedTarget).map(feature => {
                        const col = columns.find(c => c.name === feature);
                        const isSelected = selectedFeatures.includes(feature);
                        return (
                          <motion.label
                            key={feature}
                            className={cn(
                              'flex items-center gap-3 rounded-lg px-3 py-2 cursor-pointer transition-colors',
                              isSelected ? 'bg-primary/5' : 'hover:bg-muted/50',
                            )}
                            whileHover={{ x: 2 }}
                            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                          >
                            <Checkbox
                              checked={isSelected}
                              onCheckedChange={() => toggleFeature(feature)}
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{feature}</p>
                            </div>
                            {col && (
                              <Badge variant="outline" className="text-[10px] shrink-0">
                                {col.role === 'numeric' ? <Hash className="mr-1 h-2.5 w-2.5 text-primary" /> : <Tag className="mr-1 h-2.5 w-2.5 text-secondary-foreground" />}
                                {col.dtype}
                              </Badge>
                            )}
                          </motion.label>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          )}

          {/* ─── STEP 2: Model Recommendations ───────────────────────────── */}
          {currentStep === 2 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              <motion.div variants={itemVariants}>
                <div className="flex items-center gap-2 mb-4">
                  <Star className="h-5 w-5 text-amber-500" />
                  <h3 className="text-lg font-semibold">Best Model Suggestion</h3>
                </div>
                <Card className="mb-4 border-primary/15 bg-primary/5">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-primary" />
                      <CardTitle className="text-base">Why This Suggestion Fits This Dataset</CardTitle>
                    </div>
                    <CardDescription>Universal explainability based on the detected target, feature mix, and model behavior on structured data.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {modelSuggestionExplanation.map((line) => (
                      <div key={line} className="flex items-start gap-2 rounded-xl border bg-background/70 p-3">
                        <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                        <p className="text-sm leading-6 text-muted-foreground">{line}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
                {recommendedModel && (
                  <motion.div variants={scaleVariants}>
                    <Card
                      className={cn(
                        'cursor-pointer transition-all duration-200 hover:shadow-lg hover:shadow-primary/10',
                        selectedModel === recommendedModel.id
                          ? 'border-primary bg-primary/5 ring-2 ring-primary/20'
                          : 'hover:border-primary/30',
                      )}
                      onClick={() => setSelectedModel(recommendedModel.id)}
                      whileHover={{ y: -2, scale: 1.01 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <CardHeader className="pb-2">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2">
                            <div className={cn(
                              'flex h-8 w-8 items-center justify-center rounded-lg',
                              selectedModel === recommendedModel.id ? 'bg-primary text-primary-foreground' : 'bg-primary/10 text-primary',
                            )}>
                              <BrainCircuit className="h-4 w-4" />
                            </div>
                            <CardTitle className="text-base">{recommendedModel.name}</CardTitle>
                          </div>
                          {selectedModel === recommendedModel.id && (
                            <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}>
                              <CheckCircle2 className="h-5 w-5 text-primary" />
                            </motion.div>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <p className="text-sm text-muted-foreground">{recommendedModel.description}</p>
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge className="bg-primary text-primary-foreground text-[10px]">Recommended First</Badge>
                        </div>
                        <div className="rounded-lg border bg-muted/30 px-3 py-3 text-sm text-muted-foreground">
                          <span className="font-medium text-foreground">Why this model:</span> {recommendedModel.whyRecommended}
                        </div>
                        <div className="rounded-lg border border-primary/15 bg-primary/5 px-3 py-3 text-sm text-muted-foreground">
                          <span className="font-medium text-foreground">Why it is suggested:</span> This recommendation is based on target type, feature composition, and expected tabular-data behavior. Final selection should still be validated in the comparison step.
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </motion.div>

              {/* Other Models */}
              {otherModels.length > 0 && (
                <motion.div variants={itemVariants}>
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="h-5 w-5 text-muted-foreground" />
                    <h3 className="text-lg font-semibold">Other Available Models</h3>
                    <Badge variant="secondary" className="text-xs">{otherModels.length} models</Badge>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {otherModels.map(model => (
                      <motion.div key={model.id} variants={scaleVariants}>
                        <Card
                          className={cn(
                            'cursor-pointer transition-all duration-200 hover:shadow-md',
                            selectedModel === model.id
                              ? 'border-primary bg-primary/5 ring-2 ring-primary/20'
                              : 'hover:border-primary/30',
                          )}
                          onClick={() => setSelectedModel(model.id)}
                          whileHover={{ y: -1, scale: 1.005 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <div className={cn(
                                  'flex h-8 w-8 items-center justify-center rounded-lg',
                                  selectedModel === model.id ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground',
                                )}>
                                  <BrainCircuit className="h-4 w-4" />
                                </div>
                                <div>
                                  <p className="text-sm font-medium">{model.name}</p>
                                  <p className="text-xs text-muted-foreground mt-0.5">{model.description}</p>
                                </div>
                              </div>
                              <div className="flex flex-col items-end gap-1">
                                {selectedModel === model.id && (
                                  <CheckCircle2 className="h-5 w-5 text-primary" />
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* ─── STEP 3: Training Configuration ───────────────────────────── */}
          {currentStep === 3 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              {/* Config Card */}
              <motion.div variants={itemVariants}>
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Gauge className="h-5 w-5 text-primary" />
                      <CardTitle>Training Configuration</CardTitle>
                    </div>
                    <CardDescription>Configure hyperparameters for model training</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-8">
                    {/* Test/Train Split */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium">Train/Test Split</p>
                          <p className="text-xs text-muted-foreground">Percentage of data reserved for testing</p>
                        </div>
                        <Badge variant="outline" className="font-mono text-xs">
                          {(testSize / 100).toFixed(2)}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-xs text-muted-foreground w-8">10%</span>
                        <Slider
                          value={[testSize]}
                          onValueChange={(v) => setTestSize(v[0])}
                          min={10}
                          max={40}
                          step={5}
                          className="flex-1"
                        />
                        <span className="text-xs text-muted-foreground w-8">40%</span>
                      </div>
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Train: {(100 - testSize)}%</span>
                        <span>Test: {testSize}%</span>
                      </div>
                    </div>

                    {/* CV Folds */}
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium">Cross-Validation Folds</p>
                        <p className="text-xs text-muted-foreground">Number of folds for k-fold cross validation</p>
                      </div>
                      <div className="flex gap-3">
                        {[3, 5, 10].map(fold => (
                          <motion.button
                            key={fold}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setCvFolds(fold)}
                            className={cn(
                              'flex-1 rounded-lg border-2 p-3 text-center transition-all',
                              cvFolds === fold
                                ? 'border-primary bg-primary/5'
                                : 'border-transparent bg-muted/50 hover:bg-muted',
                            )}
                          >
                            <p className="text-lg font-bold">{fold}</p>
                            <p className="text-xs text-muted-foreground">folds</p>
                          </motion.button>
                        ))}
                      </div>
                    </div>

                    {/* Training Mode */}
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium">Training Mode</p>
                        <p className="text-xs text-muted-foreground">Balanced mode runs cross-validation on a smaller sample so large datasets still get CV metrics.</p>
                      </div>
                      <Select value={trainingMode} onValueChange={(value) => setTrainingMode(value as TrainingMode)}>
                        <SelectTrigger className="w-full md:w-72">
                          <SelectValue placeholder="Select training mode" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="balanced">Balanced</SelectItem>
                          <SelectItem value="fast">Fast</SelectItem>
                        </SelectContent>
                      </Select>
                      <div className="rounded-lg border bg-muted/30 px-3 py-3 text-xs text-muted-foreground">
                        {trainingMode === 'balanced'
                          ? 'Balanced mode keeps large-dataset sampling, but still computes CV on a smaller sample for more complete metrics.'
                          : 'Fast mode skips CV on large datasets to return results as quickly as possible.'}
                      </div>
                    </div>

                    {/* Random State */}
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium">Random State</p>
                        <p className="text-xs text-muted-foreground">Seed for reproducible results</p>
                      </div>
                      <input
                        type="number"
                        value={randomState}
                        onChange={(e) => setRandomState(parseInt(e.target.value) || 0)}
                        className="w-32 rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                      />
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Summary + Action Buttons */}
              <motion.div variants={itemVariants}>
                <Card className="border-dashed">
                  <CardContent className="p-6">
                    <div className="flex flex-col sm:flex-row items-center gap-4">
                      <div className="flex-1 text-center sm:text-left">
                        <p className="text-sm font-medium">Ready to Train</p>
                        <p className="text-xs text-muted-foreground">
                          {selectedModel ? `Selected: ${selectedModel}` : 'No model selected — auto-train will use best suggested model'}
                          {' · '}{selectedFeatures.length} features · {data?.length} rows
                        </p>
                      </div>
                      <div className="flex gap-3">
                        <Button
                          onClick={() => {
                            setSelectedModel(recommendedModel?.id || '');
                            setTimeout(() => { updateMlStep(4); handleTrain(); }, 100);
                          }}
                          className="bg-primary text-primary-foreground hover:bg-primary/90"
                        >
                          <Zap className="mr-2 h-4 w-4" />
                          Auto-Train Best
                        </Button>
                        <Button
                          onClick={handleTrainAndGoPredict}
                          variant="outline"
                          className="border-secondary text-secondary-foreground"
                        >
                          <ArrowRight className="mr-2 h-4 w-4" />
                          One-Click Train To Prediction
                        </Button>
                        {selectedModel && (
                          <Button
                            onClick={() => {
                              updateMlStep(4);
                              setTimeout(handleTrain, 300);
                            }}
                            variant="outline"
                            className="border-primary/20 text-primary"
                          >
                            <Play className="mr-2 h-4 w-4" />
                            Train Selected
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          )}
          {/* ─── STEP 4: Training Progress & Results ──────────────────────── */}
          {currentStep === 4 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              {/* Training Progress */}
              {(isTraining || (!trainResults && !trainingError)) && (
                <motion.div variants={itemVariants}>
                  <Card className="border border-primary/20">
                    <CardContent className="p-8 text-center">
                      <AnimatePresence mode="wait">
                        {isTraining ? (
                          <motion.div
                            key="training"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="space-y-4"
                          >
                            <div className="relative mx-auto h-20 w-20">
                              <motion.div
                                className="absolute inset-0 rounded-full border-4 border-primary/20"
                                style={{ borderTopColor: 'hsl(var(--primary))' }}
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                              />
                              <div className="absolute inset-0 flex items-center justify-center">
                                <BrainCircuit className="h-8 w-8 text-primary" />
                              </div>
                            </div>
                            <p className="text-sm font-medium">Training Model...</p>
                            <p className="text-xs text-muted-foreground">{selectedModel || recommendedModel?.id}</p>
                            <div className="max-w-xs mx-auto">
                              <Progress value={Math.min(trainingProgress, 100)} className="h-2" />
                              <p className="text-xs text-muted-foreground mt-1">{Math.round(trainingProgress)}%</p>
                            </div>
                          </motion.div>
                        ) : (
                          <motion.div key="ready" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                            <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
                            <p className="text-sm text-muted-foreground">Click below to start training</p>
                            <Button
                              onClick={handleTrain}
                              className="bg-primary text-primary-foreground hover:bg-primary/90"
                            >
                              <Play className="mr-2 h-4 w-4" />
                              Start Training
                            </Button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Training Error */}
              {trainingError && (
                <motion.div variants={itemVariants}>
                  <Card className="border-rose-500/20 bg-rose-500/5">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="h-5 w-5 text-rose-500 mt-0.5" />
                        <div>
                          <p className="text-sm font-medium text-rose-600 dark:text-rose-400">Training Failed</p>
                          <p className="text-xs text-muted-foreground mt-1">{trainingError}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Results */}
                  {trainResults && (
                <>
                  {/* Success Banner */}
                  <motion.div variants={itemVariants}>
                    <Card className="border border-primary/20 bg-gradient-to-r from-primary/6 to-secondary/70">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-3">
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: 'spring', stiffness: 300, damping: 15 }}
                          >
                            <CheckCircle2 className="h-6 w-6 text-primary" />
                          </motion.div>
                          <div>
                            <p className="text-sm font-medium">Training Complete!</p>
                            <p className="text-xs text-muted-foreground">
                              {selectedModel || recommendedModel?.id} trained on {selectedFeatures.length} features
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <Card className="border-primary/15 bg-primary/5">
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                          <Info className="h-4 w-4 text-primary" />
                          <CardTitle className="text-base">How To Read These Results</CardTitle>
                        </div>
                        <CardDescription>Universal interpretation guidance for the trained model and this uploaded dataset.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {modelResultsExplanation.map((line) => (
                          <div key={line} className="flex items-start gap-2 rounded-xl border bg-background/70 p-3">
                            <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
                            <p className="text-sm leading-6 text-muted-foreground">{line}</p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  </motion.div>

                  {/* Metrics Cards */}
                  <motion.div variants={itemVariants}>
                    <h3 className="text-base font-semibold mb-3 flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-primary" />
                      Performance Metrics
                    </h3>
                    <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
                      {Object.entries(primaryMetrics).map(([key, value]) => {
                        const Icon = metricIcons[key] || BarChart3;
                        return (
                          <motion.div key={key} variants={scaleVariants}>
                            <Card className="text-center">
                              <CardContent className="p-4">
                                <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
                                  <Icon className="h-5 w-5 text-primary" />
                                </div>
                                <p className={cn('text-2xl font-bold', getMetricColor(key, value))}>
                                  {value !== null ? formatMetricValue(value) : '—'}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1">{key}</p>
                              </CardContent>
                            </Card>
                          </motion.div>
                        );
                      })}
                    </div>
                  </motion.div>

                  {optimizationSummary && (
                    <motion.div variants={itemVariants}>
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center gap-2">
                            <Gauge className="h-4 w-4 text-primary" />
                            <CardTitle className="text-base">Large Dataset Optimization</CardTitle>
                          </div>
                          <CardDescription>Training performance details for this run</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                            <div className="rounded-lg border px-3 py-2">
                              <p className="text-xs text-muted-foreground">Rows available</p>
                              <p className="text-sm font-semibold mt-1">{(optimizationSummary.training_rows_available ?? data?.length ?? 0).toLocaleString()}</p>
                            </div>
                            <div className="rounded-lg border px-3 py-2">
                              <p className="text-xs text-muted-foreground">Rows used to train</p>
                              <p className="text-sm font-semibold mt-1">{(optimizationSummary.training_rows_used ?? data?.length ?? 0).toLocaleString()}</p>
                            </div>
                            <div className="rounded-lg border px-3 py-2">
                              <p className="text-xs text-muted-foreground">CV rows evaluated</p>
                              <p className="text-sm font-semibold mt-1">{(optimizationSummary.cv_rows_evaluated ?? 0).toLocaleString()}</p>
                            </div>
                            <div className="rounded-lg border px-3 py-2">
                              <p className="text-xs text-muted-foreground">Sampling</p>
                              <p className="text-sm font-semibold mt-1">{optimizationSummary.training_sampled ? 'Enabled for speed' : 'Full training rows used'}</p>
                            </div>
                            <div className="rounded-lg border px-3 py-2">
                              <p className="text-xs text-muted-foreground">Mode</p>
                              <p className="text-sm font-semibold mt-1 capitalize">{optimizationSummary.training_mode ?? trainingMode}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {/* Overfitting & CV */}
                  <div className="grid gap-4 md:grid-cols-2">
                    <motion.div variants={itemVariants}>
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center gap-2">
                            <Shield className="h-4 w-4 text-primary" />
                            <CardTitle className="text-base">Model Health</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-sm">Overfitting</span>
                              <Badge
                                variant={overfittingStatus === 'detected' ? 'destructive' : 'outline'}
                                className={cn(
                                  overfittingStatus === 'healthy' && 'border-primary/20 bg-primary/10 text-primary',
                                  overfittingStatus === 'watch' && 'border-amber-500/20 bg-amber-500/10 text-amber-600 dark:text-amber-400',
                                )}
                              >
                                {overfittingStatus === 'detected' ? (
                                  <><AlertTriangle className="mr-1 h-3 w-3" />Detected</>
                                ) : overfittingStatus === 'watch' ? (
                                  <><AlertTriangle className="mr-1 h-3 w-3" />Watch</>
                                ) : (
                                  <><CheckCircle2 className="mr-1 h-3 w-3" />Healthy</>
                                )}
                              </Badge>
                            </div>
                            <div className="rounded-lg bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                              {overfittingExplanation || 'Overfitting health is estimated from train, test, and cross-validation performance gaps.'}
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">CV Folds</span>
                              <span className="text-sm font-medium">{cvFolds}</span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">CV Mean</span>
                              <span className="text-sm font-medium text-primary">
                                {cvDisplayValue}
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">CV Std</span>
                              <span className="text-sm font-medium">
                                {cvStdDisplayValue}
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">Train-Test Gap</span>
                              <span className={cn('text-sm font-medium', generalizationGap !== null && generalizationGap > 0.08 && 'text-amber-600 dark:text-amber-400')}>
                                {generalizationGap !== null ? generalizationGap.toFixed(4) : '?'}
                              </span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">Train-CV Gap</span>
                              <span className={cn('text-sm font-medium', cvGap !== null && cvGap > 0.08 && 'text-amber-600 dark:text-amber-400')}>
                                {cvGapDisplayValue}
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>

                    {/* CV Scores Detail */}
                    <motion.div variants={itemVariants}>
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center gap-2">
                            <Activity className="h-4 w-4 text-primary" />
                            <CardTitle className="text-base">Cross-Validation Scores</CardTitle>
                          </div>
                        </CardHeader>
                        <CardContent>
                          {cvScores.length > 0 ? (
                            <div className="space-y-2">
                              {cvScores.map((score, idx) => {
                                const mean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
                                const isAboveMean = score >= mean;
                                return (
                                  <div key={idx} className="flex items-center gap-2">
                                    <span className="text-xs text-muted-foreground w-12">Fold {idx + 1}</span>
                                    <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                                      <motion.div
                                        className={cn('h-full rounded-full', isAboveMean ? 'bg-primary' : 'bg-amber-500')}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min(score * 100, 100)}%` }}
                                        transition={{ duration: 0.6, delay: idx * 0.08, ease: 'easeOut' }}
                                      />
                                    </div>
                                    <span className="text-xs font-mono w-16 text-right">{score.toFixed(4)}</span>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <p className="text-xs text-muted-foreground">{cvWasSkipped ? 'Cross-validation skipped for this large dataset to keep training responsive.' : optimizationSummary?.cv_sampled ? 'Cross-validation was computed on a smaller sample to keep large-dataset training responsive.' : 'No CV scores available'}</p>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  </div>

                  {/* Feature Importance Chart */}
                  {hasMeaningfulFeatureImportance && (
                    <motion.div variants={itemVariants}>
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center gap-2">
                            <BarChart3 className="h-4 w-4 text-primary" />
                            <CardTitle className="text-base">Feature Importance</CardTitle>
                          </div>
                          <CardDescription>Relative importance of each feature in model predictions</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={trainResults.feature_importance.slice(0, 15)} layout="vertical" margin={{ left: 80, right: 20, top: 5, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={ML_CHART_COLORS.grid} />
                                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#475569' }} width={120} />
                                <RechartsTooltip
                                  contentStyle={{
                                    backgroundColor: 'hsl(var(--popover))',
                                    border: '1px solid hsl(var(--border))',
                                    borderRadius: '8px',
                                    fontSize: '12px',
                                  }}
                                />
                                <Bar dataKey="importance" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                                  {trainResults.feature_importance.slice(0, 15).map((entry, index) => (
                                    <Cell
                                      key={index}
                                      fill={
                                        index === 0
                                          ? ML_CHART_COLORS.primary
                                          : index === 1
                                          ? ML_CHART_COLORS.secondary
                                          : index === 2
                                          ? ML_CHART_COLORS.accent
                                          : ML_CHART_COLORS.bars[index % ML_CHART_COLORS.bars.length]
                                      }
                                    />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {actualVsPredictedData.length > 0 && (
                    <motion.div variants={itemVariants}>
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center gap-2">
                            <Activity className="h-4 w-4 text-primary" />
                            <CardTitle className="text-base">Actual vs Predicted</CardTitle>
                          </div>
                          <CardDescription>Sample prediction quality from the latest training run</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                              <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke={ML_CHART_COLORS.grid} />
                                <XAxis
                                  type="number"
                                  dataKey="actualStandardized"
                                  name="Standardized Actual Value"
                                  domain={actualVsPredictedDomain}
                                  tick={{ fontSize: 11 }}
                                  tickFormatter={formatPredictionChartValue}
                                  label={{ value: 'Standardized Actual Value', position: 'insideBottom', offset: -5, style: { fontSize: 12 } }}
                                />
                                <YAxis
                                  type="number"
                                  dataKey="predictedStandardized"
                                  name="Standardized Predicted Value"
                                  domain={actualVsPredictedDomain}
                                  tick={{ fontSize: 11 }}
                                  tickFormatter={formatPredictionChartValue}
                                  label={{ value: 'Standardized Predicted Value', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 12 } }}
                                />
                                <RechartsTooltip
                                  formatter={(value: number, name: string) => [formatPredictionChartValue(value), name]}
                                  labelFormatter={(_, payload) => {
                                    const point = payload?.[0]?.payload;
                                    return point
                                      ? `Sample ${point.index} | Raw Actual: ${formatPredictionChartValue(point.actual)} | Raw Predicted: ${formatPredictionChartValue(point.predicted)}`
                                      : 'Prediction';
                                  }}
                                />
                                <ReferenceLine
                                  segment={[
                                    { x: actualVsPredictedDomain[0], y: actualVsPredictedDomain[0] },
                                    { x: actualVsPredictedDomain[1], y: actualVsPredictedDomain[1] },
                                  ]}
                                  stroke={ML_CHART_COLORS.secondary}
                                  strokeDasharray="4 4"
                                />
                                <Scatter data={actualVsPredictedData} fill={ML_CHART_COLORS.primary} isAnimationActive={false} />
                              </ScatterChart>
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {/* Model Analysis */}
                  {modelAnalysis && (() => {
                    const parsedAnalysis = parseModelAnalysis(modelAnalysis);
                    return (
                      <motion.div variants={itemVariants}>
                        <Card className="overflow-hidden border border-primary/20 bg-gradient-to-br from-primary/6 via-background to-secondary/70">
                          <CardHeader className="pb-4">
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex items-center gap-2">
                                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                                  <Bot className="h-4 w-4 text-primary" />
                                </div>
                                <div>
                                  <CardTitle className="text-base">Model Analysis</CardTitle>
                                  <CardDescription>AI-generated insights about your trained model</CardDescription>
                                </div>
                              </div>
                              <Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">
                                Insight Summary
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="rounded-xl border border-primary/15 bg-background/70 p-4">
                              <div className="flex items-center gap-2 text-primary">
                                <Lightbulb className="h-4 w-4" />
                                <p className="text-sm font-semibold">{parsedAnalysis.title}</p>
                              </div>
                              {parsedAnalysis.summary && (
                                <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{parsedAnalysis.summary}</p>
                              )}
                            </div>

                            {parsedAnalysis.bullets.length > 0 && (
                              <div className="grid gap-3 md:grid-cols-2">
                                {parsedAnalysis.bullets.map((bullet, index) => (
                                  <div key={index} className="flex items-start gap-2 rounded-xl border bg-background/60 p-3">
                                    <div className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary">
                                      <Sparkles className="h-3.5 w-3.5" />
                                    </div>
                                    <p className="text-sm text-muted-foreground leading-relaxed">{bullet}</p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </motion.div>
                    );
                  })()}

                  {/* Retrain Button */}
                  <motion.div variants={itemVariants}>
                    <div className="flex justify-center gap-3">
                      <Button variant="outline" onClick={handleTrain} disabled={isTraining}>
                        <Zap className="mr-2 h-4 w-4" />
                        Retrain Model
                      </Button>
                    </div>
                  </motion.div>
                </>
              )}
            </motion.div>
          )}

          {/* ─── STEP 5: Model Comparison ─────────────────────────────────── */}
          {currentStep === 5 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              {/* Compare Button */}
              <motion.div variants={itemVariants}>
                <Card className="border-dashed">
                  <CardContent className="p-6">
                    <div className="flex flex-col sm:flex-row items-center gap-4">
                      <div className="flex-1 text-center sm:text-left">
                        <p className="text-sm font-medium flex items-center justify-center sm:justify-start gap-2">
                          <GitCompareArrows className="h-4 w-4 text-primary" />
                          Compare Top Models
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Train and compare the top {recommendedCandidates.length} recommended models side by side
                        </p>
                      </div>
                      <Button
                        onClick={handleCompare}
                        disabled={isComparing}
                        className="bg-primary text-primary-foreground hover:bg-primary/90"
                      >
                        {isComparing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Comparing... ({Math.round(compareProgress)}%)
                          </>
                        ) : (
                          <>
                            <Swords className="mr-2 h-4 w-4" />
                            Compare Models
                          </>
                        )}
                      </Button>
                    </div>
                    {isComparing && (
                      <Progress value={compareProgress} className="mt-4 h-1.5" />
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Comparison Table */}
              {comparisonResults.length > 0 && (
                <motion.div variants={itemVariants}>
                  <Card>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <Trophy className="h-5 w-5 text-amber-500" />
                        <CardTitle className="text-lg">Comparison Results</CardTitle>
                      </div>
                      <CardDescription>
                        {bestModel && <span className="font-medium text-primary">{bestModel.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} is the current best performer in this comparison</span>}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left py-2 px-3 text-xs text-muted-foreground font-medium">Model</th>
                              {Object.keys(primaryMetrics).map(key => (
                                <th key={key} className="text-right py-2 px-3 text-xs text-muted-foreground font-medium">{key}</th>
                              ))}
                              <th className="text-right py-2 px-3 text-xs text-muted-foreground font-medium">CV Std</th>
                            </tr>
                          </thead>
                          <tbody>
                            {comparisonResults.map(result => {
                              const isBest = result.modelId === bestModel;
                              const primaryMetricKey = Object.keys(primaryMetrics)[0];
                              return (
                                <motion.tr
                                  key={result.modelId}
                                  className={cn(
                                    'border-b last:border-0 transition-colors',
                                    isBest && 'bg-primary/5',
                                  )}
                                  initial={{ opacity: 0, x: -10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                >
                                  <td className="py-3 px-3">
                                    <div className="flex items-center gap-2">
                                      {isBest && <Trophy className="h-3.5 w-3.5 text-amber-500" />}
                                      <span className={cn('font-medium', isBest && 'text-primary')}>
                                        {result.modelName}
                                      </span>
                                    </div>
                                  </td>
                                  {Object.keys(primaryMetrics).map(key => {
                                    const val = result.metrics[key];
                                    const bestVal = comparisonResults.reduce((max, r) => Math.max(max, r.metrics[key] || 0), 0);
                                    const isTop = val === bestVal && comparisonResults.filter(r => r.metrics[key] === bestVal).length === 1;
                                    return (
                                      <td key={key} className={cn('text-right py-3 px-3 font-mono text-xs', isTop && 'text-primary font-bold')}>
                                        {val !== undefined ? formatMetricValue(val) : '—'}
                                      </td>
                                    );
                                  })}
                                  <td className="text-right py-3 px-3 font-mono text-xs">
                                    {result.cvScores.length > 1
                                      ? Math.sqrt(result.cvScores.reduce((sum, v) => sum + (v - result.cvScores.reduce((a, b) => a + b, 0) / result.cvScores.length) ** 2, 0) / (result.cvScores.length - 1)).toFixed(4)
                                      : '—'}
                                  </td>
                                </motion.tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>

                      {/* Use This Model Button */}
                      {bestModel && (
                        <div className="mt-4 flex justify-center">
                          <Button
                            onClick={() => {
                              const winningResult = comparisonResults.find((result) => result.modelId === bestModel);
                              if (winningResult) {
                                activateComparedModel(winningResult);
                                toast({ title: 'Model Selected', description: `${winningResult.modelName} is now the active model.` });
                              }
                            }}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                          >
                            <CheckCircle2 className="mr-2 h-4 w-4" />
                            Use Best Model
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </motion.div>
          )}

          {/* ─── STEP 6: Summary & Proceed ────────────────────────────────── */}
          {currentStep === 6 && (
            <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
              {trainResults ? (
                <>
                  {/* Model Summary Card */}
                  <motion.div variants={itemVariants}>
                    <Card className="border border-primary/20 bg-gradient-to-br from-primary/6 to-secondary/70">
                      <CardHeader>
                        <div className="flex items-center gap-2">
                          <Award className="h-5 w-5 text-primary" />
                          <CardTitle className="text-lg">Model Summary</CardTitle>
                        </div>
                        <CardDescription>Your trained model is ready for predictions</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
                          <div className="rounded-lg bg-background/50 p-3 text-center">
                            <p className="text-xs text-muted-foreground">Model</p>
                            <p className="text-sm font-semibold capitalize mt-1">{(selectedModel || '').replace(/_/g, ' ')}</p>
                          </div>
                          <div className="rounded-lg bg-background/50 p-3 text-center">
                            <p className="text-xs text-muted-foreground">Problem Type</p>
                            <p className="text-sm font-semibold capitalize mt-1">{targetProblemType}</p>
                          </div>
                          <div className="rounded-lg bg-background/50 p-3 text-center">
                            <p className="text-xs text-muted-foreground">Target</p>
                            <p className="text-sm font-semibold mt-1">{selectedTarget}</p>
                          </div>
                          <div className="rounded-lg bg-background/50 p-3 text-center">
                            <p className="text-xs text-muted-foreground">Features</p>
                            <p className="text-sm font-semibold mt-1">{selectedFeatures.length}</p>
                          </div>
                        </div>

                        {optimizationSummary && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-2">Training Scale</p>
                            <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
                              <div className="rounded-lg bg-background/50 p-3 text-center">
                                <p className="text-xs text-muted-foreground">Rows available</p>
                                <p className="text-sm font-semibold mt-1">{(optimizationSummary.training_rows_available ?? data?.length ?? 0).toLocaleString()}</p>
                              </div>
                              <div className="rounded-lg bg-background/50 p-3 text-center">
                                <p className="text-xs text-muted-foreground">Rows used</p>
                                <p className="text-sm font-semibold mt-1">{(optimizationSummary.training_rows_used ?? data?.length ?? 0).toLocaleString()}</p>
                              </div>
                              <div className="rounded-lg bg-background/50 p-3 text-center">
                                <p className="text-xs text-muted-foreground">CV rows</p>
                                <p className="text-sm font-semibold mt-1">{(optimizationSummary.cv_rows_evaluated ?? 0).toLocaleString()}</p>
                              </div>
                              <div className="rounded-lg bg-background/50 p-3 text-center">
                                <p className="text-xs text-muted-foreground">Sampling</p>
                                <p className="text-sm font-semibold mt-1">{optimizationSummary.training_sampled ? 'Enabled' : 'Off'}</p>
                              </div>
                              <div className="rounded-lg bg-background/50 p-3 text-center">
                                <p className="text-xs text-muted-foreground">Mode</p>
                                <p className="text-sm font-semibold mt-1 capitalize">{optimizationSummary.training_mode ?? trainingMode}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Key Metrics */}
                        <div>
                          <p className="text-xs text-muted-foreground mb-2">Key Metrics</p>
                          <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
                            {Object.entries(primaryMetrics).map(([key, value]) => {
                              const Icon = metricIcons[key] || BarChart3;
                              return (
                                <div key={key} className="flex items-center gap-2 rounded-lg bg-background/50 p-3">
                                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                                    <Icon className="h-4 w-4 text-primary" />
                                  </div>
                                  <div>
                                    <p className={cn('text-sm font-bold', getMetricColor(key, value))}>
                                      {value !== null ? formatMetricValue(value) : '—'}
                                    </p>
                                    <p className="text-[10px] text-muted-foreground">{key}</p>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        {/* Mini Feature Importance */}
                        {hasMeaningfulFeatureImportance && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-2">Top 5 Features</p>
                            <div className="space-y-1.5">
                              {trainResults.feature_importance.slice(0, 5).map((f, i) => {
                                const maxImportance = trainResults.feature_importance[0]?.importance || 1;
                                return (
                                  <div key={f.name} className="flex items-center gap-2">
                                    <span className="text-xs text-muted-foreground w-4">{i + 1}</span>
                                    <span className="text-xs font-medium w-24 truncate">{f.name}</span>
                                    <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                                      <motion.div
                                        className="h-full rounded-full bg-gradient-to-r from-primary to-primary/70"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${(f.importance / maxImportance) * 100}%` }}
                                        transition={{ duration: 0.5, delay: i * 0.1 }}
                                      />
                                    </div>
                                    <span className="text-xs font-mono w-14 text-right">{f.importance.toFixed(3)}</span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Analysis */}
                        {modelAnalysis && (
                          <div className="rounded-lg bg-background/50 p-3">
                            <p className="text-xs text-muted-foreground mb-1">Model Analysis</p>
                            <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">{modelAnalysis}</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>

                  {/* Action Buttons */}
                  <motion.div variants={itemVariants}>
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                      <Button
                        onClick={goToPredict}
                        size="lg"
                        className="bg-primary text-primary-foreground hover:bg-primary/90"
                      >
                        Proceed to Predictions
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                      <Button
                        onClick={handleTrainAndGoPredict}
                        variant="outline"
                        className="border-secondary text-secondary-foreground"
                      >
                        <Zap className="mr-2 h-4 w-4" />
                        Retrain And Open Prediction
                      </Button>
                      <Button variant="outline" onClick={() => updateMlStep(2)}>
                        Train Another Model
                      </Button>
                    </div>
                  </motion.div>
                </>
              ) : (
                <motion.div variants={itemVariants}>
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <CircleDot className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
                      <p className="text-sm font-medium">No Model Trained Yet</p>
                      <p className="text-xs text-muted-foreground mt-1">Go back and train a model to see the summary.</p>
                      <Button variant="outline" className="mt-4" onClick={() => updateMlStep(3)}>
                        Go to Training
                        <ChevronRight className="ml-2 h-4 w-4" />
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>


      {/* Custom scrollbar styles */}
      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: hsl(var(--muted-foreground) / 0.3); border-radius: 999px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: hsl(var(--muted-foreground) / 0.5); }
      `}</style>
    </div>
  );
}








