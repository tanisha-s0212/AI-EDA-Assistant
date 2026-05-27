'use client';

import React from 'react';
import axios from 'axios';
import {
  AlertCircle,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Database,
  Download,
  Eye,
  FileText,
  Loader2,
  MessageSquare,
  Play,
  RefreshCw,
  Send,
  SkipForward,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Upload,
  Wand2,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getApiErrorMessage } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import type { AgenticStepStatus, Recommendation } from '@/lib/store';
import { cn } from '@/lib/utils';

const agenticApiClient = axios.create({
  baseURL: (process.env.NEXT_PUBLIC_AGENTIC_API_BASE || '/api/agentic').replace(/\/$/, ''),
  withCredentials: true,
});

const EXECUTABLE_STEPS = [
  'Data Understanding',
  'EDA',
  'Data Cleaning',
  'Time Series Forecast',
  'ML Forecast',
  'Loss Forecast',
  'Profit Forecast',
  'ML Assistant',
  'Prediction',
  'Report Generation',
];

const UI_STEPS = [
  { id: 'upload', label: 'Data Upload', handler: null, description: 'Confirm the active dataset and cached workspace context.', icon: Upload },
  { id: 'understanding', label: 'Data Understanding', handler: 'Data Understanding', description: 'Profile column roles, data types, row counts, and quality signals.', icon: Database },
  { id: 'eda', label: 'EDA', handler: 'EDA', description: 'Summarize distributions, correlations, and exploratory findings.', icon: BarChart3 },
  { id: 'cleaning', label: 'Data Cleaning', handler: 'Data Cleaning', description: 'Repair missing values, duplicates, date columns, and inferred data types.', icon: Sparkles },
  { id: 'ts', label: 'TS Forecast', handler: 'Time Series Forecast', description: 'Fit a statistical time-series forecast using detected date and target columns.', icon: TrendingUp },
  { id: 'mlf', label: 'ML Forecast', handler: 'ML Forecast', description: 'Train a feature-engineered forecast model and compare predictive quality.', icon: TrendingUp },
  { id: 'loss', label: 'Loss Forecast', handler: 'Loss Forecast', description: 'Estimate future loss pressure, risk scores, and top loss drivers.', icon: TrendingDown },
  { id: 'profit', label: 'Profit Forecast', handler: 'Profit Forecast', description: 'Generate optimistic, baseline, and pessimistic profit scenarios.', icon: TrendingUp },
  { id: 'ml', label: 'ML Assistant', handler: 'ML Assistant', description: 'Train a supervised model for the strongest available target and features.', icon: Wand2 },
  { id: 'prediction', label: 'Prediction', handler: 'Prediction', description: 'Run an approved prediction from the backend-trained model.', icon: Sparkles },
  { id: 'report', label: 'Report', handler: 'Report Generation', description: 'Compile the approved workflow outputs into a final run artifact.', icon: FileText },
] as const;

type UiStep = (typeof UI_STEPS)[number];
type HandlerStep = NonNullable<UiStep['handler']>;

type AgenticHealth = {
  agentic_enabled: boolean;
  db_connected: boolean;
  db_fallback_active: boolean;
};

type SuggestResponse = {
  session_id: string;
  recommendations: Recommendation[];
};

type StatusResponse = {
  steps: Record<string, AgenticStepStatus>;
  recommendations?: Recommendation[];
  next_suggested_step?: Recommendation | null;
  results?: Record<string, PersistedStepResult>;
  last_result?: PersistedStepResult | null;
};

type ExecuteStepResponse = {
  status: AgenticStepStatus;
  step_id?: string;
  decision_id?: string;
  step_name: string;
  output_summary?: string | null;
  error?: unknown;
  result?: unknown;
  next_recommendations?: Recommendation[];
  next_suggested_step?: Recommendation | null;
};

type PersistedStepResult = {
  step_id?: string;
  step_name?: string;
  status: AgenticStepStatus;
  executed_at?: string;
  result?: {
    output_summary?: string | null;
    error?: unknown;
    result?: unknown;
    next_recommendations?: Recommendation[];
    [key: string]: unknown;
  };
};

type ChatResponse = {
  answer?: string;
  error?: string;
  provider?: string;
};

type AgenticWorkspaceProps = {
  datasetId: string | null;
  fileName: string | null;
};

type StepArtifact = {
  step: string;
  summary: string;
  completedAt: number;
  status: AgenticStepStatus;
  result?: unknown;
};

type StructuredError = {
  step: string;
  reason: string;
  detail?: unknown;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

function AgenticWorkspaceStyles() {
  return (
    <style jsx global>{`
      @keyframes ida-memory-scroll {
        from { transform: translateX(0); }
        to { transform: translateX(-50%); }
      }
      .ida-memory-marquee {
        animation: ida-memory-scroll 32s linear infinite;
      }
      .ida-agent-terminal {
        scrollbar-width: thin;
        scrollbar-color: #22c55e #0a0a0a;
      }
      .ida-agent-terminal::-webkit-scrollbar { width: 8px; }
      .ida-agent-terminal::-webkit-scrollbar-track { background: #0a0a0a; }
      .ida-agent-terminal::-webkit-scrollbar-thumb { background: #16a34a; border-radius: 999px; }
    `}</style>
  );
}

function normalizeStepName(step: string) {
  return step === 'Report' ? 'Report Generation' : step;
}

function artifactFromPersisted(stepName: string, persisted: PersistedStepResult): StepArtifact {
  const result = persisted.result ?? {};
  return {
    step: stepName,
    status: persisted.status,
    summary: result.output_summary ?? (result.error ? String(result.error) : persisted.status),
    completedAt: persisted.executed_at ? new Date(persisted.executed_at).getTime() : Date.now(),
    result,
  };
}

function getUiStatus(step: UiStep, statuses: Record<string, AgenticStepStatus>, datasetId: string | null): AgenticStepStatus {
  if (step.id === 'upload') return datasetId ? 'completed' : 'pending';
  return statuses[step.handler ?? ''] ?? 'pending';
}

function completedUiCount(statuses: Record<string, AgenticStepStatus>, datasetId: string | null) {
  return UI_STEPS.filter((step) => {
    const status = getUiStatus(step, statuses, datasetId);
    return status === 'completed' || status === 'skipped';
  }).length;
}

function statusDotClass(status: AgenticStepStatus, active: boolean) {
  if (status === 'completed' || status === 'skipped') return 'bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.45)]';
  if (active || status === 'running') return 'bg-blue-400 shadow-[0_0_18px_rgba(96,165,250,0.55)]';
  if (status === 'failed') return 'bg-red-400 shadow-[0_0_18px_rgba(248,113,113,0.45)]';
  return 'bg-slate-500';
}

function formatDuration(ms?: number) {
  if (!ms) return null;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function getPreferredDateColumn() {
  const { columns } = useAppStore.getState();
  return columns.find((column) => column.role === 'datetime')?.name
    ?? columns.find((column) => /date|month|time|period/i.test(column.name))?.name
    ?? '';
}

function getPreferredTargetColumn() {
  const { columns } = useAppStore.getState();
  return columns.find((column) => /sales|revenue|amount|profit|loss|target|value|price|cost/i.test(column.name) && column.role === 'numeric')?.name
    ?? columns.find((column) => column.role === 'numeric')?.name
    ?? columns.find((column) => !/id$/i.test(column.name) && column.role !== 'identifier')?.name
    ?? '';
}

function buildMemoryFacts(fileName: string | null) {
  const state = useAppStore.getState();
  const facts = [
    fileName ? `Dataset: ${fileName}` : null,
    state.totalRows || state.rawData?.length ? `${(state.totalRows || state.rawData?.length || 0).toLocaleString()} rows` : null,
    state.columns.length ? `${state.columns.length} columns` : null,
    state.timeSeriesForecastResult?.frequency ? `${state.timeSeriesForecastResult.frequency} frequency` : null,
    state.timeSeriesForecastResult?.dataset_profile?.volatility ? `Volatility ${formatNumber(state.timeSeriesForecastResult.dataset_profile.volatility)}` : null,
    state.mlForecastResult?.training_summary?.model_name ? `${state.mlForecastResult.training_summary.model_name} best` : null,
    state.mlForecastResult?.metrics?.mae ? `ML MAE ${formatNumber(state.mlForecastResult.metrics.mae)}` : null,
    state.timeSeriesForecastResult?.metrics?.mae ? `TS MAE ${formatNumber(state.timeSeriesForecastResult.metrics.mae)}` : null,
    state.lossSummary?.top_loss_driver ? `Top loss driver ${state.lossSummary.top_loss_driver}` : null,
    state.lossForecast?.some((row) => Number(row.total_loss) === 0) ? 'Zero-tail detected' : null,
    state.predictionResult !== null ? `Prediction ${String(state.predictionResult)}` : null,
  ].filter(Boolean) as string[];
  return facts.length ? facts : ['Awaiting dataset profile', 'Agent memory will update after each step'];
}

function buildInsight(activeStep: UiStep | null) {
  const state = useAppStore.getState();
  if (state.mlForecastResult?.analysis) return state.mlForecastResult.analysis;
  if (state.timeSeriesForecastResult?.analysis) return state.timeSeriesForecastResult.analysis;
  if (state.cleaningDone) return 'The cleaned dataset is now the safest base layer for forecasting and model training. Continue with forecasting to compare statistical and ML approaches.';
  if (state.columns.length) return `The agent has profiled ${state.columns.length} columns. Review missing values and type inference before committing to forecasts.`;
  return activeStep ? `Start with ${activeStep.label} so the agent can ground every later decision in the active dataset.` : 'Upload or select a dataset to activate the agentic run.';
}

function buildConfidence(step: UiStep | null, completedCount: number) {
  const state = useAppStore.getState();
  const qualityScores = [
    state.timeSeriesForecastResult?.data_quality?.score,
    state.mlForecastResult?.data_quality?.score,
  ].filter((score): score is number => typeof score === 'number' && Number.isFinite(score));
  const base = qualityScores.length ? qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length : 0.78;
  const historyBoost = Math.min(0.16, completedCount * 0.018);
  const stepPenalty = step?.handler && state.agenticStepStatuses[step.handler] === 'failed' ? 0.22 : 0;
  return Math.max(42, Math.min(99, Math.round((base + historyBoost - stepPenalty) * 100)));
}

function getStepContext(step: UiStep | null, artifacts: StepArtifact[]) {
  const state = useAppStore.getState();
  const latest = artifacts.at(-1);
  const findings = [
    latest ? `${latest.step} finished: ${latest.summary}` : null,
    state.columns.length ? `${state.columns.length} columns available for downstream reasoning.` : null,
    state.cleaningDone ? `${state.cleaningLogs.length} cleaning actions recorded.` : null,
    state.mlForecastResult ? `ML forecast selected ${state.mlForecastResult.training_summary.model_name}.` : null,
    state.timeSeriesForecastResult ? `TS forecast MAE is ${formatNumber(state.timeSeriesForecastResult.metrics.mae)}.` : null,
  ].filter(Boolean) as string[];
  const issues = [
    !state.rawData?.length ? 'No dataset rows are loaded in the workspace preview.' : null,
    step?.handler?.includes('Forecast') && !getPreferredDateColumn() ? 'A date-like column is required for forecasting.' : null,
    step?.handler?.includes('Forecast') && !getPreferredTargetColumn() ? 'A numeric target column is required for forecasting.' : null,
  ].filter(Boolean) as string[];
  return { findings: findings.slice(0, 4), issues: issues.slice(0, 3) };
}

function StepResultCard({ artifact }: { artifact: StepArtifact | null }) {
  if (!artifact) return null;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-950 dark:text-slate-100">Backend Step Result</h3>
        <Badge variant="outline" className="dark:border-slate-700 dark:text-slate-300">{artifact.step}</Badge>
      </div>
      <p className="text-sm text-slate-700 dark:text-slate-300">{artifact.summary}</p>
      <pre className="mt-3 max-h-64 overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-slate-300">
        {JSON.stringify({ status: artifact.status, result: artifact.result }, null, 2)}
      </pre>
    </div>
  );
}

function DataPreviewTable() {
  const state = useAppStore();
  const data = (state.cleanedData ?? state.rawData ?? []).slice(0, 5);
  const columns = state.columns.slice(0, 8).map((column) => column.name);
  if (!data.length || !columns.length) return null;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <h3 className="mb-3 text-sm font-semibold text-slate-950 dark:text-slate-100">Live Data Preview</h3>
      <div className="max-h-56 overflow-auto rounded-lg border border-slate-200 dark:border-slate-700">
        <table className="w-full min-w-[720px] border-collapse font-mono text-xs">
          <thead className="sticky top-0 bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            <tr>
              {columns.map((column) => <th key={column} className="border-b border-slate-200 px-3 py-2 text-left dark:border-slate-700">{column}</th>)}
            </tr>
          </thead>
          <tbody className="text-slate-700 dark:text-slate-300">
            {data.map((row, index) => (
              <tr key={index} className="odd:bg-slate-50 dark:odd:bg-slate-950/60">
                {columns.map((column) => <td key={column} className="max-w-40 truncate border-b border-slate-100 px-3 py-2 dark:border-slate-800">{String(row[column] ?? '')}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ForecastResultsCard() {
  const { timeSeriesForecastResult, mlForecastResult } = useAppStore();
  const result = mlForecastResult ?? timeSeriesForecastResult;
  if (!result?.future_forecast?.length) return null;
  const points = [
    ...result.history.slice(-8).map((point) => ({ period: point.period, value: point.actual, kind: 'actual' as const })),
    ...result.future_forecast.map((point) => ({ period: point.period, value: point.predicted, kind: 'forecast' as const })),
  ];
  const values = points.map((point) => point.value).filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const coords = points.map((point, index) => {
    const x = points.length === 1 ? 8 : 8 + (index / (points.length - 1)) * 284;
    const y = 82 - ((point.value - min) / range) * 64;
    return `${x},${y}`;
  }).join(' ');
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-950 dark:text-slate-100">Forecast Results</h3>
        <Badge className="bg-blue-600 text-white">{mlForecastResult ? 'ML Forecast' : 'TS Forecast'}</Badge>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {result.future_forecast.slice(0, 3).map((point) => (
          <div key={point.period} className="rounded-lg border border-blue-100 bg-blue-50 p-3 dark:border-blue-500/30 dark:bg-blue-950/30">
            <p className="text-xs text-blue-700 dark:text-blue-300">{point.period}</p>
            <p className="mt-1 text-lg font-bold text-blue-950 dark:text-blue-100">{formatNumber(point.predicted)}</p>
          </div>
        ))}
      </div>
      <svg viewBox="0 0 300 92" className="mt-4 h-28 w-full overflow-visible">
        <polyline points={coords} fill="none" stroke="#2563eb" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((point, index) => {
          const [x, y] = coords.split(' ')[index].split(',').map(Number);
          return <circle key={`${point.period}-${index}`} cx={x} cy={y} r="3" fill={point.kind === 'forecast' ? '#22c55e' : '#2563eb'} />;
        })}
      </svg>
      <div className="flex justify-between gap-3 text-xs text-slate-500 dark:text-slate-400">
        <span>{points[0]?.period}</span>
        <span>{points.at(-1)?.period}</span>
      </div>
    </div>
  );
}

function ShapPanel() {
  const { mlForecastResult } = useAppStore();
  const features = mlForecastResult?.shap_feature_importance?.slice(0, 5) ?? [];
  if (!features.length) return null;
  const max = Math.max(...features.map((feature) => Math.abs(feature.importance)), 1);
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <h3 className="mb-4 text-sm font-semibold text-slate-950 dark:text-slate-100">SHAP Feature Importance</h3>
      <div className="space-y-3">
        {features.map((feature) => (
          <div key={feature.name}>
            <div className="mb-1 flex justify-between gap-3 text-xs">
              <span className="truncate text-slate-600 dark:text-slate-300">{feature.name}</span>
              <span className="font-mono text-slate-500 dark:text-slate-400">{formatNumber(feature.importance, 4)}</span>
            </div>
            <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800">
              <div className="h-2 rounded-full bg-blue-600" style={{ width: `${Math.max(4, (Math.abs(feature.importance) / max) * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ExecutionLog({ logs, running }: { logs: string[]; running: boolean }) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  React.useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [logs]);
  if (!running && logs.length === 0) return null;
  return (
    <div ref={ref} className="ida-agent-terminal max-h-[120px] overflow-auto rounded-xl border border-green-500/20 bg-[#0a0a0a] p-3 font-mono text-xs leading-5 text-green-400 shadow-inner">
      {(logs.length ? logs : ['waiting for execution...']).map((line, index) => <div key={`${line}-${index}`}>{line}</div>)}
    </div>
  );
}

export default function AgenticWorkspace({ datasetId, fileName }: AgenticWorkspaceProps) {
  const {
    agenticSessionId,
    agenticStepStatuses,
    agenticRecommendations,
    setAgenticSessionId,
    setAgenticStepStatus,
    setAgenticRecommendations,
    setAgenticLastSyncedAt,
  } = useAppStore();
  const store = useAppStore();
  const [health, setHealth] = React.useState<AgenticHealth | null>(null);
  const [isSuggesting, setIsSuggesting] = React.useState(false);
  const [runningStep, setRunningStep] = React.useState<string | null>(null);
  const [runningAll, setRunningAll] = React.useState(false);
  const [banner, setBanner] = React.useState<string | null>(null);
  const [executionLogs, setExecutionLogs] = React.useState<string[]>([]);
  const [artifacts, setArtifacts] = React.useState<StepArtifact[]>([]);
  const [stepDurations, setStepDurations] = React.useState<Record<string, number>>({});
  const [expandedDetails, setExpandedDetails] = React.useState(false);
  const [structuredError, setStructuredError] = React.useState<StructuredError | null>(null);
  const [chatOpen, setChatOpen] = React.useState(false);
  const [chatInput, setChatInput] = React.useState('');
  const [chatMessages, setChatMessages] = React.useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = React.useState(false);
  const lastSuggestedDatasetRef = React.useRef<string | null>(null);

  const activeRecommendation = agenticRecommendations[0] ?? null;
  const completedCount = completedUiCount(agenticStepStatuses, datasetId);
  const pendingCount = UI_STEPS.length - completedCount;
  const progress = Math.round((completedCount / UI_STEPS.length) * 100);
  const activeHandlerStep = normalizeStepName(activeRecommendation?.step ?? runningStep ?? '') || null;
  const activeUiStep = UI_STEPS.find((step) => step.handler === activeHandlerStep) ?? UI_STEPS.find((step) => getUiStatus(step, agenticStepStatuses, datasetId) === 'pending') ?? UI_STEPS[0];
  const latestArtifact = artifacts.at(-1) ?? null;
  const confidence = buildConfidence(activeUiStep, completedCount);
  const context = getStepContext(activeUiStep, artifacts);
  const memoryFacts = buildMemoryFacts(fileName);
  const insight = buildInsight(activeUiStep);
  const bestModel = store.mlForecastResult?.training_summary?.model_name ?? store.selectedModel ?? store.timeSeriesForecastResult?.training_summary?.model_name ?? 'Pending';
  const forecastMae = store.mlForecastResult?.metrics?.mae ?? store.timeSeriesForecastResult?.metrics?.mae;
  const previewReady = ['Data Understanding', 'EDA', 'Data Cleaning'].some((step) => agenticStepStatuses[step] === 'completed');

  const appendLog = React.useCallback((line: string) => {
    const stamp = new Date().toLocaleTimeString([], { hour12: false });
    setExecutionLogs((current) => [...current.slice(-80), `[${stamp}] ${line}`]);
  }, []);

  const refreshHealth = React.useCallback(async () => {
    try {
      const response = await agenticApiClient.get<AgenticHealth>('/health');
      setHealth(response.data);
      setBanner(response.data.db_connected ? null : 'Agentic database unavailable - backend execution is paused');
    } catch {
      setHealth(null);
      setBanner('Agentic layer unavailable - manual mode active');
    }
  }, []);

  React.useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const refreshStatus = React.useCallback(async () => {
    if (!agenticSessionId) return;
    try {
      const response = await agenticApiClient.get<StatusResponse>(`/session/${agenticSessionId}/status`);
      Object.entries(response.data.steps ?? {}).forEach(([step, status]) => {
        setAgenticStepStatus(step, status);
      });
      if (response.data.recommendations) {
        setAgenticRecommendations(response.data.recommendations.slice(0, 1));
      }
      if (response.data.results) {
        const nextArtifacts = Object.entries(response.data.results).map(([step, result]) => artifactFromPersisted(step, result));
        setArtifacts(nextArtifacts.sort((left, right) => left.completedAt - right.completedAt));
      }
      if (runningStep) {
        const backendStatus = response.data.steps?.[runningStep];
        if (backendStatus === 'completed' || backendStatus === 'failed' || backendStatus === 'skipped') {
          setRunningStep(null);
        }
      }
      setAgenticLastSyncedAt(Date.now());
    } catch {
      setBanner('Agentic layer unavailable - manual mode active');
    }
  }, [agenticSessionId, runningStep, setAgenticLastSyncedAt, setAgenticRecommendations, setAgenticStepStatus]);

  React.useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  React.useEffect(() => {
    if (!agenticSessionId || (!runningStep && !runningAll)) return;
    const interval = window.setInterval(() => void refreshStatus(), 3000);
    return () => window.clearInterval(interval);
  }, [agenticSessionId, refreshStatus, runningAll, runningStep]);

  const suggestNextSteps = React.useCallback(async () => {
    if (!datasetId) return null;
    setIsSuggesting(true);
    setBanner(null);
    try {
      const response = await agenticApiClient.post<SuggestResponse>('/suggest-next-steps', { dataset_path: datasetId });
      setAgenticSessionId(response.data.session_id);
      setAgenticRecommendations(response.data.recommendations.slice(0, 1));
      EXECUTABLE_STEPS.forEach((step) => setAgenticStepStatus(step, 'pending'));
      setAgenticLastSyncedAt(Date.now());
      appendLog(`suggested next action: ${response.data.recommendations[0]?.step ?? 'none'}`);
      return response.data.session_id;
    } catch (error) {
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable - manual mode active'));
      return null;
    } finally {
      setIsSuggesting(false);
    }
  }, [appendLog, datasetId, setAgenticLastSyncedAt, setAgenticRecommendations, setAgenticSessionId, setAgenticStepStatus]);

  React.useEffect(() => {
    if (!datasetId || health?.agentic_enabled === false || lastSuggestedDatasetRef.current === datasetId) return;
    lastSuggestedDatasetRef.current = datasetId;
    void suggestNextSteps();
  }, [datasetId, health?.agentic_enabled, suggestNextSteps]);

  const downloadReport = async (sessionId: string, partial = false) => {
    const response = await agenticApiClient.get(`/session/${sessionId}/report`, { responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    link.download = partial ? `agentic_partial_results_${sessionId}.html` : `agentic_run_${sessionId}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const executeStep = async (stepName: HandlerStep, sessionId: string, decision: 'accepted' | 'skipped' = 'accepted') => {
    const startedAt = performance.now();
    setStructuredError(null);
    setRunningStep(stepName);
    setAgenticStepStatus(stepName, 'running');
    appendLog(`${decision === 'skipped' ? 'skipping' : 'starting'} ${stepName}`);
    try {
      const definition = UI_STEPS.find((step) => step.handler === stepName);
      const response = await agenticApiClient.post<ExecuteStepResponse>('/execute-step', {
        session_id: sessionId,
        step_name: stepName,
        decision,
        approved_by: 'current_user',
        reasoning: decision === 'skipped' ? 'Skipped from agentic workspace.' : '',
        step_definition: definition ? {
          id: definition.id,
          label: definition.label,
          handler: definition.handler,
          description: definition.description,
        } : { handler: stepName },
      });
      if (response.data.status === 'failed') {
        throw response.data.error ?? new Error(`${stepName} failed.`);
      }
      const finalStatus = response.data.status ?? (decision === 'skipped' ? 'skipped' : 'completed');
      const outputSummary = response.data.output_summary ?? `${stepName} ${finalStatus}.`;
      setAgenticStepStatus(stepName, finalStatus);
      const duration = performance.now() - startedAt;
      setStepDurations((current) => ({ ...current, [stepName]: duration }));
      const artifact = { step: stepName, summary: outputSummary, completedAt: Date.now(), status: finalStatus, result: response.data };
      setArtifacts((current) => [...current, artifact]);
      setAgenticRecommendations(response.data.next_recommendations ?? []);
      appendLog(`backend completed ${stepName}: ${outputSummary}`);
      return outputSummary;
    } catch (error) {
      const reason = getApiErrorMessage(error, `${stepName} failed.`);
      setAgenticStepStatus(stepName, 'failed');
      setStructuredError({ step: stepName, reason, detail: error });
      setBanner(reason);
      appendLog(`error: ${reason}`);
      throw error;
    } finally {
      setRunningStep(null);
    }
  };

  const acceptRecommendation = async () => {
    if (!activeRecommendation) return;
    const sessionId = agenticSessionId ?? await suggestNextSteps();
    if (!sessionId) return;
    await executeStep(normalizeStepName(activeRecommendation.step) as HandlerStep, sessionId);
  };

  const skipStep = async (stepName?: string) => {
    const target = normalizeStepName(stepName ?? activeRecommendation?.step ?? '');
    if (!target || !agenticSessionId) return;
    try {
      await executeStep(target as HandlerStep, agenticSessionId, 'skipped');
    } catch (error) {
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable - manual mode active'));
    }
  };

  const runAll = async () => {
    const sessionId = agenticSessionId ?? await suggestNextSteps();
    if (!sessionId) return;
    setRunningAll(true);
    try {
      for (const step of EXECUTABLE_STEPS) {
        const status = useAppStore.getState().agenticStepStatuses[step];
        if (status === 'completed' || status === 'skipped') continue;
        await executeStep(step as HandlerStep, sessionId);
      }
    } finally {
      setRunningAll(false);
    }
  };

  const askAgent = async (message: string) => {
    setChatOpen(true);
    setChatLoading(true);
    const userMessage: ChatMessage = { id: `${Date.now()}-user`, role: 'user', content: message };
    setChatMessages((current) => [...current, userMessage]);
    try {
      const response = await agenticApiClient.post<ChatResponse>('/core/chat', { message, mode: 'ask', provider: 'auto' });
      setChatMessages((current) => [...current, { id: `${Date.now()}-assistant`, role: 'assistant', content: response.data.answer ?? response.data.error ?? 'No answer returned.' }]);
    } catch (error) {
      setChatMessages((current) => [...current, { id: `${Date.now()}-assistant`, role: 'assistant', content: getApiErrorMessage(error, 'The agent could not answer right now.') }]);
    } finally {
      setChatLoading(false);
    }
  };

  const sendChat = async () => {
    const message = chatInput.trim();
    if (!message) return;
    setChatInput('');
    await askAgent(message);
  };

  const askAgentToFix = async () => {
    if (!structuredError) return;
    await askAgent(`Codex repair request: The IDA Agentic Core step "${structuredError.step}" failed with "${structuredError.reason}". Provide a targeted fix plan using the current dataset ${fileName ?? datasetId ?? 'unknown dataset'} and the completed step context.`);
  };

  const applyInsightFix = async () => {
    await askAgent(`Turn this dataset insight into an executable next action for the IDA workflow: ${insight}`);
  };

  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-xl dark:border-slate-800 dark:bg-slate-950">
      <AgenticWorkspaceStyles />
      <header className="flex flex-col gap-3 border-b border-slate-800 bg-[#0f172a] p-4 text-white lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-blue-600 text-sm font-black text-white ring-2 ring-blue-300/30">IDA</div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">{fileName ?? 'No dataset selected'}</p>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <Badge className={cn('rounded-full border px-2.5 py-0.5 text-xs', runningStep ? 'border-blue-400/40 bg-blue-500/15 text-blue-200' : health?.agentic_enabled === false ? 'border-red-400/40 bg-red-500/15 text-red-200' : 'border-emerald-400/40 bg-emerald-500/15 text-emerald-200')}>
                {runningStep ? `Running ${runningStep}` : health?.agentic_enabled === false ? 'Agent offline' : 'Live agent ready'}
              </Badge>
              <span className="text-xs text-slate-400">{completedCount}/{UI_STEPS.length} completed</span>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" onClick={() => void runAll()} disabled={!datasetId || Boolean(runningStep) || runningAll} className="bg-blue-600 text-white hover:bg-blue-500">
            {runningAll ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run All
          </Button>
          <Button size="sm" variant="outline" onClick={() => void suggestNextSteps()} disabled={!datasetId || isSuggesting} className="border-slate-600 bg-slate-900 text-slate-100 hover:bg-slate-800 hover:text-white">
            {isSuggesting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ChevronRight className="mr-2 h-4 w-4" />}
            Suggest Next
          </Button>
          <Button size="sm" variant="outline" onClick={() => agenticSessionId && void downloadReport(agenticSessionId, true)} disabled={!agenticSessionId || completedCount <= 1} className="border-slate-600 bg-slate-900 text-slate-100 hover:bg-slate-800 hover:text-white">
            <Download className="mr-2 h-4 w-4" />
            Download Current Results
          </Button>
        </div>
      </header>

      <div className="grid min-h-[720px] lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="border-r border-slate-800 bg-slate-950 p-4 text-slate-200">
          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-semibold uppercase tracking-wide text-slate-400">Pipeline</span>
              <span className="font-mono text-blue-300">{completedCount}/{UI_STEPS.length}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
          </div>
          <div className="space-y-1.5">
            {UI_STEPS.map((step) => {
              const status = getUiStatus(step, agenticStepStatuses, datasetId);
              const active = activeUiStep?.id === step.id || runningStep === step.handler;
              const duration = step.handler ? formatDuration(stepDurations[step.handler]) : datasetId ? '0.1s' : null;
              const Icon = step.icon;
              return (
                <div key={step.id} className={cn('rounded-lg border border-transparent px-3 py-2 transition', active && 'border-l-4 border-l-blue-500 bg-blue-500/10')}>
                  <div className="flex items-center gap-2">
                    <span className={cn('h-2.5 w-2.5 shrink-0 rounded-full', statusDotClass(status, active))} />
                    <Icon className="h-4 w-4 shrink-0 text-slate-400" />
                    <span className={cn('min-w-0 flex-1 truncate text-sm font-medium', active ? 'text-white' : 'text-slate-300')}>{step.label}</span>
                    {duration && <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">{duration}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        <main className="flex min-w-0 flex-col bg-slate-100 dark:bg-slate-950">
          <div className="border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
            <div className="overflow-hidden">
              <div className="ida-memory-marquee flex w-max gap-2">
                {[...memoryFacts, ...memoryFacts].map((fact, index) => (
                  <span key={`${fact}-${index}`} className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700 dark:border-blue-500/30 dark:bg-blue-950/40 dark:text-blue-200">{fact}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="min-h-0 flex-1 space-y-4 overflow-auto p-4 pb-24">
            {banner && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-200">
                <AlertCircle className="mr-2 inline h-4 w-4" />
                {banner}
              </div>
            )}

            <div className="rounded-xl border border-l-4 border-slate-200 border-l-blue-600 bg-white p-4 shadow-sm dark:border-slate-700 dark:border-l-blue-500 dark:bg-slate-900">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-lg font-semibold text-slate-950 dark:text-slate-100">{activeUiStep?.label ?? 'Agentic Step'}</h2>
                    <Badge className="bg-blue-600 text-white">{confidence}% confidence</Badge>
                  </div>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{activeUiStep?.description}</p>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setExpandedDetails((value) => !value)} className="shrink-0 dark:text-slate-200 dark:hover:bg-slate-800">
                  <Eye className="h-4 w-4" />
                </Button>
              </div>
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <div className="space-y-2">
                  {context.findings.map((finding) => (
                    <div key={finding} className="flex gap-2 text-sm text-slate-700 dark:text-slate-300">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                      <span>{finding}</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-2">
                  {context.issues.length ? context.issues.map((issue) => (
                    <div key={issue} className="flex gap-2 text-sm text-amber-700 dark:text-amber-300">
                      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                      <span>{issue}</span>
                    </div>
                  )) : (
                    <div className="flex gap-2 text-sm text-slate-500 dark:text-slate-400">
                      <Check className="mt-0.5 h-4 w-4 shrink-0 text-blue-500" />
                      <span>No blocking issues detected for this step.</span>
                    </div>
                  )}
                </div>
              </div>
              {expandedDetails && (
                <pre className="mt-4 overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-slate-300">
                  {JSON.stringify({ activeStep: activeUiStep, recommendation: activeRecommendation, context, confidence }, null, 2)}
                </pre>
              )}
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={() => void acceptRecommendation()} disabled={!activeRecommendation || Boolean(runningStep)} className="bg-blue-600 text-white hover:bg-blue-500">
                  {runningStep ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Check className="mr-2 h-4 w-4" />}
                  Accept & Continue
                </Button>
                <Button variant="outline" onClick={() => void skipStep()} disabled={!activeRecommendation || Boolean(runningStep)} className="dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800">
                  <SkipForward className="mr-2 h-4 w-4" />
                  Skip
                </Button>
              </div>
            </div>

            <ExecutionLog logs={executionLogs} running={Boolean(runningStep)} />

            {structuredError && (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 dark:border-red-500/30 dark:bg-red-950/30">
                <h3 className="font-semibold text-red-800 dark:text-red-200">{structuredError.step} failed</h3>
                <p className="mt-1 text-sm text-red-700 dark:text-red-300">{structuredError.reason}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" onClick={() => agenticSessionId && void executeStep(structuredError.step as HandlerStep, agenticSessionId)}><RefreshCw className="mr-2 h-4 w-4" />Retry</Button>
                  <Button size="sm" variant="outline" onClick={() => void skipStep(structuredError.step)} className="dark:border-slate-700 dark:text-slate-200">Skip</Button>
                  <Button size="sm" variant="outline" onClick={() => void askAgentToFix()} className="dark:border-slate-700 dark:text-slate-200"><Wand2 className="mr-2 h-4 w-4" />Ask agent to fix</Button>
                </div>
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-4">
              <MetricCard label="Steps Completed" value={String(completedCount)} />
              <MetricCard label="Steps Pending" value={String(pendingCount)} />
              <MetricCard label="Best Model Selected" value={bestModel} />
              <MetricCard label="Forecast MAE" value={formatNumber(forecastMae)} />
            </div>

            <StepResultCard artifact={latestArtifact} />

            {previewReady && <DataPreviewTable />}

            <div className="grid gap-4 xl:grid-cols-2">
              <ForecastResultsCard />
              <ShapPanel />
            </div>

            <div className="rounded-xl border border-l-4 border-amber-200 border-l-amber-500 bg-white p-4 shadow-sm dark:border-slate-700 dark:border-l-amber-400 dark:bg-slate-900">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <h3 className="font-semibold text-slate-950 dark:text-slate-100">Agent Insight</h3>
                  <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">{insight}</p>
                </div>
                <Button variant="outline" onClick={() => void applyInsightFix()} className="shrink-0 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800">
                  <Wand2 className="mr-2 h-4 w-4" />
                  Apply Fix
                </Button>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900">
            <button className="mb-2 flex w-full items-center justify-between text-sm font-semibold text-slate-700 dark:text-slate-200" onClick={() => setChatOpen((value) => !value)}>
              <span className="inline-flex items-center gap-2"><MessageSquare className="h-4 w-4" /> Ask about completed steps</span>
              <ChevronDown className={cn('h-4 w-4 transition', chatOpen && 'rotate-180')} />
            </button>
            {chatOpen && (
              <div className="space-y-3">
                <div className="max-h-48 space-y-2 overflow-auto rounded-lg bg-slate-50 p-3 dark:bg-slate-950">
                  {chatMessages.length ? chatMessages.map((message) => (
                    <div key={message.id} className={cn('rounded-lg px-3 py-2 text-sm', message.role === 'user' ? 'ml-auto max-w-[85%] bg-blue-600 text-white' : 'mr-auto max-w-[90%] bg-white text-slate-700 dark:bg-slate-800 dark:text-slate-200')}>
                      {message.content}
                    </div>
                  )) : <p className="text-sm text-slate-500 dark:text-slate-400">Ask why a model was selected, what changed during cleaning, or what the agent recommends next.</p>}
                  {chatLoading && <div className="text-sm text-slate-500 dark:text-slate-400">Agent is thinking...</div>}
                </div>
                <div className="flex gap-2">
                  <input
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') void sendChat();
                    }}
                    placeholder="Why did you choose XGBoost?"
                    className="h-10 min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                  />
                  <Button onClick={() => void sendChat()} disabled={chatLoading || !chatInput.trim()} className="bg-blue-600 text-white hover:bg-blue-500">
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
      <p className="mt-2 truncate text-xl font-bold text-slate-950 dark:text-slate-100">{value}</p>
    </div>
  );
}
