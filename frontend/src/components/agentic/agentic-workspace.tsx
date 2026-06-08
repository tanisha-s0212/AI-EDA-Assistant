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
  FileJson,
  FileSpreadsheet,
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
  XCircle,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { getApiErrorMessage } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import type { AgenticStepStatus, Recommendation, TabId } from '@/lib/store';
import { cn } from '@/lib/utils';

function renderStructuredOrMarkdown(text: string): string {
  const trimmed = text.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === 'object' && (parsed.next_action || parsed.execution_plan || parsed.validation)) {
      const parts: string[] = [];
      if (parsed.next_action) parts.push(`<div class="agent-json-field"><span class="agent-field-label">Next Action</span><span class="agent-field-value">${escHtml(String(parsed.next_action))}</span></div>`);
      if (parsed.execution_plan && Array.isArray(parsed.execution_plan)) {
        parts.push('<div class="agent-json-field"><span class="agent-field-label">Execution Plan</span><ol class="agent-plan-list">');
        for (const item of parsed.execution_plan) {
          parts.push(`<li><strong>Step ${item.step ?? ''}:</strong> ${escHtml(String(item.task ?? ''))}${item.api_call ? ` <code>${escHtml(String(item.api_call))}</code>` : ''}</li>`);
        }
        parts.push('</ol></div>');
      }
      if (parsed.validation) parts.push(`<div class="agent-json-field"><span class="agent-field-label">Validation</span><span class="agent-field-value">${escHtml(String(parsed.validation))}</span></div>`);
      if (parsed.reasoning) parts.push(`<div class="agent-json-field"><span class="agent-field-label">Reasoning</span><span class="agent-field-value">${escHtml(String(parsed.reasoning))}</span></div>`);
      return parts.join('');
    }
  } catch {
    /* not JSON — render as markdown */
  }
  return renderMarkdown(trimmed);
}

function escHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(text: string): string {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/^### (.*$)/gm, '<h3 class="agent-section-heading">$1</h3>');
  html = html.replace(/^## (.*$)/gm, '<h2 class="agent-section-heading">$1</h2>');
  html = html.replace(/^# (.*$)/gm, '<h1 class="agent-section-heading">$1</h1>');
  html = html.replace(/^- (.*)/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
  html = html.replace(/^\d+\. (.*)/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => match.includes('<ul>') ? match : '<ol>' + match + '</ol>');
  html = html.replace(/\n\n/g, '</p><p class="agent-paragraph">');
  html = '<p class="agent-paragraph">' + html + '</p>';
  html = html.replace(/<p class="agent-paragraph"><\/p>/g, '');
  html = html.replace(/<p class="agent-paragraph">\s*<\/p>/g, '');
  return html;
}

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
      .agent-response-panel {
        padding: 4px;
        line-height: 1.6;
        font-size: 14px;
      }
      .agent-response-panel strong {
        font-weight: 600;
        color: var(--color-text-primary, #0f172a);
        display: block;
        margin-top: 12px;
        margin-bottom: 4px;
      }
      .agent-response-panel ol,
      .agent-response-panel ul {
        padding-left: 20px;
        margin: 8px 0;
      }
      .agent-response-panel li {
        margin-bottom: 6px;
      }
      .agent-markdown p.agent-paragraph {
        margin: 6px 0;
      }
      .agent-markdown h1.agent-section-heading,
      .agent-markdown h2.agent-section-heading,
      .agent-markdown h3.agent-section-heading {
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 6px;
        font-size: inherit;
      }
      .agent-markdown code {
        background: rgba(0,0,0,0.06);
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.9em;
      }
      .dark .agent-markdown code {
        background: rgba(255,255,255,0.1);
      }
      .agent-markdown pre {
        background: rgba(0,0,0,0.08);
        padding: 10px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 8px 0;
        font-size: 0.85em;
      }
      .dark .agent-markdown pre {
        background: rgba(255,255,255,0.05);
      }
      .agent-json-field {
        margin: 8px 0;
        padding: 8px 10px;
        border-radius: 8px;
        background: rgba(59,130,246,0.06);
        border-left: 3px solid #3b82f6;
      }
      .dark .agent-json-field {
        background: rgba(59,130,246,0.1);
      }
      .agent-field-label {
        display: block;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3b82f6;
        margin-bottom: 4px;
      }
      .agent-field-value {
        display: block;
        font-size: 0.875rem;
        color: inherit;
      }
      .agent-plan-list {
        margin: 4px 0 0 !important;
        padding-left: 18px !important;
      }
      .agent-plan-list li {
        margin-bottom: 4px !important;
        font-size: 0.85rem;
      }
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

function StepIndicator({ status, active, running }: { status: AgenticStepStatus; active: boolean; running: boolean }) {
  const size = 18;
  const strokeW = 1.5;
  const center = size / 2;
  const r = (size - strokeW * 2) / 2;
  if (status === 'completed') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={center} cy={center} r={r} fill="#10b981" />
        <path d="M6 9.5l3 3 4-5" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (active || running) {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={center} cy={center} r={r} fill="none" stroke="#60a5fa" strokeWidth={strokeW} />
        <circle cx={center} cy={center} r={r * 0.55} fill="#60a5fa" className="animate-pulse" />
      </svg>
    );
  }
  if (status === 'failed') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={center} cy={center} r={r} fill="#ef4444" />
        <path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="#fff" strokeWidth={2} strokeLinecap="round" />
      </svg>
    );
  }
  if (status === 'skipped') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={center} cy={center} r={r} fill="none" stroke="#64748b" strokeWidth={strokeW} />
        <line x1={6} y1={center} x2={size - 6} y2={center} stroke="#64748b" strokeWidth={2} strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle cx={center} cy={center} r={r} fill="none" stroke="#475569" strokeWidth={strokeW} />
    </svg>
  );
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

function bestModelBadgeColor(model: string) {
  const name = model.toLowerCase();
  if (name.includes('xgboost') || name.includes('xgb')) return 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200';
  if (name.includes('prophet')) return 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-200';
  if (name.includes('sarima')) return 'bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-200';
  if (name.includes('holt') || name.includes('winter')) return 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-200';
  return 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200';
}

function buildConfidenceBreakdown() {
  const state = useAppStore.getState();
  const qualityScores = [
    state.timeSeriesForecastResult?.data_quality?.score,
    state.mlForecastResult?.data_quality?.score,
  ].filter((score): score is number => typeof score === 'number' && Number.isFinite(score));
  const avgQuality = qualityScores.length ? qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length : 0.78;
  const completedCount = completedUiCount(state.agenticStepStatuses, state.datasetId);
  const historyBoost = Math.min(0.16, completedCount * 0.018);
  const schemaScore = Math.round(Math.min(100, (0.85 + historyBoost) * 100));
  const typesScore = Math.round(Math.min(100, (0.90 + historyBoost) * 100));
  const qualityScore = Math.round(Math.min(100, (avgQuality + historyBoost) * 100));
  return { schemaScore, typesScore, qualityScore };
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
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: 'smooth' });
  }, [logs]);
  const levelColor: Record<string, string> = {
    INFO: 'text-emerald-400',
    SUCCESS: 'text-teal-400',
    WARNING: 'text-yellow-400',
    ERROR: 'text-red-400',
    AGENT: 'text-blue-400',
  };
  const levelBg: Record<string, string> = {
    INFO: 'bg-emerald-500/10',
    SUCCESS: 'bg-teal-500/10',
    WARNING: 'bg-yellow-500/10',
    ERROR: 'bg-red-500/10',
    AGENT: 'bg-blue-500/10',
  };
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'));
    } catch { /* ignore */ }
  };
  if (!running && logs.length === 0) return null;
  return (
    <div className="relative">
      {logs.length > 0 && (
        <button onClick={handleCopy} className="absolute -top-1 right-0 z-10 rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-200">Copy Logs</button>
      )}
      <div ref={ref} className="ida-agent-terminal max-h-[200px] overflow-auto rounded-xl border border-green-500/20 bg-[#0a0a0a] p-3 font-mono text-xs leading-5 shadow-inner">
        {(logs.length ? logs : ['waiting for execution...']).map((line, index) => {
          const levelMatch = line.match(/\[(INFO|SUCCESS|WARNING|ERROR|AGENT)\]/);
          const level = levelMatch?.[1] ?? 'INFO';
          const colorClass = levelColor[level] ?? 'text-green-400';
          const bgClass = levelBg[level] ?? '';
          return (
            <div key={`${line}-${index}`} className={`${colorClass} ${bgClass} rounded px-1`}>{line.replace(/\[(INFO|SUCCESS|WARNING|ERROR|AGENT)\]\s*/, '')}</div>
          );
        })}
      </div>
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
  const [buttonStatus, setButtonStatus] = React.useState<'idle' | 'loading' | 'success' | 'failed'>('idle');
  const [chatOpen, setChatOpen] = React.useState(false);
  const [chatInput, setChatInput] = React.useState('');
  const [chatMessages, setChatMessages] = React.useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = React.useState(false);
  const lastSuggestedDatasetRef = React.useRef<string | null>(null);
  const chatEndRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const SUGGESTED_QUESTIONS = [
    'Why was this model selected?',
    'What changed during cleaning?',
    'What should I do next?',
    'Are the forecasts reliable?',
    'Explain the best model',
    'What features matter most?',
  ];

  interface InsightAction {
    tab?: string;
    api?: string;
    fix_type?: string;
    parameters?: Record<string, unknown>;
  }
  const [pipelineStats, setPipelineStats] = React.useState({ steps_completed: 0, steps_total: UI_STEPS.length, best_model: 'Pending' as string, forecast_mae: 'N/A' as string, eta_seconds: 0, mae_improvement: 0 });
  const [autoAccept, setAutoAccept] = React.useState(false);
  const pollingRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  const pollPipelineStatus = React.useCallback(async () => {
    if (!agenticSessionId) return;
    try {
      const response = await agenticApiClient.get<StatusResponse>(`/session/${agenticSessionId}/status`);
      const steps = Object.values(response.data.steps ?? {});
      const completed = steps.filter((s) => s === 'completed' || s === 'skipped').length;
      const storeState = useAppStore.getState();
      const best = storeState.mlForecastResult?.training_summary?.model_name ?? storeState.selectedModel ?? storeState.timeSeriesForecastResult?.training_summary?.model_name ?? 'Pending';
      const mae = storeState.mlForecastResult?.metrics?.mae ?? storeState.timeSeriesForecastResult?.metrics?.mae;
      const naive = storeState.mlForecastResult?.naive_baseline ?? storeState.timeSeriesForecastResult?.naive_baseline;
      const maeImprovement = naive?.mae_improvement_pct ?? 0;
      const durations = Object.values(stepDurations).filter((d) => d > 0);
      const avgStepMs = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;
      const remaining = UI_STEPS.length - completed;
      const etaSeconds = Math.round((avgStepMs * remaining) / 1000);
      setPipelineStats({
        steps_completed: completed,
        steps_total: UI_STEPS.length,
        best_model: best,
        forecast_mae: typeof mae === 'number' ? mae.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : 'N/A',
        eta_seconds: etaSeconds,
        mae_improvement: maeImprovement,
      });
      if (completed >= UI_STEPS.length && pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    } catch {
      // ignore polling errors
    }
  }, [agenticSessionId]);

  const handleSuggestedQuestion = (question: string) => {
    setChatInput('');
    void askAgent(question);
  };

  const activeRecommendation = agenticRecommendations[0] ?? null;
  const completedCount = completedUiCount(agenticStepStatuses, datasetId);
  const pendingCount = UI_STEPS.length - completedCount;
  const progress = Math.round((completedCount / UI_STEPS.length) * 100);
  const pipelineComplete = completedCount >= UI_STEPS.length;
  const activeHandlerStep = normalizeStepName(activeRecommendation?.step ?? runningStep ?? '') || null;
  const activeUiStep = !pipelineComplete
    ? (UI_STEPS.find((step) => step.handler === activeHandlerStep) ?? UI_STEPS.find((step) => getUiStatus(step, agenticStepStatuses, datasetId) === 'pending') ?? UI_STEPS[0])
    : UI_STEPS[UI_STEPS.length - 1];
  const latestArtifact = artifacts.at(-1) ?? null;
  const confidence = buildConfidence(activeUiStep, completedCount);
  const context = getStepContext(activeUiStep, artifacts);
  const memoryFacts = buildMemoryFacts(fileName);
  const insight = buildInsight(activeUiStep);
  const bestModel = store.mlForecastResult?.training_summary?.model_name ?? store.selectedModel ?? store.timeSeriesForecastResult?.training_summary?.model_name ?? 'Pending';
  const forecastMae = store.mlForecastResult?.metrics?.mae ?? store.timeSeriesForecastResult?.metrics?.mae;
  const previewReady = ['Data Understanding', 'EDA', 'Data Cleaning'].some((step) => agenticStepStatuses[step] === 'completed');

  React.useEffect(() => {
    if (!agenticSessionId || completedCount === 0) return;
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(() => void pollPipelineStatus(), 3000);
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [agenticSessionId, completedCount, pollPipelineStatus]);

  const appendLog = React.useCallback((line: string) => {
    const stamp = new Date().toLocaleTimeString([], { hour12: false });
    let level = 'INFO';
    if (/^backend completed/i.test(line)) level = 'SUCCESS';
    else if (/^error:/.test(line)) level = 'ERROR';
    else if (/^suggested next/.test(line)) level = 'AGENT';
    else if (/failed.*flagging|data issue|warn/i.test(line)) level = 'WARNING';
    else if (/starting/i.test(line)) level = 'INFO';
    setExecutionLogs((current) => [...current.slice(-80), `[${stamp}] [${level}] ${line}`]);
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
        setArtifacts((current) => {
          const existingSteps = new Set(current.map((a) => a.step));
          const newArtifacts = Object.entries(response.data.results ?? {})
            .filter(([step]) => !existingSteps.has(step))
            .map(([step, result]) => artifactFromPersisted(step, result));
          return [...current, ...newArtifacts].sort((left, right) => left.completedAt - right.completedAt);
        });
        Object.entries(response.data.results).forEach(([step, persisted]) => {
          if (persisted.status === 'completed') {
            const innerResult = (persisted.result as Record<string, unknown> | undefined)?.result ?? persisted.result;
            bridgeStepResultToStore(step, { status: 'completed', step_name: step, result: innerResult } as ExecuteStepResponse);
          }
        });
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

  const downloadPipelineData = async (sessionId: string, format: 'json' | 'csv' | 'pdf') => {
    const ext = format === 'pdf' ? 'pdf' : format;
    const endpoint = format === 'pdf' ? `/session/${sessionId}/report` : `/session/${sessionId}/export/${format}`;
    try {
      const response = await agenticApiClient.get(endpoint, { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = url;
      link.download = `pipeline_${sessionId}.${ext}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch {
      appendLog(`download ${format} failed`);
    }
  };

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
      if (finalStatus === 'completed') {
        bridgeStepResultToStore(stepName, response.data);
      }
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

  const bridgeStepResultToStore = (stepName: string, response: ExecuteStepResponse) => {
    const resultPayload = (response.result ?? {}) as Record<string, unknown>;
    const update: Record<string, unknown> = {};

    switch (stepName) {
      case 'Data Understanding': {
        const columns = resultPayload.columns;
        if (Array.isArray(columns)) update.columns = columns;
        break;
      }
      case 'EDA':
        break;
      case 'Data Cleaning': {
        const cleaned = resultPayload.cleaned_data ?? resultPayload.data;
        if (Array.isArray(cleaned)) update.cleanedData = cleaned;
        const clnLogs = resultPayload.cleaning_logs ?? resultPayload.logs;
        if (Array.isArray(clnLogs)) update.cleaningLogs = clnLogs;
        update.cleaningDone = true;
        break;
      }
      case 'Time Series Forecast': {
        const tsResult: Record<string, unknown> = { status: 'completed' };
        const fields: [string, string][] = [
          ['best_model', 'best_model'], ['smape', 'smape'], ['mae', 'mae'],
          ['rmse', 'rmse'], ['mape', 'mape'], ['reason', 'reason'],
          ['stationarity', 'stationarity'], ['future_forecast', 'future_forecast'],
          ['model_comparison', 'model_comparison_new'],
        ];
        for (const [source, target] of fields) {
          const value = resultPayload[source];
          if (value != null) tsResult[target] = value;
        }
        const insight = resultPayload.insight as Record<string, unknown> | undefined;
        if (insight) {
          tsResult.insight = insight;
          tsResult.analysis = String(insight.insight_text ?? '');
        }
        tsResult.history = resultPayload.history ?? [];
        tsResult.test_forecast = resultPayload.test_forecast ?? [];
        const rmae = resultPayload.mae;
        tsResult.metrics = resultPayload.metrics ?? { mae: typeof rmae === 'number' ? rmae : 0, rmse: typeof resultPayload.rmse === 'number' ? resultPayload.rmse : 0, mape: typeof resultPayload.mape === 'number' ? resultPayload.mape : 0 };
        tsResult.training_summary = resultPayload.training_summary ?? { model_type: String(resultPayload.best_model ?? ''), model_name: String(resultPayload.best_model ?? ''), status: 'completed' };
        update.timeSeriesForecastResult = tsResult;
        break;
      }
      case 'ML Forecast': {
        if (Object.keys(resultPayload).length > 0) update.mlForecastResult = resultPayload;
        break;
      }
      case 'Loss Forecast': {
        const lossData = resultPayload.forecast ?? resultPayload;
        if (Array.isArray(lossData)) update.lossForecast = lossData;
        if (resultPayload.summary) update.lossSummary = resultPayload.summary;
        break;
      }
      case 'Profit Forecast': {
        const profitData = resultPayload.forecast ?? resultPayload;
        if (Array.isArray(profitData)) update.profitForecast = profitData;
        break;
      }
      case 'ML Assistant': {
        const modelName = resultPayload.model_name ?? resultPayload.model_type;
        if (modelName) update.selectedModel = String(modelName);
        if (resultPayload.metrics) update.modelMetrics = resultPayload.metrics;
        if (resultPayload.model_id) update.modelId = String(resultPayload.model_id);
        update.modelTrained = true;
        break;
      }
      case 'Prediction': {
        const predResult = resultPayload.prediction_label ?? resultPayload.prediction;
        if (predResult != null) update.predictionResult = predResult;
        break;
      }
      case 'Report Generation': {
        update.reportGenerated = true;
        const reportUrl = resultPayload.download_url ?? resultPayload.url;
        if (reportUrl) update.reportUrl = String(reportUrl);
        break;
      }
    }

    if (Object.keys(update).length > 0) {
      useAppStore.setState(update as Partial<typeof store>);
    }
  };

  const acceptRecommendation = async () => {
    if (!activeRecommendation) return;
    setButtonStatus('loading');
    const sessionId = agenticSessionId ?? await suggestNextSteps();
    if (!sessionId) { setButtonStatus('idle'); return; }
    try {
      await executeStep(normalizeStepName(activeRecommendation.step) as HandlerStep, sessionId);
      setButtonStatus('success');
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } catch {
      setButtonStatus('failed');
      return;
    }
    setButtonStatus('idle');
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
    setAutoAccept(true);
    appendLog('Run All started with auto-accept enabled');
    const total = EXECUTABLE_STEPS.length;
    try {
      for (let index = 0; index < total; index++) {
        const step = EXECUTABLE_STEPS[index];
        const status = useAppStore.getState().agenticStepStatuses[step];
        if (status === 'completed' || status === 'skipped') {
          appendLog(`Run All: ${step} already ${status}, skipping`);
          continue;
        }
        appendLog(`Run All: executing ${step} (${index + 1}/${total})`);
        try {
          await executeStep(step as HandlerStep, sessionId);
          await pollPipelineStatus();
        } catch {
          appendLog(`Run All: ${step} failed — flagging and continuing`);
          setAgenticStepStatus(step, 'failed');
          await pollPipelineStatus();
        }
      }
      appendLog('Run All: pipeline complete');
    } finally {
      setRunningAll(false);
      setAutoAccept(false);
      void pollPipelineStatus();
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
    const { setActiveTab, setHighlightedColumns, columns } = useAppStore.getState();
    const columnsNeedingAttention = columns.filter((c) => c.nullCount > 0).map((c) => c.name);
    setHighlightedColumns(columnsNeedingAttention.slice(0, 5));
    setActiveTab('cleaning');
  };

  function tryParseStructuredInsight(raw: string): { action?: InsightAction } | null {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && parsed.action) return parsed;
    } catch {
      /* not JSON */
    }
    return null;
  }

  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-xl dark:border-slate-800 dark:bg-slate-950">
      <AgenticWorkspaceStyles />
      <header className="flex flex-col gap-3 border-b border-slate-800 bg-[#0f172a] p-4 text-white lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <div className="relative shrink-0">
            <svg className="absolute inset-0 h-14 w-14 -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="rgba(59,130,246,0.15)" strokeWidth="2.5" />
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 15.5}`} strokeDashoffset={`${2 * Math.PI * 15.5 * (1 - completedCount / UI_STEPS.length)}`} style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
            </svg>
            <div className="grid h-14 w-14 place-items-center rounded-full bg-blue-600 text-base font-black text-white ring-2 ring-blue-300/30">IDA</div>
          </div>
          <div className="min-w-0">
            <p className="truncate text-lg font-bold leading-tight">{fileName ?? 'No dataset selected'}</p>
            <p className="mt-0.5 text-xs text-slate-400">{store.totalRows?.toLocaleString() ?? 0} rows | {store.columns.length ?? 0} columns</p>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <Badge className={cn('rounded-full border px-2.5 py-0.5 text-xs', runningStep ? 'animate-pulse border-blue-400/40 bg-blue-500/15 text-blue-200' : health?.agentic_enabled === false ? 'border-red-400/40 bg-red-500/15 text-red-200' : 'border-emerald-400/40 bg-emerald-500/15 text-emerald-200')}>
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
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="outline" disabled={!agenticSessionId || completedCount <= 1} className="border-slate-600 bg-slate-900 text-slate-100 hover:bg-slate-800 hover:text-white">
                <Download className="mr-2 h-4 w-4" />
                Download Results
                <ChevronDown className="ml-1 h-3.5 w-3.5 opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 border-slate-700 bg-slate-900 text-slate-100">
              <DropdownMenuItem onClick={() => agenticSessionId && void downloadPipelineData(agenticSessionId, 'json')} className="cursor-pointer hover:bg-slate-800">
                <FileJson className="mr-2 h-4 w-4 text-blue-400" />
                <div>
                  <p className="text-sm font-medium">Download as JSON</p>
                  <p className="text-[10px] text-slate-400">Full pipeline state as JSON</p>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => agenticSessionId && void downloadPipelineData(agenticSessionId, 'csv')} className="cursor-pointer hover:bg-slate-800">
                <FileSpreadsheet className="mr-2 h-4 w-4 text-emerald-400" />
                <div>
                  <p className="text-sm font-medium">Download as CSV</p>
                  <p className="text-[10px] text-slate-400">Tabular data as CSV archive</p>
                </div>
              </DropdownMenuItem>
              <DropdownMenuSeparator className="bg-slate-700" />
              <DropdownMenuItem onClick={() => agenticSessionId && void downloadPipelineData(agenticSessionId, 'pdf')} className="cursor-pointer hover:bg-slate-800">
                <FileText className="mr-2 h-4 w-4 text-amber-400" />
                <div>
                  <p className="text-sm font-medium">Pipeline Report (PDF)</p>
                  <p className="text-[10px] text-slate-400">Complete report with visuals</p>
                </div>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <div className="grid min-h-[720px] lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="border-r border-slate-800 bg-slate-950 p-4 text-slate-200">
          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-semibold uppercase tracking-wide text-slate-400">Pipeline</span>
              <span className="font-mono text-blue-300">{completedCount}/{UI_STEPS.length}</span>
            </div>
            <div className="relative h-6 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-blue-500 transition-all duration-700 ease-out" style={{ width: `${progress}%` }} />
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold tracking-wide text-white mix-blend-difference">{Math.round(progress)}% complete</span>
            </div>
          </div>
          <div className="space-y-1.5">
            {UI_STEPS.map((step) => {
              const status = getUiStatus(step, agenticStepStatuses, datasetId);
              const active = activeUiStep?.id === step.id || runningStep === step.handler;
              const running = runningStep === step.handler;
              const duration = step.handler ? formatDuration(stepDurations[step.handler]) : datasetId ? '0.1s' : null;
              const Icon = step.icon;
              return (
                <div key={step.id} className={cn('rounded-lg border border-transparent px-3 py-2 transition', active && 'border-l-4 border-l-blue-500 bg-blue-500/10')}>
                  <div className="flex items-center gap-2">
                    <StepIndicator status={status} active={active} running={running} />
                    <Icon className="h-4 w-4 shrink-0 text-slate-400" />
                    <span className={cn('min-w-0 flex-1 truncate text-sm font-medium', active ? 'text-white' : 'text-slate-300')}>{step.label}</span>
                    {status === 'completed' && duration && <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">{duration}</span>}
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

            {activeRecommendation && activeUiStep && (
              <div className="rounded-xl border border-l-4 border-l-blue-500 bg-blue-50 p-4 shadow-sm dark:border-slate-700 dark:border-l-blue-400 dark:bg-slate-900/80">
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 text-lg">➤</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm font-semibold text-slate-950 dark:text-slate-100">Suggested Next Step</h3>
                      <Badge className="bg-blue-600 text-white text-[10px]">{activeRecommendation.reason ? `${Math.min(99, Math.round(confidence / 10) * 10 + 50)}% match` : 'Recommended'}</Badge>
                    </div>
                    <p className="mt-0.5 text-base font-bold text-blue-700 dark:text-blue-300">{activeUiStep.label}</p>
                    <div className="mt-3 grid gap-2 text-xs md:grid-cols-3">
                      <div className="rounded-lg border border-blue-200 bg-white p-2.5 dark:border-blue-800 dark:bg-slate-950">
                        <span className="font-semibold text-slate-500 dark:text-slate-400">Why</span>
                        <p className="mt-0.5 text-slate-700 dark:text-slate-300">{activeRecommendation.reason || `Continue with ${activeUiStep.label} for deeper analysis.`}</p>
                      </div>
                      <div className="rounded-lg border border-blue-200 bg-white p-2.5 dark:border-blue-800 dark:bg-slate-950">
                        <span className="font-semibold text-slate-500 dark:text-slate-400">Risk</span>
                        <p className="mt-0.5 text-slate-700 dark:text-slate-300">{context.issues.length ? context.issues.slice(0, 2).join('; ') : 'No known risks for this step.'}</p>
                      </div>
                      <div className="rounded-lg border border-blue-200 bg-white p-2.5 dark:border-blue-800 dark:bg-slate-950">
                        <span className="font-semibold text-slate-500 dark:text-slate-400">ETA</span>
                        <p className="mt-0.5 text-slate-700 dark:text-slate-300">
                          {(() => {
                            const avg = Object.values(stepDurations).filter(d => d > 0);
                            const mean = avg.length ? avg.reduce((a, b) => a + b, 0) / avg.length : 3000;
                            const est = Math.round(mean / 1000);
                            return est >= 60 ? `~${Math.round(est / 60)} min` : `~${est} seconds`;
                          })()}
                        </p>
                      </div>
                    </div>
                    {context.findings.length > 0 && (
                      <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                        <span className="font-semibold">Last step found:</span> {context.findings.join('; ')}
                      </div>
                    )}
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button size="sm" onClick={() => void acceptRecommendation()} disabled={buttonStatus === 'loading' || runningAll} className="bg-blue-600 text-white hover:bg-blue-500">
                        {buttonStatus === 'loading' ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Play className="mr-1.5 h-3.5 w-3.5" />}
                        {buttonStatus === 'loading' ? 'Running...' : 'Run This Step'}
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => void skipStep()} disabled={runningAll} className="dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800">
                        <SkipForward className="mr-1.5 h-3.5 w-3.5" />
                        Skip
                      </Button>
                    </div>
                  </div>
                </div>
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
                  {context.issues.length > 0 && (
                    <div className="flex gap-2 text-sm text-amber-600 dark:text-amber-400">
                      <span className="mt-0.5 shrink-0">⚡</span>
                      <span><strong>{context.issues.length}</strong> {context.issues.length === 1 ? 'column needs' : 'columns need'} attention</span>
                    </div>
                  )}
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
              <div className="mt-4 space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Confidence breakdown</p>
                {(() => {
                  const breakdown = buildConfidenceBreakdown();
                  const items = [
                    { label: 'Schema', value: breakdown.schemaScore },
                    { label: 'Types', value: breakdown.typesScore },
                    { label: 'Quality', value: breakdown.qualityScore },
                  ];
                  return items.map((item) => (
                    <div key={item.label} className="flex items-center gap-2 text-xs">
                      <span className="w-12 text-right text-slate-400">{item.label}</span>
                      <div className="flex-1 overflow-hidden rounded-full bg-slate-700">
                        <div className="h-2 rounded-full bg-blue-500 transition-all duration-700" style={{ width: `${item.value}%` }} />
                      </div>
                      <span className="w-10 font-mono text-slate-300">{item.value}%</span>
                    </div>
                  ));
                })()}
              </div>
              {expandedDetails && (
                <pre className="mt-4 overflow-auto rounded-lg bg-slate-950 p-3 text-xs text-slate-300">
                  {JSON.stringify({ activeStep: activeUiStep, recommendation: activeRecommendation, context, confidence }, null, 2)}
                </pre>
              )}
              {pipelineComplete ? (
                <div className="mt-4 flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-950/30 dark:text-emerald-200">
                  <CheckCircle2 className="h-5 w-5" />
                  Pipeline complete — all steps have been processed
                </div>
              ) : (
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button onClick={() => void acceptRecommendation()} disabled={!activeRecommendation || buttonStatus === 'loading' || runningAll} className={cn('text-white', buttonStatus === 'failed' ? 'bg-red-600 hover:bg-red-500' : buttonStatus === 'success' ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-blue-600 hover:bg-blue-500')}>
                    {buttonStatus === 'loading' ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Validating step...</> : buttonStatus === 'success' ? <><CheckCircle2 className="mr-2 h-4 w-4" /> Step accepted — loading EDA</> : buttonStatus === 'failed' ? <><XCircle className="mr-2 h-4 w-4" /> Failed — retry?</> : <><Check className="mr-2 h-4 w-4" /> Accept & Continue</>}
                  </Button>
                  <Button variant="outline" onClick={() => void skipStep()} disabled={!activeRecommendation || Boolean(runningStep) || runningAll} className="dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800">
                    <SkipForward className="mr-2 h-4 w-4" />
                    Skip
                  </Button>
                </div>
              )}
            </div>

            <ExecutionLog logs={executionLogs} running={Boolean(runningStep)} />

            {structuredError && (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 dark:border-red-500/30 dark:bg-red-950/30">
                <h3 className="font-semibold text-red-800 dark:text-red-200">{structuredError.step} failed</h3>
                <p className="mt-1 text-sm text-red-700 dark:text-red-300">{structuredError.reason}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" onClick={() => agenticSessionId && void executeStep(structuredError.step as HandlerStep, agenticSessionId)} disabled={runningAll}><RefreshCw className="mr-2 h-4 w-4" />Retry</Button>
                  <Button size="sm" variant="outline" onClick={() => void skipStep(structuredError.step)} disabled={runningAll} className="dark:border-slate-700 dark:text-slate-200">Skip</Button>
                  <Button size="sm" variant="outline" onClick={() => void askAgentToFix()} disabled={runningAll} className="dark:border-slate-700 dark:text-slate-200"><Wand2 className="mr-2 h-4 w-4" />Ask agent to fix</Button>
                </div>
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-4">
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Steps Completed</p>
                <div className="mt-2 flex items-center gap-3">
                  <svg className="shrink-0 -rotate-90" width="44" height="44" viewBox="0 0 36 36">
                    <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(100,116,139,0.2)" strokeWidth="2.5" />
                    <circle cx="18" cy="18" r="15" fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 15}`} strokeDashoffset={`${2 * Math.PI * 15 * (1 - pipelineStats.steps_completed / pipelineStats.steps_total)}`} style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
                  </svg>
                  <span className="text-xl font-bold text-slate-950 dark:text-slate-100">{pipelineStats.steps_completed}/{pipelineStats.steps_total}</span>
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Steps Remaining</p>
                <p className="mt-2 text-xl font-bold text-slate-950 dark:text-slate-100">{Math.max(0, pipelineStats.steps_total - pipelineStats.steps_completed)}</p>
                {pipelineStats.eta_seconds > 0 && <p className="mt-1 text-[11px] text-slate-400">~{pipelineStats.eta_seconds >= 60 ? `${Math.round(pipelineStats.eta_seconds / 60)}m` : `${pipelineStats.eta_seconds}s`} remaining</p>}
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Best Model Selected</p>
                {pipelineStats.best_model === 'Pending' ? (
                  <div className="mt-2 space-y-1.5">
                    <div className="h-4 w-24 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                    <div className="h-3 w-16 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                  </div>
                ) : (
                  <span className={cn('mt-2 inline-block rounded-full px-2.5 py-0.5 text-sm font-bold', bestModelBadgeColor(pipelineStats.best_model))}>{pipelineStats.best_model}</span>
                )}
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Forecast MAE</p>
                {pipelineStats.forecast_mae === 'N/A' ? (
                  <div className="mt-2 space-y-1.5">
                    <div className="h-4 w-20 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                    <div className="h-3 w-14 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
                  </div>
                ) : (
                  <>
                    <p className="mt-2 text-xl font-bold text-slate-950 dark:text-slate-100">{pipelineStats.forecast_mae}</p>
                    {pipelineStats.mae_improvement > 0 && (
                      <p className="mt-1 text-[11px] text-emerald-500">↓ {pipelineStats.mae_improvement}% better than baseline</p>
                    )}
                  </>
                )}
              </div>
            </div>

            <StepResultCard artifact={latestArtifact} />

            {previewReady && <DataPreviewTable />}

            <div className="grid gap-4 xl:grid-cols-2">
              <ForecastResultsCard />
              <ShapPanel />
            </div>

            <div className="rounded-xl border border-l-4 border-amber-200 border-l-amber-500 bg-white p-4 shadow-sm dark:border-slate-700 dark:border-l-amber-400 dark:bg-slate-900">
              <div className="flex items-center gap-2">
                <span className="text-lg">💡</span>
                <h3 className="font-semibold text-slate-950 dark:text-slate-100">Agent Insight</h3>
                <Badge className="ml-auto bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200 text-[10px]">NEW</Badge>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">{insight}</p>
              {context.issues.length > 0 && (
                <div className="mt-3 space-y-1">
                  {context.issues.map((issue) => (
                    <div key={issue} className="flex gap-2 text-sm text-amber-700 dark:text-amber-300">
                      <span className="mt-0.5 shrink-0 text-[10px]">⚠️</span>
                      <span>{issue}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="text-slate-400">Confidence:</span>
                  <div className="inline-flex items-center gap-1">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                      <div className="h-full rounded-full bg-amber-500" style={{ width: `${confidence}%` }} />
                    </div>
                    <span className="font-mono text-slate-500 dark:text-slate-400">{confidence}%</span>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="text-slate-400">Priority:</span>
                  <span className={cn('rounded px-1.5 py-0.5 font-semibold', context.issues.length > 2 ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' : context.issues.length > 0 ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300')}>
                    {context.issues.length > 2 ? 'HIGH' : context.issues.length > 0 ? 'MEDIUM' : 'LOW'}
                  </span>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button size="sm" onClick={() => void applyInsightFix()} className="bg-amber-600 text-white hover:bg-amber-500">
                  <Wand2 className="mr-1.5 h-3.5 w-3.5" />
                  Apply Fix → Data Cleaning
                </Button>
                <Button size="sm" variant="outline" className="dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800">
                  <Eye className="mr-1.5 h-3.5 w-3.5" />
                  Preview Impact
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
                <div className="agent-response-panel max-h-72 space-y-2 overflow-auto rounded-lg border border-slate-100 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                  {chatMessages.length ? (
                    <>
                      {chatMessages.map((message) => (
                        <div key={message.id} className={cn('flex flex-col', message.role === 'user' ? 'items-end' : 'items-start')}>
                          <span className={cn('mb-0.5 text-[10px] font-medium tracking-wide uppercase', message.role === 'user' ? 'text-blue-500' : 'text-slate-400')}>
                            {message.role === 'user' ? 'You' : 'Agent'}
                          </span>
                          {message.role === 'user' ? (
                            <div className="rounded-xl px-3.5 py-2.5 text-sm leading-relaxed bg-blue-600 text-white">
                              {message.content}
                            </div>
                          ) : (
                            <div
                              className="agent-markdown rounded-xl px-3.5 py-2.5 text-sm leading-relaxed bg-white text-slate-700 shadow-sm ring-1 ring-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-700"
                              dangerouslySetInnerHTML={{ __html: renderStructuredOrMarkdown(message.content) }}
                            />
                          )}
                        </div>
                      ))}
                      <div ref={chatEndRef} />
                    </>
                  ) : (
                    <div className="space-y-3 py-2">
                      <p className="text-center text-xs text-slate-400 dark:text-slate-500">Pick a question or type your own</p>
                      <div className="flex flex-wrap justify-center gap-1.5">
                        {SUGGESTED_QUESTIONS.map((question) => (
                          <button
                            key={question}
                            onClick={() => handleSuggestedQuestion(question)}
                            disabled={chatLoading}
                            className="cursor-pointer rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-blue-500 dark:hover:bg-blue-900/30 dark:hover:text-blue-300"
                          >
                            {question}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {chatLoading && (
                    <div className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-blue-50 to-slate-50 px-3 py-2 text-sm text-slate-500 dark:from-blue-950/20 dark:to-slate-900">
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-blue-500" />
                      </span>
                      Agent is thinking…
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <input
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') void sendChat();
                    }}
                    placeholder="Ask anything about the results…"
                    className="h-10 min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500"
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


