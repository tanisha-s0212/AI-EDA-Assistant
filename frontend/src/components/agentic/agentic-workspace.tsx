'use client';

import React from 'react';
import axios from 'axios';
import { AlertCircle, Check, CheckCircle2, Download, Loader2, Play, SkipForward } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import type { AgenticStepStatus, ColumnInfo, Recommendation } from '@/lib/store';
import { cn } from '@/lib/utils';

const agenticApiClient = axios.create({
  baseURL: (process.env.NEXT_PUBLIC_AGENTIC_API_BASE || '/api/agentic').replace(/\/$/, ''),
  withCredentials: true,
});

const PIPELINE_STEPS = [
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

type AgenticHealth = {
  agentic_enabled: boolean;
  db_connected: boolean;
  db_fallback_active: boolean;
};

type SuggestResponse = {
  session_id: string;
  recommendations: Recommendation[];
};

type ExecuteResponse = {
  status: AgenticStepStatus | 'not_yet_wired';
  output_summary: string | null;
  error?: string;
  next_recommendations?: Recommendation[];
};

type StatusResponse = {
  steps: Record<string, AgenticStepStatus>;
  recommendations?: Recommendation[];
};

type AgenticWorkspaceProps = {
  datasetId: string | null;
  fileName: string | null;
};

function statusProgress(statuses: Record<string, AgenticStepStatus>) {
  const completed = PIPELINE_STEPS.filter((step) => ['completed', 'skipped'].includes(statuses[step])).length;
  return Math.round((completed / PIPELINE_STEPS.length) * 100);
}

function statusRank(status: AgenticStepStatus | undefined) {
  if (status === 'completed' || status === 'skipped') return 1;
  return 0;
}

function HighlightedText({ text }: { text: string }) {
  const pattern = /(MAE|RMSE|MAPE|forecast|prediction|predicted|value|result)\s*[:=]?\s*([₹$€£]?\s*-?\d[\d,]*(?:\.\d+)?%?)/gi;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > lastIndex) parts.push(text.slice(lastIndex, index));
    parts.push(
      <span key={`${match[0]}-${index}`} className="rounded-md bg-blue-500/10 px-1.5 py-0.5 font-mono text-[0.92em] font-semibold text-blue-100 ring-1 ring-blue-300/20">
        {match[0]}
      </span>,
    );
    lastIndex = index + match[0].length;
  }

  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return <>{parts}</>;
}

function AgenticWorkspaceStyles() {
  return (
    <style jsx global>{`
      @keyframes ida-agent-pulse-blue {
        0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.42), 0 0 34px rgba(59, 130, 246, 0.2); }
        50% { box-shadow: 0 0 0 9px rgba(59, 130, 246, 0), 0 0 44px rgba(99, 102, 241, 0.34); }
      }
      @keyframes ida-agent-pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.44), 0 0 30px rgba(34, 197, 94, 0.22); }
        50% { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0), 0 0 42px rgba(16, 185, 129, 0.32); }
      }
      @keyframes ida-agent-ring-spin {
        to { transform: rotate(360deg); }
      }
      @keyframes ida-agent-fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes ida-agent-active-step {
        0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.42); }
        50% { transform: scale(1.06); box-shadow: 0 0 0 7px rgba(59, 130, 246, 0); }
      }
      @keyframes ida-agent-download-bob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(2px); }
      }
      .ida-agent-fade-in { animation: ida-agent-fade-in 240ms ease-out both; }
      .ida-agent-idle { animation: ida-agent-pulse-blue 2.4s ease-in-out infinite; }
      .ida-agent-complete { animation: ida-agent-pulse-green 2s ease-in-out infinite; }
      .ida-agent-processing::before {
        content: "";
        position: absolute;
        inset: -3px;
        border-radius: 9999px;
        background: conic-gradient(from 0deg, #2563eb, #7c3aed, #06b6d4, #2563eb);
        animation: ida-agent-ring-spin 900ms linear infinite;
        z-index: -1;
      }
      .ida-agent-download:hover svg { animation: ida-agent-download-bob 650ms ease-in-out infinite; }
    `}</style>
  );
}

type StoreTabId = ReturnType<typeof useAppStore.getState>['activeTab'];

const STEP_TO_TAB: Record<string, StoreTabId> = {
  'Data Understanding': 'understanding',
  EDA: 'eda',
  'Data Cleaning': 'cleaning',
  'Time Series Forecast': 'forecast_ts',
  'ML Forecast': 'forecast_ml',
  'Loss Forecast': 'loss_forecast',
  'Profit Forecast': 'profit_forecast',
  'ML Assistant': 'ml',
  Prediction: 'prediction',
  'Report Generation': 'report',
};

function getNextRecommendation(completedStep: string, findings: string[] = []): Recommendation[] {
  const currentIndex = PIPELINE_STEPS.indexOf(completedStep);
  const state = useAppStore.getState();
  const nextStep = PIPELINE_STEPS.slice(currentIndex + 1).find((step) => {
    const status = state.agenticStepStatuses[step];
    return status !== 'completed' && status !== 'skipped';
  });

  if (!nextStep) return [];

  return [{
    step: nextStep,
    reason: 'The previous approved action finished, so the assistant is ready to continue the application workflow.',
    findings: findings.length ? findings : [`${completedStep} completed.`, 'The next action will run only after approval.'],
  }];
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

function getAutoFeatureColumns(targetColumn: string) {
  const { columns } = useAppStore.getState();
  return columns
    .filter((column) => column.name !== targetColumn && column.role !== 'identifier')
    .map((column) => column.name)
    .slice(0, 30);
}

function inferProblemType(targetColumn: string): 'regression' | 'classification' {
  const { columns } = useAppStore.getState();
  const target = columns.find((column) => column.name === targetColumn);
  if (!target) return 'regression';
  return target.role === 'numeric' && target.uniqueCount > 12 ? 'regression' : 'classification';
}

function normalizePredictionFeatures(features: string[]) {
  const state = useAppStore.getState();
  const sampleRow = (state.cleanedData ?? state.rawData ?? [])[0] ?? {};
  const columnsByName = new Map(state.columns.map((column) => [column.name, column]));

  return Object.fromEntries(features.map((feature) => {
    const value = sampleRow[feature];
    const column = columnsByName.get(feature);
    if (column?.role === 'numeric') {
      const numericValue = typeof value === 'number' ? value : Number(value);
      return [feature, Number.isFinite(numericValue) ? numericValue : 0];
    }
    return [feature, value == null || value === '' ? 'unknown' : String(value)];
  }));
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
  const [health, setHealth] = React.useState<AgenticHealth | null>(null);
  const [isSuggesting, setIsSuggesting] = React.useState(false);
  const [runningStep, setRunningStep] = React.useState<string | null>(null);
  const [banner, setBanner] = React.useState<string | null>(null);
  const [lastSummary, setLastSummary] = React.useState<string | null>(null);
  const lastSuggestedDatasetRef = React.useRef<string | null>(null);

  const activeRecommendation = agenticRecommendations[0] ?? null;
  const progress = statusProgress(agenticStepStatuses);
  const workflowComplete = PIPELINE_STEPS.every((step) => ['completed', 'skipped'].includes(agenticStepStatuses[step]));
  const hasFailure = PIPELINE_STEPS.some((step) => agenticStepStatuses[step] === 'failed');
  const activeTimelineStep = runningStep ?? activeRecommendation?.step ?? null;
  const completedStepCount = PIPELINE_STEPS.filter((step) => ['completed', 'skipped'].includes(agenticStepStatuses[step])).length;

  const refreshHealth = React.useCallback(async () => {
    try {
      const response = await agenticApiClient.get<AgenticHealth>('/health');
      setHealth(response.data);
      setBanner(response.data.db_connected ? null : 'Running in offline mode — decisions will not persist across sessions');
    } catch {
      setHealth(null);
      setBanner('Agentic layer unavailable — manual mode active');
    }
  }, []);

  React.useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  React.useEffect(() => {
    if (!agenticSessionId) return;
    const pollStatus = async () => {
      try {
        const response = await agenticApiClient.get<StatusResponse>(`/session/${agenticSessionId}/status`);
        const currentStatuses = useAppStore.getState().agenticStepStatuses;
        Object.entries(response.data.steps ?? {}).forEach(([step, status]) => {
          const currentStatus = currentStatuses[step];
          if (currentStatus === 'completed' || currentStatus === 'skipped' || currentStatus === 'running') return;
          setAgenticStepStatus(step, status);
        });
        if (response.data.recommendations) {
          const hasActiveLocalRecommendation = useAppStore.getState().agenticRecommendations.length > 0;
          if (!hasActiveLocalRecommendation) {
            setAgenticRecommendations(response.data.recommendations.slice(0, 1));
          }
        }
        setAgenticLastSyncedAt(Date.now());
      } catch {
        setBanner('Agentic layer unavailable — manual mode active');
      }
    };

    void pollStatus();
    const interval = window.setInterval(() => {
      void pollStatus();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [agenticSessionId, setAgenticLastSyncedAt, setAgenticRecommendations, setAgenticStepStatus]);

  const suggestNextSteps = React.useCallback(async () => {
    if (!datasetId) return;
    setIsSuggesting(true);
    setBanner(null);
    try {
      const response = await agenticApiClient.post<SuggestResponse>('/suggest-next-steps', {
        dataset_path: datasetId,
      });
      setAgenticSessionId(response.data.session_id);
      setAgenticRecommendations(response.data.recommendations.slice(0, 1));
      PIPELINE_STEPS.forEach((step) => setAgenticStepStatus(step, 'pending'));
      setAgenticLastSyncedAt(Date.now());
    } catch (error) {
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable — manual mode active'));
    } finally {
      setIsSuggesting(false);
    }
  }, [datasetId, setAgenticLastSyncedAt, setAgenticRecommendations, setAgenticSessionId, setAgenticStepStatus]);

  React.useEffect(() => {
    if (!datasetId || health?.agentic_enabled === false || lastSuggestedDatasetRef.current === datasetId) return;
    lastSuggestedDatasetRef.current = datasetId;
    void suggestNextSteps();
  }, [datasetId, health?.agentic_enabled, suggestNextSteps]);

  const downloadReport = async (sessionId: string) => {
    const response = await agenticApiClient.get(`/session/${sessionId}/report`, { responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agentic_run_${sessionId}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const runApprovedStep = async (stepName: string) => {
    const state = useAppStore.getState();
    const tab = STEP_TO_TAB[stepName];
    if (tab) state.setActiveTab(tab);

    if (stepName === 'Data Understanding' || stepName === 'EDA') {
      return `${stepName} reviewed from the uploaded dataset profile. The application tab is open for inspection.`;
    }

    if (stepName === 'Data Cleaning') {
      if (!state.datasetId) {
        useAppStore.setState({ cleanedData: state.rawData, cleaningDone: true, cleanedRowCount: state.rawData?.length ?? 0 });
        return 'Cleaning marked complete for the in-browser dataset preview.';
      }

      const response = await apiClient.post('/clean-dataset', {
        dataset_id: state.datasetId,
        remove_duplicates: true,
        handle_missing: true,
        convert_dates: true,
        standardize_names: true,
        infer_dtypes: true,
      });
      const result = response.data;
      useAppStore.setState({
        rawData: result.data,
        cleanedData: result.data,
        columns: (result.columns ?? state.columns).map((column: ColumnInfo) => ({ ...column, sample: Array.isArray(column.sample) ? column.sample : [] })),
        cleaningLogs: result.logs ?? [],
        cleaningDone: true,
        cleanedRowCount: result.rowCount ?? result.data?.length ?? state.totalRows,
        loadedRowCount: result.loadedRowCount ?? result.data?.length ?? state.loadedRowCount,
        previewLoaded: Boolean(result.previewLoaded),
        duplicates: result.duplicates ?? 0,
        reportGenerated: false,
        reportUrl: null,
      });
      return `Data Cleaning completed with ${(result.logs ?? []).length} recorded operation(s).`;
    }

    if (stepName === 'Time Series Forecast' || stepName === 'ML Forecast') {
      const latest = useAppStore.getState();
      const dateColumn = getPreferredDateColumn();
      const targetColumn = getPreferredTargetColumn();
      if (!dateColumn || !targetColumn) {
        throw new Error('A date-like column and numeric target are required for automated forecasting.');
      }

      const isMlForecast = stepName === 'ML Forecast';
      const response = await apiClient.post(isMlForecast ? '/forecast/ml/run' : '/forecast/ts/run', {
        dataset_id: latest.datasetId ?? null,
        data: latest.datasetId ? [] : latest.cleanedData ?? latest.rawData ?? [],
        date_column: dateColumn,
        target_column: targetColumn,
        forecast_periods: 3,
        test_percentage: 20,
        ...(isMlForecast
          ? { lag_periods: 3, model_type: 'gradient_boosting', feature_groups: ['trend', 'calendar', 'lags', 'rolling'] }
          : { model_type: 'sarima' }),
      });
      useAppStore.setState(isMlForecast ? { mlForecastResult: response.data } : { timeSeriesForecastResult: response.data });
      return `${stepName} completed for ${targetColumn} over ${dateColumn}.`;
    }

    if (stepName === 'Loss Forecast') {
      if (!state.datasetId) throw new Error('Loss Forecast needs a cached dataset id.');
      await state.runLossForecast(state.datasetId, 30, { confirmedAssumptions: true });
      return 'Loss Forecast completed and stored in the application workflow.';
    }

    if (stepName === 'Profit Forecast') {
      if (!state.datasetId) throw new Error('Profit Forecast needs a cached dataset id.');
      await state.runProfitForecast(state.datasetId, 30, {
        confirmedAssumptions: true,
        scenarios: { optimistic: 1.15, baseline: 1.0, pessimistic: 0.85 },
      });
      return 'Profit Forecast completed and stored in the application workflow.';
    }

    if (stepName === 'ML Assistant') {
      const latest = useAppStore.getState();
      const targetColumn = getPreferredTargetColumn();
      const featureColumns = getAutoFeatureColumns(targetColumn);
      if (!targetColumn || featureColumns.length === 0) {
        throw new Error('Automated model training needs one target and at least one feature column.');
      }

      const problemType = inferProblemType(targetColumn);
      const modelType = problemType === 'regression' ? 'ridge_regression' : 'random_forest';
      const response = await apiClient.post('/train', {
        data: latest.datasetId ? [] : latest.cleanedData ?? latest.rawData ?? [],
        dataset_id: latest.datasetId ?? null,
        target_column: targetColumn,
        feature_columns: featureColumns,
        problem_type: problemType,
        model_type: modelType,
        test_size: 0.2,
        random_state: 42,
        cv_folds: 5,
        training_mode: 'fast',
      });
      const result = response.data;
      useAppStore.setState({
        targetColumn,
        problemType,
        selectedFeatures: featureColumns,
        selectedModel: modelType,
        modelId: result.model_id ?? null,
        modelMetrics: result.metrics ?? null,
        modelTrained: true,
        featureImportance: result.feature_importance ?? [],
        uploadedModel: {
          name: modelType,
          type: modelType,
          target: targetColumn,
          problem: problemType,
          trainedAt: new Date().toISOString(),
          metrics: result.metrics ?? {},
          features: featureColumns,
        },
      });
      return `${modelType} trained for ${targetColumn} using ${featureColumns.length} feature(s).`;
    }

    if (stepName === 'Prediction') {
      const latest = useAppStore.getState();
      if (!latest.modelId) throw new Error('Prediction needs a trained model id.');
      const features = normalizePredictionFeatures(latest.selectedFeatures);
      const response = await apiClient.post('/predict', { model_id: latest.modelId, features });
      const predictionValue = response.data.prediction_label ?? response.data.prediction;
      useAppStore.setState({
        predictionResult: predictionValue,
        predictionAnalysis: `Automated prediction generated from the approved agentic flow for ${latest.targetColumn ?? 'the selected target'}.`,
        predictionProbabilities: response.data.probabilities ?? null,
        predictionHistory: [
          ...latest.predictionHistory,
          {
            id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
            prediction: predictionValue,
            confidence: response.data.confidence,
            probabilities: response.data.probabilities,
            features,
            timestamp: new Date().toISOString(),
          },
        ],
      });
      return `Prediction completed with result ${String(predictionValue)}.`;
    }

    if (stepName === 'Report Generation') {
      if (agenticSessionId) await downloadReport(agenticSessionId);
      return 'Agentic run report downloaded and the application Report tab is open for the final workflow PDF.';
    }

    return `${stepName} completed.`;
  };

  const acceptRecommendation = async () => {
    if (!activeRecommendation || !agenticSessionId) return;

    setRunningStep(activeRecommendation.step);
    setAgenticStepStatus(activeRecommendation.step, 'running');
    try {
      const outputSummary = await runApprovedStep(activeRecommendation.step);
      await agenticApiClient.post('/decision', {
        session_id: agenticSessionId,
        step_name: activeRecommendation.step,
        decision: 'accepted',
        reasoning: outputSummary,
      });
      setAgenticStepStatus(activeRecommendation.step, 'completed');
      setLastSummary(outputSummary);
      setAgenticRecommendations(getNextRecommendation(activeRecommendation.step, [outputSummary]));
    } catch (error) {
      setAgenticStepStatus(activeRecommendation.step, 'failed');
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable — manual mode active'));
    } finally {
      setRunningStep(null);
    }
  };

  const skipRecommendation = async () => {
    if (!activeRecommendation || !agenticSessionId) return;
    try {
      const response = await agenticApiClient.post<{ next_recommendations?: Recommendation[] }>('/decision', {
        session_id: agenticSessionId,
        step_name: activeRecommendation.step,
        decision: 'skipped',
        reasoning: 'Skipped from agentic workspace.',
      });
      setAgenticStepStatus(activeRecommendation.step, 'skipped');
      setAgenticRecommendations(response.data.next_recommendations ?? []);
    } catch (error) {
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable — manual mode active'));
    }
  };

  return (
    <section className="relative mb-5 overflow-hidden rounded-2xl border border-white/10 bg-[linear-gradient(145deg,#0f172a_0%,#172033_48%,#1e293b_100%)] p-4 text-slate-100 shadow-[0_30px_90px_-44px_rgba(2,8,23,0.9)] ring-1 ring-white/10">
      <AgenticWorkspaceStyles />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_0%,rgba(59,130,246,0.18),transparent_34%),radial-gradient(circle_at_88%_20%,rgba(124,58,237,0.14),transparent_30%)]" />
      <div className="pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-white/35 to-transparent" />
      {banner && (
        <Alert className="ida-agent-fade-in relative z-10 mb-4 border-amber-300/25 bg-amber-300/10 text-amber-50 shadow-[0_18px_42px_-30px_rgba(251,191,36,0.8)] backdrop-blur-xl">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{banner}</AlertDescription>
        </Alert>
      )}

      <div className="relative z-10 grid gap-4 lg:grid-cols-[1fr_19rem]">
        <div className="min-w-0 space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-3">
                <span
                  className={cn(
                    'relative grid h-9 w-9 place-items-center rounded-full bg-slate-950 text-[10px] font-black text-white ring-1 ring-white/20',
                    runningStep || isSuggesting ? 'ida-agent-processing' : workflowComplete ? 'ida-agent-complete' : 'ida-agent-idle',
                  )}
                >
                  IDA
                </span>
                <div>
                  <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-blue-200/70">Agent Workspace</p>
                  <p className="mt-1 text-xs text-slate-400">{runningStep ? `Processing ${runningStep}` : workflowComplete ? 'Workflow complete' : 'Idle and ready'}</p>
                </div>
              </div>
              <h2 className="mt-4 truncate text-xl font-semibold tracking-normal text-white">
                {fileName ?? 'No dataset selected'}
              </h2>
            </div>
            <Button
              size="sm"
              onClick={suggestNextSteps}
              disabled={!datasetId || isSuggesting || health?.agentic_enabled === false}
              className="rounded-lg border border-white/10 bg-blue-500/90 text-white shadow-[0_14px_34px_-22px_rgba(59,130,246,0.8)] transition-all duration-200 ease-out hover:-translate-y-0.5 hover:bg-blue-400 hover:shadow-[0_20px_42px_-22px_rgba(59,130,246,0.95)]"
            >
              {isSuggesting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              Suggest Next Steps
            </Button>
          </div>

          {activeRecommendation ? (
            <Card
              className={cn(
                'ida-agent-fade-in gap-4 rounded-xl border border-white/15 bg-white/[0.075] py-4 text-slate-100 shadow-[0_24px_70px_-45px_rgba(15,23,42,0.95)] backdrop-blur-2xl transition-all duration-200 ease-out',
                hasFailure
                  ? 'bg-[linear-gradient(#1e293b,#1e293b)_padding-box,linear-gradient(135deg,#ef4444,#f97316)_border-box]'
                  : 'bg-[linear-gradient(rgba(30,41,59,0.78),rgba(15,23,42,0.78))_padding-box,linear-gradient(135deg,#2563eb,#7c3aed)_border-box]',
              )}
            >
              <CardHeader className="px-4">
                <div className="flex items-start justify-between gap-3">
                  <CardTitle className="text-base font-semibold text-white">{activeRecommendation.step}</CardTitle>
                  <Badge variant="outline" className="rounded-md border-blue-300/30 bg-blue-400/10 text-blue-100">Recommended</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3 px-4">
                <p className="text-sm font-normal text-slate-300">{activeRecommendation.reason}</p>
                <div className="space-y-1">
                  {activeRecommendation.findings.map((finding) => (
                    <div key={finding} className="flex gap-2 text-sm font-normal text-slate-100">
                      <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
                      <span><HighlightedText text={finding} /></span>
                    </div>
                  ))}
                </div>
                {lastSummary && (
                  <p className="ida-agent-fade-in rounded-lg border border-white/10 bg-white/[0.075] p-3 text-sm font-normal text-slate-300 backdrop-blur-xl">
                    <HighlightedText text={lastSummary} />
                  </p>
                )}
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    onClick={acceptRecommendation}
                    disabled={Boolean(runningStep)}
                    className="rounded-lg border-l-4 border-l-blue-300 bg-blue-500 text-white shadow-[0_12px_30px_-22px_rgba(37,99,235,0.9)] transition-all duration-200 ease-out hover:-translate-y-0.5 hover:bg-blue-400 hover:shadow-[0_22px_46px_-24px_rgba(37,99,235,1)]"
                  >
                    {runningStep ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : activeRecommendation.step === 'Report Generation' ? <Download className="mr-2 h-4 w-4" /> : <Check className="mr-2 h-4 w-4" />}
                    Accept & Continue
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={skipRecommendation}
                    disabled={Boolean(runningStep)}
                    className="rounded-lg border-white/15 border-l-slate-400 bg-white/[0.075] text-slate-100 backdrop-blur-xl transition-all duration-200 ease-out hover:-translate-y-0.5 hover:bg-white/12 hover:text-white hover:shadow-[0_20px_42px_-30px_rgba(148,163,184,0.9)]"
                  >
                    <SkipForward className="mr-2 h-4 w-4" />
                    Skip
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div
              className={cn(
                'ida-agent-fade-in rounded-xl border border-white/15 bg-white/[0.07] p-4 text-sm font-normal text-slate-300 shadow-[0_24px_60px_-44px_rgba(15,23,42,0.9)] backdrop-blur-2xl',
                workflowComplete
                  ? 'bg-[linear-gradient(rgba(30,41,59,0.72),rgba(15,23,42,0.72))_padding-box,linear-gradient(135deg,#2563eb,#7c3aed)_border-box]'
                  : 'border-dashed',
              )}
            >
              {workflowComplete && agenticSessionId ? (
                <div className="flex flex-col gap-3">
                  <span>The approved agentic flow is complete. Download the consolidated run report when you are ready.</span>
                  <Button
                    size="sm"
                    onClick={() => void downloadReport(agenticSessionId)}
                    className="ida-agent-download w-full rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-[0_18px_44px_-24px_rgba(79,70,229,0.95)] transition-all duration-200 ease-out hover:brightness-110"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download Report
                  </Button>
                </div>
              ) : (
                'Upload or select a dataset, then ask the agent for the next approved step.'
              )}
            </div>
          )}
        </div>

        <aside className="rounded-xl border border-white/15 bg-white/[0.075] p-4 shadow-[0_24px_70px_-48px_rgba(15,23,42,0.95)] backdrop-blur-2xl">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-200/70">Pipeline</p>
            <span className="rounded-md bg-white/10 px-2 py-1 font-mono text-xs font-semibold text-blue-100">{progress}%</span>
          </div>
          <div className="relative">
            <div className="absolute left-[0.875rem] top-4 h-[calc(100%-2rem)] w-px bg-slate-600/60" />
            <div
              className="absolute left-[0.875rem] top-4 w-px bg-gradient-to-b from-blue-400 to-indigo-400 transition-all duration-500 ease-out"
              style={{ height: `calc((100% - 2rem) * ${Math.max(0, completedStepCount - 1)} / ${Math.max(1, PIPELINE_STEPS.length - 1)})` }}
            />
            <div className="space-y-3">
              {PIPELINE_STEPS.map((step) => {
              const status = agenticStepStatuses[step] ?? 'pending';
              const isCompleted = status === 'completed' || status === 'skipped';
              const isActive = status === 'running' || step === activeTimelineStep;
              return (
                <div key={step} className="relative grid grid-cols-[1.75rem_1fr_auto] items-start gap-3 text-sm">
                  <span
                    className={cn(
                      'relative z-10 grid h-7 w-7 place-items-center rounded-full border text-[10px] transition-all duration-300 ease-out',
                      isCompleted && 'border-blue-300 bg-blue-500 text-white shadow-[0_0_24px_rgba(59,130,246,0.42)]',
                      isActive && !isCompleted && 'border-blue-300 bg-blue-400/20 text-blue-100',
                      status === 'failed' && 'border-red-300 bg-red-500/20 text-red-100',
                      !isCompleted && !isActive && status !== 'failed' && 'border-slate-500 bg-slate-700/70 text-slate-400',
                    )}
                    style={isActive && !isCompleted ? { animation: 'ida-agent-active-step 1.5s ease-in-out infinite' } : undefined}
                  >
                    {isCompleted ? <CheckCircle2 className="h-4 w-4" /> : statusRank(status) ? <Check className="h-3.5 w-3.5" /> : ''}
                  </span>
                  <div className="min-w-0 pb-1">
                    <p className={cn('truncate font-semibold', isCompleted ? 'text-white' : isActive ? 'text-blue-100' : 'text-slate-400')}>
                      {step}
                    </p>
                    <p className="mt-0.5 truncate text-xs font-normal text-slate-500">
                      {isCompleted ? 'Completed' : isActive ? 'Active step' : status === 'failed' ? 'Needs attention' : 'Pending'}
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className={cn(
                      'rounded-md px-2 py-0.5 text-[10px] font-medium',
                      status === 'completed' && 'border-blue-300/30 bg-blue-400/10 text-blue-100',
                      status === 'running' && 'border-blue-300/30 bg-blue-400/10 text-blue-100',
                      status === 'failed' && 'border-red-300/30 bg-red-400/10 text-red-100',
                      status === 'skipped' && 'border-slate-400/30 bg-slate-400/10 text-slate-300',
                      status === 'pending' && 'border-slate-500/30 bg-slate-600/10 text-slate-400',
                    )}
                  >
                    {status}
                  </Badge>
                </div>
              );
            })}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
