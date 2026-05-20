'use client';

import React from 'react';
import axios from 'axios';
import { AlertCircle, Check, Download, Loader2, Play, SkipForward } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
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
      await state.runLossForecast(state.datasetId, 30);
      return 'Loss Forecast completed and stored in the application workflow.';
    }

    if (stepName === 'Profit Forecast') {
      if (!state.datasetId) throw new Error('Profit Forecast needs a cached dataset id.');
      await state.runProfitForecast(state.datasetId, 30);
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
    <section className="mb-5 rounded-xl border border-white/72 bg-white/78 p-4 shadow-[0_18px_50px_-38px_rgba(31,95,168,0.34)] ring-1 ring-white/54 backdrop-blur-xl dark:border-white/10 dark:bg-white/8 dark:ring-white/8">
      {banner && (
        <Alert className="mb-4 border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-100">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{banner}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div className="min-w-0 space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Agent Workspace</p>
              <h2 className="mt-1 truncate text-lg font-semibold tracking-normal text-foreground">
                {fileName ?? 'No dataset selected'}
              </h2>
            </div>
            <Button size="sm" onClick={suggestNextSteps} disabled={!datasetId || isSuggesting || health?.agentic_enabled === false} className="rounded-sm">
              {isSuggesting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              Suggest Next Steps
            </Button>
          </div>

          {activeRecommendation ? (
            <Card className="gap-4 rounded-lg border-blue-100 bg-blue-50/70 py-4 shadow-none dark:border-cyan-400/20 dark:bg-cyan-400/10">
              <CardHeader className="px-4">
                <div className="flex items-start justify-between gap-3">
                  <CardTitle className="text-base">{activeRecommendation.step}</CardTitle>
                  <Badge variant="outline" className="rounded-sm bg-white/70 dark:bg-white/8">Recommended</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3 px-4">
                <p className="text-sm text-muted-foreground">{activeRecommendation.reason}</p>
                <div className="space-y-1">
                  {activeRecommendation.findings.map((finding) => (
                    <div key={finding} className="flex gap-2 text-sm text-foreground">
                      <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                      <span>{finding}</span>
                    </div>
                  ))}
                </div>
                {lastSummary && <p className="rounded-md border bg-white/70 p-3 text-sm text-muted-foreground dark:bg-white/8">{lastSummary}</p>}
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" onClick={acceptRecommendation} disabled={Boolean(runningStep)} className="rounded-sm">
                    {runningStep ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : activeRecommendation.step === 'Report Generation' ? <Download className="mr-2 h-4 w-4" /> : <Check className="mr-2 h-4 w-4" />}
                    Accept & Continue
                  </Button>
                  <Button size="sm" variant="outline" onClick={skipRecommendation} disabled={Boolean(runningStep)} className="rounded-sm bg-white/70 dark:bg-white/8">
                    <SkipForward className="mr-2 h-4 w-4" />
                    Skip
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-300 bg-white/54 p-4 text-sm text-muted-foreground dark:border-white/12 dark:bg-white/5">
              {workflowComplete && agenticSessionId ? (
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <span>The approved agentic flow is complete. Download the consolidated run report when you are ready.</span>
                  <Button size="sm" onClick={() => void downloadReport(agenticSessionId)} className="rounded-sm">
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

        <aside className="rounded-lg border border-slate-200 bg-white/70 p-4 dark:border-white/10 dark:bg-white/5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Pipeline</p>
            <span className="text-xs font-semibold text-muted-foreground">{progress}%</span>
          </div>
          <Progress value={progress} className="mb-3 h-1.5" />
          <div className="space-y-2">
            {PIPELINE_STEPS.map((step) => {
              const status = agenticStepStatuses[step] ?? 'pending';
              return (
                <div key={step} className="flex items-center justify-between gap-3 text-sm">
                  <span className="truncate">{step}</span>
                  <Badge
                    variant="outline"
                    className={cn(
                      'rounded-sm px-2 py-0.5 text-[10px]',
                      status === 'completed' && 'border-emerald-200 bg-emerald-50 text-emerald-700',
                      status === 'running' && 'border-blue-200 bg-blue-50 text-blue-700',
                      status === 'failed' && 'border-red-200 bg-red-50 text-red-700',
                      status === 'skipped' && 'border-slate-200 bg-slate-50 text-slate-600'
                    )}
                  >
                    {status}
                  </Badge>
                </div>
              );
            })}
          </div>
        </aside>
      </div>
    </section>
  );
}
