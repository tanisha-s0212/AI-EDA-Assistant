'use client';

import React from 'react';
import { AlertCircle, Check, Download, Loader2, Play, SkipForward } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import type { AgenticStepStatus, Recommendation } from '@/lib/store';
import { cn } from '@/lib/utils';

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

  const activeRecommendation = agenticRecommendations[0] ?? null;
  const progress = statusProgress(agenticStepStatuses);

  const refreshHealth = React.useCallback(async () => {
    try {
      const response = await apiClient.get<AgenticHealth>('/agentic/health');
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
        const response = await apiClient.get<StatusResponse>(`/agentic/session/${agenticSessionId}/status`);
        Object.entries(response.data.steps ?? {}).forEach(([step, status]) => setAgenticStepStatus(step, status));
        if (response.data.recommendations) {
          setAgenticRecommendations(response.data.recommendations);
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

  const suggestNextSteps = async () => {
    if (!datasetId) return;
    setIsSuggesting(true);
    setBanner(null);
    try {
      const response = await apiClient.post<SuggestResponse>('/agentic/suggest-next-steps', {
        dataset_path: datasetId,
      });
      setAgenticSessionId(response.data.session_id);
      setAgenticRecommendations(response.data.recommendations);
      PIPELINE_STEPS.forEach((step) => setAgenticStepStatus(step, 'pending'));
      setAgenticLastSyncedAt(Date.now());
    } catch (error) {
      setBanner(getApiErrorMessage(error, 'Agentic layer unavailable — manual mode active'));
    } finally {
      setIsSuggesting(false);
    }
  };

  const downloadReport = async (sessionId: string) => {
    const response = await apiClient.get(`/agentic/session/${sessionId}/report`, { responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agentic_run_${sessionId}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const acceptRecommendation = async () => {
    if (!activeRecommendation || !agenticSessionId) return;
    if (activeRecommendation.step === 'Report Generation') {
      await downloadReport(agenticSessionId);
      setAgenticStepStatus(activeRecommendation.step, 'completed');
      setAgenticRecommendations([]);
      return;
    }

    setRunningStep(activeRecommendation.step);
    setAgenticStepStatus(activeRecommendation.step, 'running');
    try {
      const response = await apiClient.post<ExecuteResponse>('/agentic/execute-step', {
        session_id: agenticSessionId,
        step_name: activeRecommendation.step,
        approved_by: 'current_user',
      });
      const nextStatus = response.data.status === 'not_yet_wired' ? 'failed' : response.data.status;
      setAgenticStepStatus(activeRecommendation.step, nextStatus);
      setLastSummary(response.data.output_summary ?? response.data.error ?? null);
      setAgenticRecommendations(response.data.next_recommendations ?? []);
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
      const response = await apiClient.post<{ next_recommendations?: Recommendation[] }>('/agentic/decision', {
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
                    Accept
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
              Upload or select a dataset, then ask the agent for the next approved step.
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
