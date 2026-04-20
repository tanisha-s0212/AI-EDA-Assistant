'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, BrainCircuit, Database, FileText, FilePenLine, Loader2, Sparkles, Target, TrendingUp, Upload } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/lib/store';
import { computeEdaStats } from '@/lib/eda';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';

const STEP_TAB_MAP = {
  1: 'upload',
  2: 'understanding',
  3: 'eda',
  4: 'cleaning',
  5: 'forecast_ts',
  6: 'forecast_ml',
  7: 'ml',
  8: 'prediction',
} as const;

function buildReportFileName(fileName: string | null, extension = 'pdf'): string {
  const baseName = (fileName ?? 'dataset')
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9-_ ]+/g, '')
    .trim()
    .replace(/\s+/g, '_');
  const stamp = new Date().toISOString().slice(0, 10);
  return `${baseName || 'dataset'}_workflow_report_${stamp}.${extension}`;
}

function sanitizeForJson<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeForJson(item)) as T;
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .map(([key, item]) => [key, sanitizeForJson(item)])
    ) as T;
  }
  if (typeof value === 'number' && !Number.isFinite(value)) {
    return null as T;
  }
  return value;
}

function normalizeForecastTrainingSummary<T extends object | null | undefined>(summary: T) {
  if (!summary || typeof summary !== 'object') return summary;
  return {
    lag_periods: 0,
    ...summary,
  } as T;
}

async function getBlobErrorMessage(error: unknown, fallback: string): Promise<string> {
  const message = getApiErrorMessage(error, fallback);
  if (!(error instanceof Error) && typeof error !== 'object') {
    return message;
  }

  const maybeBlob = (error as { response?: { data?: unknown } }).response?.data;
  if (!(maybeBlob instanceof Blob)) {
    return message;
  }

  try {
    const text = await maybeBlob.text();
    const parsed = JSON.parse(text) as { detail?: string; error?: string };
    return parsed.detail || parsed.error || message;
  } catch {
    return message;
  }
}

function getDownloadFileName(contentDisposition: string | undefined, fallback: string) {
  if (!contentDisposition) return fallback;
  const utfMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utfMatch?.[1]) {
    return decodeURIComponent(utfMatch[1]);
  }
  const asciiMatch = contentDisposition.match(/filename="?([^"]+)"?/i);
  return asciiMatch?.[1] ?? fallback;
}

function downloadBlobUrl(blobUrl: string, fileName: string) {
  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = fileName;
  anchor.rel = 'noopener';
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
}

export default function ReportTab() {
  const store = useAppStore();
  const {
    rawData, cleanedData, fileName, datasetId, columns, totalRows, duplicates, memoryUsage,
    previewLoaded, loadedRowCount,
    cleaningLogs, cleaningDone, cleanedRowCount, aiInsights,
    targetColumn, problemType, selectedFeatures, selectedModel, modelMetrics, featureImportance,
    uploadedModel, predictionResult, predictionAnalysis, predictionProbabilities, predictionHistory,
    timeSeriesForecastResult, mlForecastResult, modelTrained,
    reportGenerated, reportUrl, setReportGenerated, setReportUrl, setActiveTab,
  } = store;

  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);
  const [generatingDocument, setGeneratingDocument] = useState(false);
  const [reportFileName, setReportFileName] = useState(() => buildReportFileName(fileName));
  const analysisData = cleanedData ?? rawData ?? [];
  const edaStats = useMemo(() => computeEdaStats(analysisData, columns), [analysisData, columns]);

  useEffect(() => {
    setReportFileName(buildReportFileName(fileName));
  }, [fileName]);

  const workflowSteps = useMemo(() => [
    {
      step: 1,
      title: 'Upload',
      icon: Upload,
      status: rawData ? 'Completed' : 'Pending',
      detail: rawData
        ? previewLoaded && (totalRows || rawData.length) > loadedRowCount
          ? `${fileName ?? 'Dataset'} entered the application with ${(totalRows || rawData.length).toLocaleString()} total rows, while ${loadedRowCount.toLocaleString()} preview rows were loaded in-browser and the full dataset stayed cached on the backend for downstream work.`
          : `${fileName ?? 'Dataset'} entered the application with ${(totalRows || rawData.length).toLocaleString()} rows available directly in the workspace for downstream work.`
        : 'The report begins only after a dataset has been uploaded into the workflow.',
    },
    {
      step: 2,
      title: 'Data Understanding',
      icon: Database,
      status: columns.length > 0 ? 'Completed' : 'Pending',
      detail: columns.length > 0
        ? `${columns.length} columns were profiled to understand data types, null behavior, cardinality, and date or numeric roles before transformation.`
        : 'Column profiling has not been captured yet.',
    },
    {
      step: 3,
      title: 'Exploratory Data Analysis',
      icon: Database,
      status: columns.length > 0 ? 'Completed' : 'Pending',
      detail: columns.length > 0
        ? `Exploratory analysis summarized ${edaStats.numericColumns.length} numeric columns, ${edaStats.categoricalColumns.length} categorical columns, and ${edaStats.correlations.length} correlation signals.`
        : 'Exploratory data analysis output is not available yet.',
    },
    {
      step: 4,
      title: 'Data Cleaning',
      icon: Sparkles,
      status: cleaningDone ? 'Completed' : 'Pending',
      detail: cleaningDone
        ? `${cleaningLogs.length} cleaning actions were recorded and the cleaned dataset retained ${(cleanedRowCount ?? rawData?.length ?? 0).toLocaleString()} rows for analysis.`
        : 'Data cleaning follows exploratory data analysis and should be completed before final reporting when data corrections are needed.',
    },
    {
      step: 5,
      title: 'Forecast TS',
      icon: TrendingUp,
      status: timeSeriesForecastResult ? 'Completed' : 'Skipped',
      detail: timeSeriesForecastResult
        ? `${timeSeriesForecastResult.training_summary.model_name} generated a ${timeSeriesForecastResult.training_summary.forecast_periods}-period time-series forecast with backtest metrics and interval estimates.`
        : 'Time-series forecasting is optional and will only appear in the PDF if this tab was executed.',
    },
    {
      step: 6,
      title: 'Forecast ML',
      icon: TrendingUp,
      status: mlForecastResult ? 'Completed' : 'Skipped',
      detail: mlForecastResult
        ? `${mlForecastResult.training_summary.model_name} produced an ML forecast using ${mlForecastResult.generated_features.length} engineered features and explainability outputs.`
        : 'ML forecasting is optional and will only appear in the PDF if this tab was executed.',
    },
    {
      step: 7,
      title: 'ML Assistant',
      icon: BrainCircuit,
      status: modelTrained ? 'Completed' : 'Pending',
      detail: modelTrained
        ? `${selectedModel ?? 'A selected model'} was trained for ${problemType} on ${selectedFeatures.length} selected features${targetColumn ? ` targeting ${targetColumn}` : ''}.`
        : 'The supervised ML branch has not been trained yet.',
    },
    {
      step: 8,
      title: 'Prediction',
      icon: Target,
      status: predictionResult !== null ? 'Completed' : 'Pending',
      detail: predictionResult !== null
        ? `The latest inference result is ${String(predictionResult)} and will close the report as the final application outcome.`
        : 'Prediction output has not been generated yet, so the report will end with workflow context rather than a final inference result.',
    },
  ], [
    cleanedRowCount, cleaningDone, cleaningLogs.length, columns.length, edaStats.categoricalColumns.length, edaStats.correlations.length,
    edaStats.numericColumns.length, fileName, mlForecastResult, modelTrained, predictionResult, problemType, rawData, selectedFeatures.length,
    selectedModel, targetColumn, timeSeriesForecastResult, totalRows,
  ]);

  const reportNarrative = useMemo(() => {
    const sections = [
      'The PDF report compiles the application journey from ingestion through final prediction so the document mirrors how the user moved across the product.',
      previewLoaded && (totalRows || rawData?.length || 0) > loadedRowCount
        ? `Because this dataset was loaded as a responsive preview in the browser, the report will call out that ${loadedRowCount.toLocaleString()} rows were shown interactively while the backend kept the full ${(totalRows || rawData?.length || 0).toLocaleString()}-row dataset available for cleaning, forecasting, and training.`
        : 'This dataset was fully loaded in the active workspace, so the report can describe the in-app view and backend processing scope as the same dataset slice.',
      cleaningDone
        ? 'Because cleaning was completed, the report can anchor all later steps to the cleaned cached dataset rather than the original raw preview.'
        : 'Cleaning has not been completed, so the report may contain a less stable workflow narrative.',
      timeSeriesForecastResult || mlForecastResult
        ? 'Forecasting sections will be included conditionally based on which forecasting tabs were run in this session.'
        : 'Forecasting sections will be omitted because no forecast results are currently available.',
      predictionResult !== null
        ? 'The report will close with the final prediction result and supporting model context.'
        : 'The report will still generate without a prediction, but the final outcome section will be lighter.',
    ];
    return sections;
  }, [cleaningDone, mlForecastResult, predictionResult, timeSeriesForecastResult]);

  const completedSteps = workflowSteps.filter((step) => step.status === 'Completed').length;
  const pendingSteps = workflowSteps.filter((step) => step.status === 'Pending').length;
  const optionalSteps = workflowSteps.filter((step) => step.status === 'Skipped').length;
  const reportPayload = useMemo(() => sanitizeForJson({
    datasetId: datasetId ?? null,
    sessionId: datasetId ?? null,
    fileName: fileName ?? 'Untitled Dataset',
    totalRows: totalRows || (rawData?.length ?? 0),
    previewLoaded,
    loadedRowCount,
    columns: columns.map((column) => ({
      name: column.name,
      dtype: column.dtype,
      nonNull: column.nonNull,
      nullCount: column.nullCount,
      uniqueCount: column.uniqueCount,
      role: column.role,
    })),
    duplicates: duplicates ?? 0,
    memoryUsage: memoryUsage || 'N/A',
    cleaningLogs,
    cleaningDone,
    cleanedRowCount: cleanedRowCount ?? cleanedData?.length ?? (rawData?.length ?? 0),
    targetColumn,
    problemType,
    selectedFeatures,
    selectedModel,
    modelMetrics,
    featureImportance: featureImportance ?? [],
    aiInsights,
    uploadedModel,
    timeSeriesForecastResult: timeSeriesForecastResult ? {
      ...timeSeriesForecastResult,
      training_summary: normalizeForecastTrainingSummary(timeSeriesForecastResult.training_summary),
    } : null,
    mlForecastResult: mlForecastResult ? {
      ...mlForecastResult,
      training_summary: normalizeForecastTrainingSummary(mlForecastResult.training_summary),
    } : null,
    forecastingStepsCompleted: [
      ...(timeSeriesForecastResult ? [5] : []),
      ...(mlForecastResult ? [6] : []),
    ],
    predictionResult,
    predictionAnalysis,
    predictionProbabilities,
    predictionHistory,
    edaStats,
  }), [
    aiInsights, cleanedData?.length, cleanedRowCount, cleaningDone, cleaningLogs, columns, datasetId, duplicates, edaStats, featureImportance,
    fileName, memoryUsage, mlForecastResult, modelMetrics, predictionAnalysis, predictionHistory, predictionProbabilities, predictionResult,
    loadedRowCount, previewLoaded, problemType, rawData?.length, selectedFeatures, selectedModel, targetColumn, timeSeriesForecastResult, totalRows, uploadedModel,
  ]);

  const cacheGeneratedReport = useCallback((blob: Blob, nextFileName: string) => {
    if (reportUrl) {
      URL.revokeObjectURL(reportUrl);
    }
    const nextUrl = URL.createObjectURL(blob);
    setReportUrl(nextUrl);
    setReportFileName(nextFileName);
    setReportGenerated(true);
    return nextUrl;
  }, [reportUrl, setReportGenerated, setReportUrl]);

  const generateReport = useCallback(async (options?: { autoDownload?: boolean }) => {
    setGenerating(true);
    try {
      const response = await apiClient.post('/report/generate', reportPayload, { params: { format: 'pdf' }, responseType: 'blob' });
      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The report service returned an empty PDF.');
      }

      const contentType = response.headers['content-type'] ?? blob.type;
      if (typeof contentType === 'string' && !contentType.includes('application/pdf')) {
        const errorText = await blob.text();
        throw new Error(errorText || 'The report service returned an unexpected response.');
      }

      const responseFileName = getDownloadFileName(response.headers['content-disposition'], buildReportFileName(fileName));
      const nextUrl = cacheGeneratedReport(blob, responseFileName);

      if (options?.autoDownload) {
        downloadBlobUrl(nextUrl, responseFileName);
        toast({ title: 'Report downloaded', description: 'The PDF workflow report was regenerated and downloaded successfully.' });
      } else {
        toast({ title: 'Report ready', description: 'The PDF workflow report has been generated and is ready to download.' });
      }
    } catch (error) {
      toast({ title: 'Generation failed', description: await getBlobErrorMessage(error, 'Failed to generate the workflow report.'), variant: 'destructive' });
    } finally {
      setGenerating(false);
    }
  }, [
    cacheGeneratedReport, fileName, reportPayload, toast,
  ]);

  const handleGenerate = useCallback(() => {
    void generateReport();
  }, [generateReport]);

  const handleDownloadReport = useCallback(() => {
    if (!reportUrl) {
      toast({ title: 'Generate report first', description: 'Create the PDF once, then use Download Report to save it locally.', variant: 'destructive' });
      return;
    }
    downloadBlobUrl(reportUrl, reportFileName);
    toast({ title: 'Download started', description: 'The generated PDF is being downloaded.' });
  }, [reportFileName, reportUrl, toast]);

  const handleRegenerateReport = useCallback(() => {
    void generateReport({ autoDownload: true });
  }, [generateReport]);

  const handleDownloadDocument = useCallback(async () => {
    setGeneratingDocument(true);
    try {
      const response = await apiClient.post('/report/generate', reportPayload, { params: { format: 'doc' }, responseType: 'blob' });
      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The report service returned an empty document.');
      }
      const responseFileName = getDownloadFileName(response.headers['content-disposition'], buildReportFileName(fileName, 'doc'));
      const url = URL.createObjectURL(blob);
      downloadBlobUrl(url, responseFileName);
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      toast({ title: 'Document downloaded', description: 'The editable workflow document has been downloaded.' });
    } catch (error) {
      toast({ title: 'Document failed', description: await getBlobErrorMessage(error, 'Failed to generate the editable report document.'), variant: 'destructive' });
    } finally {
      setGeneratingDocument(false);
    }
  }, [fileName, reportPayload, toast]);

  if (!rawData) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <FileText className="h-10 w-10" />
        </div>
        <h2 className="mt-6 text-xl font-bold">No Data Available</h2>
        <p className="mt-2 max-w-md text-sm text-muted-foreground">Upload a dataset and complete the workflow to generate the final report.</p>
        <Button onClick={() => setActiveTab('upload')} className="mt-6 gap-2">
          <Database className="h-4 w-4" />
          Go To Upload
        </Button>
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <Card className="overflow-hidden border-primary/20 bg-primary text-primary-foreground">
        <CardContent className="p-8">
          <div className="flex items-start gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary-foreground/10 p-2 text-primary-foreground shadow-lg shadow-black/10">
              <Bot className="h-8 w-8" />
            </div>
            <div className="max-w-4xl">
              <h1 className="text-2xl font-bold tracking-tight">Generate Final Workflow Report</h1>
              <p className="mt-2 text-sm text-primary-foreground/80">
                This tab is now focused on one job: generate presentation-ready workflow reports that package the complete application process from data upload through data understanding, exploratory data analysis, data cleaning, forecasting, ML training, and final prediction.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Badge className="border-primary-foreground/15 bg-primary-foreground/10 text-primary-foreground">{fileName ?? 'Untitled Dataset'}</Badge>
                <Badge className="border-primary-foreground/15 bg-primary-foreground/10 text-primary-foreground">{completedSteps}/8 steps completed</Badge>
                <Badge className="border-primary-foreground/15 bg-primary-foreground/10 text-primary-foreground">{(cleanedRowCount ?? totalRows ?? rawData.length).toLocaleString()} rows in final flow</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Workflow Steps In Report</CardTitle>
          <CardDescription>The PDF uses this application storyline so the report reads like the same guided process the user completed inside the product.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-border bg-card p-4">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Completed</p>
              <p className="mt-2 text-2xl font-semibold text-primary">{completedSteps}</p>
            </div>
            <div className="rounded-2xl border border-border bg-card p-4">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Pending</p>
              <p className="mt-2 text-2xl font-semibold text-foreground">{pendingSteps}</p>
            </div>
            <div className="rounded-2xl border border-border bg-card p-4">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Optional Skipped</p>
              <p className="mt-2 text-2xl font-semibold text-muted-foreground">{optionalSteps}</p>
            </div>
          </div>

          {workflowSteps.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.step}
                className={`rounded-2xl border p-4 transition-colors ${
                  step.status === 'Completed'
                    ? 'border-primary/20 bg-primary/5'
                    : step.status === 'Skipped'
                    ? 'border-border bg-secondary/40'
                    : 'bg-card'
                }`}
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="flex gap-4">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl border bg-card shadow-sm">
                      <span className="text-sm font-semibold text-muted-foreground">{step.step}</span>
                    </div>
                    <div className={`flex h-11 w-11 items-center justify-center rounded-xl border ${step.status === 'Completed' ? 'border-primary/20 bg-primary/10 text-primary' : step.status === 'Skipped' ? 'border-border bg-secondary text-secondary-foreground' : 'border-border bg-muted/30 text-muted-foreground'}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="max-w-3xl">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-sm font-semibold">{step.title}</p>
                        <Badge variant={step.status === 'Completed' ? 'secondary' : 'outline'}>{step.status}</Badge>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-muted-foreground">{step.detail}</p>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2 pl-[62px] lg:pl-0">
                    <Button variant="ghost" size="sm" className="rounded-xl px-4" onClick={() => setActiveTab(STEP_TAB_MAP[step.step as keyof typeof STEP_TAB_MAP])}>
                      Open Step
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>What The PDF Will Capture</CardTitle>
          <CardDescription>This final export is designed to read like an executive presentation deck, not just a raw tab dump.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {reportNarrative.map((line) => (
            <div key={line} className="rounded-xl border border-border bg-muted/20 p-4 text-sm leading-6 text-muted-foreground">
              {line}
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="space-y-3">
        <Button type="button" onClick={handleGenerate} disabled={generating} size="lg" className="h-14 w-full gap-3 text-base font-semibold">
          {generating ? <Loader2 className="h-5 w-5 animate-spin" /> : <Sparkles className="h-5 w-5" />}
          {generating ? 'Generating Final PDF Report...' : reportGenerated ? 'Generate Fresh PDF Snapshot' : 'Generate Final PDF Report'}
        </Button>
        {reportGenerated && (
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="flex flex-col gap-4 p-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm font-semibold text-primary">Report ready for export</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Download the polished presentation-style PDF for distribution, or export the editable Word-compatible document for revision and stakeholder tailoring.
                </p>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Button type="button" onClick={handleDownloadReport} variant="outline" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Download PDF
                </Button>
                <Button type="button" onClick={handleDownloadDocument} variant="outline" disabled={generatingDocument} className="gap-2">
                  {generatingDocument ? <Loader2 className="h-4 w-4 animate-spin" /> : <FilePenLine className="h-4 w-4" />}
                  Download Document
                </Button>
                <Button type="button" onClick={handleRegenerateReport} disabled={generating} className="gap-2">
                  {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  Regenerate PDF
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </motion.div>
  );
}
