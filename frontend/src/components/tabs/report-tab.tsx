'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, BrainCircuit, CheckCircle2, Clock3, Database, Download, FileCode2, FileText, FilePenLine, Loader2, MinusCircle, Sparkles, Target, TrendingUp, Upload } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/lib/store';
import { computeEdaStats } from '@/lib/eda';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { ReportConfig } from '@/types/forecast';

const STEP_TAB_MAP = {
  1: 'upload',
  2: 'understanding',
  3: 'eda',
  4: 'cleaning',
  5: 'forecast_ts',
  6: 'forecast_ml',
  7: 'loss_forecast',
  8: 'profit_forecast',
  9: 'ml',
  10: 'prediction',
} as const;

const DEFAULT_REPORT_CONFIG: ReportConfig = { includeLoss: true, includeProfit: true, scenario: 'baseline' };

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
    timeSeriesForecastResult, mlForecastResult, lossForecast, profitForecast, lossSegments, scenarios, breakevenPeriod, modelTrained,
    reportGenerated, reportUrl, setReportGenerated, setReportUrl, setActiveTab,
  } = store;

  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);
  const [generatingDocument, setGeneratingDocument] = useState(false);
  const [generatingHtml, setGeneratingHtml] = useState(false);
  const [generatingDocx, setGeneratingDocx] = useState(false);
  const [reportFileName, setReportFileName] = useState(() => buildReportFileName(fileName));
  const generatedTimestamp = useMemo(() => {
    const formatted = new Intl.DateTimeFormat('en-IN', {
      dateStyle: 'medium',
      timeStyle: 'short',
      timeZone: 'Asia/Kolkata',
    }).format(new Date());
    return `${formatted} IST`;
  }, []);
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
      title: 'Loss Forecast',
      icon: TrendingUp,
      status: lossForecast?.length ? 'Completed' : 'Skipped',
      detail: lossForecast?.length
        ? `${lossForecast.length} future periods were evaluated for revenue, operational, inventory, and discount loss pressure.`
        : 'Loss forecasting is optional and appears in the report after the Loss Forecast workflow has been executed.',
    },
    {
      step: 8,
      title: 'Profit Forecast',
      icon: TrendingUp,
      status: scenarios?.baseline?.length ? 'Completed' : 'Skipped',
      detail: scenarios?.baseline?.length
        ? `Optimistic, baseline, and pessimistic P&L scenarios were generated with break-even period ${breakevenPeriod ?? 'not reached'}.`
        : 'Profit forecasting is optional and appears in the report after the Profit Forecast workflow has been executed.',
    },
    {
      step: 9,
      title: 'ML Assistant',
      icon: BrainCircuit,
      status: modelTrained ? 'Completed' : 'Pending',
      detail: modelTrained
        ? `${selectedModel ?? 'A selected model'} was trained for ${problemType} on ${selectedFeatures.length} selected features${targetColumn ? ` targeting ${targetColumn}` : ''}.`
        : 'The supervised ML branch has not been trained yet.',
    },
    {
      step: 10,
      title: 'Prediction',
      icon: Target,
      status: predictionResult !== null ? 'Completed' : 'Pending',
      detail: predictionResult !== null
        ? `The latest inference result is ${String(predictionResult)} and will close the report as the final application outcome.`
        : 'Prediction output has not been generated yet, so the report will end with workflow context rather than a final inference result.',
    },
  ], [
    cleanedRowCount, cleaningDone, cleaningLogs.length, columns.length, edaStats.categoricalColumns.length, edaStats.correlations.length,
    edaStats.numericColumns.length, fileName, loadedRowCount, mlForecastResult, modelTrained, predictionResult, previewLoaded, problemType,
    breakevenPeriod, lossForecast, rawData, scenarios, selectedFeatures.length, selectedModel, targetColumn, timeSeriesForecastResult, totalRows,
  ]);

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
    lossForecast: lossForecast ?? [],
    profitForecast: scenarios?.[DEFAULT_REPORT_CONFIG.scenario] ?? profitForecast ?? [],
    lossSegments: lossSegments ?? [],
    scenarios,
    breakevenPeriod,
    reportConfig: DEFAULT_REPORT_CONFIG,
    forecastingStepsCompleted: [
      ...(timeSeriesForecastResult ? [5] : []),
      ...(mlForecastResult ? [6] : []),
      ...(lossForecast?.length ? [7] : []),
      ...(scenarios?.baseline?.length ? [8] : []),
    ],
    predictionResult,
    predictionAnalysis,
    predictionProbabilities,
    predictionHistory,
    edaStats,
  }), [
    aiInsights, cleanedData?.length, cleanedRowCount, cleaningDone, cleaningLogs, columns, datasetId, duplicates, edaStats, featureImportance,
    fileName, memoryUsage, mlForecastResult, modelMetrics, predictionAnalysis, predictionHistory, predictionProbabilities, predictionResult,
    loadedRowCount, lossForecast, lossSegments, previewLoaded, problemType, profitForecast, rawData?.length, scenarios, selectedFeatures,
    selectedModel, targetColumn, timeSeriesForecastResult, totalRows, uploadedModel, breakevenPeriod,
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

  const generateReport = useCallback(async (options?: { autoDownload?: boolean; format?: string }) => {
    const format = options?.format ?? 'pdf';
    if (format === 'pdf') setGenerating(true);
    try {
      const response = await apiClient.post('/report/generate', reportPayload, { params: { format }, responseType: 'blob' });
      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The report service returned an empty PDF.');
      }

      const contentType = response.headers['content-type'] ?? blob.type;
      const allowedTypes: string[] = format === 'pdf' ? ['application/pdf'] : format === 'html' ? ['text/html', 'application/octet-stream'] : ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/octet-stream'];
      if (typeof contentType === 'string' && !allowedTypes.some(t => contentType.includes(t))) {
        const errorText = await blob.text();
        throw new Error(errorText || 'The report service returned an unexpected response.');
      }

      const responseFileName = getDownloadFileName(response.headers['content-disposition'], buildReportFileName(fileName));
      const nextUrl = cacheGeneratedReport(blob, responseFileName);

      if (options?.autoDownload) {
        downloadBlobUrl(nextUrl, responseFileName);
        toast({ title: 'Report downloaded', description: `The ${format.toUpperCase()} workflow report was regenerated and downloaded successfully.` });
      } else {
        toast({ title: 'Report ready', description: `The ${format.toUpperCase()} workflow report has been generated and is ready to download.` });
      }
    } catch (error) {
      toast({ title: 'Generation failed', description: await getBlobErrorMessage(error, `Failed to generate the ${(options?.format ?? 'pdf').toUpperCase()} workflow report.`), variant: 'destructive' });
    } finally {
      if (format === 'pdf') setGenerating(false);
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

  const handleDownloadHtml = useCallback(async () => {
    setGeneratingHtml(true);
    try {
      const response = await apiClient.post('/report/generate', reportPayload, { params: { format: 'html' }, responseType: 'blob' });
      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The report service returned an empty HTML document.');
      }
      const responseFileName = getDownloadFileName(response.headers['content-disposition'], buildReportFileName(fileName, 'html'));
      const url = URL.createObjectURL(blob);
      downloadBlobUrl(url, responseFileName);
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      toast({ title: 'HTML report downloaded', description: 'The standalone HTML report has been downloaded.' });
    } catch (error) {
      toast({ title: 'HTML report failed', description: await getBlobErrorMessage(error, 'Failed to generate the HTML report.'), variant: 'destructive' });
    } finally {
      setGeneratingHtml(false);
    }
  }, [fileName, reportPayload, toast]);

  const handleDownloadDocx = useCallback(async () => {
    setGeneratingDocx(true);
    try {
      const response = await apiClient.post('/report/generate', reportPayload, { params: { format: 'docx' }, responseType: 'blob' });
      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The report service returned an empty DOCX document.');
      }
      const responseFileName = getDownloadFileName(response.headers['content-disposition'], buildReportFileName(fileName, 'docx'));
      const url = URL.createObjectURL(blob);
      downloadBlobUrl(url, responseFileName);
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      toast({ title: 'DOCX report downloaded', description: 'The editable Word document report has been downloaded.' });
    } catch (error) {
      toast({ title: 'DOCX report failed', description: await getBlobErrorMessage(error, 'Failed to generate the DOCX report.'), variant: 'destructive' });
    } finally {
      setGeneratingDocx(false);
    }
  }, [fileName, reportPayload, toast]);

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
      <div className="flex flex-col items-center justify-center py-20 text-center dark:bg-gray-900">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-blue-50 text-blue-600 dark:bg-gray-800 dark:text-blue-300">
          <FileText className="h-10 w-10" />
        </div>
        <h2 className="mt-6 text-xl font-bold text-gray-950 dark:text-gray-100">No Data Available</h2>
        <p className="mt-2 max-w-md text-sm text-gray-600 dark:text-gray-400">Upload a dataset and complete the workflow to generate the final report.</p>
        <Button onClick={() => setActiveTab('upload')} className="mt-6 gap-2">
          <Database className="h-4 w-4" />
          Go To Upload
        </Button>
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6 dark:bg-gray-900">
      <Card className="overflow-hidden border-blue-200 bg-white shadow-lg shadow-blue-950/10 dark:border-gray-700 dark:bg-gray-800 dark:shadow-black/30">
        <CardContent className="p-6 md:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-blue-100 bg-blue-50 p-2 text-blue-700 shadow-sm dark:border-blue-500/30 dark:bg-blue-950/50 dark:text-blue-300">
                <Bot className="h-8 w-8" />
              </div>
              <div className="min-w-0">
                <h1 className="text-2xl font-bold tracking-tight text-gray-950 md:text-3xl dark:text-gray-100">Intelligent Data Assistant - Analysis Report</h1>
                <div className="mt-3 flex flex-wrap gap-2 text-sm">
                  <Badge className="border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:border-blue-500/40 dark:bg-blue-950/50 dark:text-blue-200 dark:hover:bg-blue-900/60">{fileName ?? 'Untitled Dataset'}</Badge>
                  <Badge className="border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 dark:hover:bg-gray-900">Generated {generatedTimestamp}</Badge>
                  <Badge className="border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:border-emerald-500/40 dark:bg-emerald-950/40 dark:text-emerald-200 dark:hover:bg-emerald-900/50">{(cleanedRowCount ?? totalRows ?? rawData.length).toLocaleString()} rows</Badge>
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row lg:justify-end">
              <Button type="button" onClick={handleDownloadHtml} disabled={generatingHtml} className="gap-2" variant="outline">
                {generatingHtml ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileCode2 className="h-4 w-4" />}
                HTML
              </Button>
              <Button type="button" onClick={handleDownloadDocx} disabled={generatingDocx} className="gap-2" variant="outline">
                {generatingDocx ? <Loader2 className="h-4 w-4 animate-spin" /> : <FilePenLine className="h-4 w-4" />}
                DOCX
              </Button>
              <Button type="button" onClick={handleDownloadReport} className="gap-2 bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:text-white dark:hover:bg-blue-400">
                <Download className="h-4 w-4" />
                Download PDF
              </Button>
            </div>
          </div>

          <div className="mt-8 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-500/30 dark:bg-emerald-950/30">
              <div className="flex items-center gap-2 text-sm font-medium text-emerald-700 dark:text-emerald-200">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-300" />
                Completed
              </div>
              <p className="mt-2 text-3xl font-bold text-gray-950 dark:text-gray-100">{completedSteps}</p>
            </div>
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-500/30 dark:bg-amber-950/30">
              <div className="flex items-center gap-2 text-sm font-medium text-amber-700 dark:text-amber-200">
                <Clock3 className="h-4 w-4 text-amber-600 dark:text-amber-300" />
                Pending
              </div>
              <p className="mt-2 text-3xl font-bold text-gray-950 dark:text-gray-100">{pendingSteps}</p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-300">
                <MinusCircle className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                Skipped
              </div>
              <p className="mt-2 text-3xl font-bold text-gray-950 dark:text-gray-100">{optionalSteps}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
        <CardHeader>
          <CardTitle className="text-gray-950 dark:text-gray-100">Workflow Steps In Report</CardTitle>
          <CardDescription className="dark:text-gray-400">The PDF uses this application storyline so the report reads like the same guided process the user completed inside the product.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-gray-500 dark:text-gray-400">Completed</p>
              <p className="mt-2 text-2xl font-semibold text-blue-600 dark:text-blue-400">{completedSteps}</p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-gray-500 dark:text-gray-400">Pending</p>
              <p className="mt-2 text-2xl font-semibold text-gray-950 dark:text-gray-100">{pendingSteps}</p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
              <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-gray-500 dark:text-gray-400">Skipped</p>
              <p className="mt-2 text-2xl font-semibold text-gray-500 dark:text-gray-400">{optionalSteps}</p>
            </div>
          </div>

          {workflowSteps.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.step}
                className={`rounded-xl border border-l-4 p-4 transition duration-200 hover:-translate-y-0.5 hover:shadow-md dark:bg-gray-800 ${
                  step.status === 'Completed'
                    ? 'border-gray-200 border-l-emerald-500 bg-emerald-50/50 dark:border-gray-700 dark:border-l-emerald-400 dark:bg-gray-800'
                    : step.status === 'Skipped'
                    ? 'border-gray-200 border-l-gray-400 bg-gray-50 dark:border-gray-700 dark:border-l-gray-500 dark:bg-gray-800'
                    : 'border-gray-200 border-l-amber-500 bg-amber-50/40 dark:border-gray-700 dark:border-l-amber-400 dark:bg-gray-800'
                }`}
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="flex gap-4">
                    <div className={`flex h-11 w-11 items-center justify-center rounded-xl border ${step.status === 'Completed' ? 'border-emerald-200 bg-emerald-100 text-emerald-700 dark:border-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300' : step.status === 'Skipped' ? 'border-gray-200 bg-gray-100 text-gray-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-400' : 'border-amber-200 bg-amber-100 text-amber-700 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-300'}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="max-w-3xl">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-sm font-semibold text-gray-950 dark:text-gray-100">{step.title}</p>
                        <Badge variant={step.status === 'Completed' ? 'secondary' : 'outline'} className="dark:border-gray-700 dark:text-gray-100">{step.status}</Badge>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-gray-600 dark:text-gray-300">{step.detail}</p>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2 pl-[60px] lg:pl-0">
                    <Button variant="outline" size="sm" className="h-8 rounded-full border-blue-200 bg-blue-50 px-3 text-xs font-semibold text-blue-700 hover:bg-blue-100 dark:border-blue-500/40 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/60" onClick={() => setActiveTab(STEP_TAB_MAP[step.step as keyof typeof STEP_TAB_MAP])}>
                      View
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <div className="space-y-3">
        <Button type="button" onClick={handleGenerate} disabled={generating} size="lg" className="h-14 w-full gap-3 text-base font-semibold">
          {generating ? <Loader2 className="h-5 w-5 animate-spin" /> : <Sparkles className="h-5 w-5" />}
          {generating ? 'Generating Final PDF Report...' : reportGenerated ? 'Generate Fresh PDF Snapshot' : 'Generate Final PDF Report'}
        </Button>
        {reportGenerated && (
          <Card className="border-blue-200 bg-blue-50 dark:border-gray-700 dark:bg-gray-800">
            <CardContent className="flex flex-col gap-4 p-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm font-semibold text-blue-700 dark:text-blue-300">Report ready for export</p>
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                  Download the polished presentation-style PDF for distribution, or export the editable Word-compatible document for revision and stakeholder tailoring.
                </p>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Button type="button" onClick={handleDownloadReport} variant="outline" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Download PDF
                </Button>
                <Button type="button" onClick={handleDownloadHtml} variant="outline" disabled={generatingHtml} className="gap-2">
                  {generatingHtml ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileCode2 className="h-4 w-4" />}
                  Download HTML
                </Button>
                <Button type="button" onClick={handleDownloadDocx} variant="outline" disabled={generatingDocx} className="gap-2">
                  {generatingDocx ? <Loader2 className="h-4 w-4 animate-spin" /> : <FilePenLine className="h-4 w-4" />}
                  Download DOCX
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
