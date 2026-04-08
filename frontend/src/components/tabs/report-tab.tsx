'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import {
  FileText, CheckCircle2, Circle, Download, Loader2, Sparkles, BarChart3, Database,
  Droplets, BrainCircuit, Target, TrendingUp, ChevronDown, AlertTriangle, Shield,
  Table2, Zap, ArrowRight, Award, Eye, EyeOff, RefreshCw,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { computeEdaStats } from '@/lib/eda';
import { useAppStore } from '@/lib/store';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';

// ──────────────────────────────────────────────────────────
// Animation variants
// ──────────────────────────────────────────────────────────
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
    transition: { type: 'spring', stiffness: 300, damping: 24 },
  },
};

function PreviewSection({
  icon: Icon,
  title,
  description,
  hasData,
  defaultOpen = false,
  children,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  hasData: boolean;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen}>
      <Card>
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="flex w-full items-center gap-3 px-6 py-4 text-left transition-colors hover:bg-muted/30"
          >
            <div
              className={cn(
                'flex h-9 w-9 items-center justify-center rounded-lg border',
                hasData
                  ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                  : 'border-border bg-muted/50 text-muted-foreground',
              )}
            >
              <Icon className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{title}</p>
              <p className="text-xs text-muted-foreground">{description}</p>
            </div>
            <Badge variant={hasData ? 'secondary' : 'outline'} className="shrink-0">
              {hasData ? 'Ready' : 'Partial'}
            </Badge>
            <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform data-[state=open]:rotate-180" />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <Separator />
          <CardContent className="pt-4">{children}</CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}


function buildReportFileName(fileName: string | null): string {
  const baseName = (fileName ?? 'dataset')
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9-_ ]+/g, '')
    .trim()
    .replace(/\s+/g, '_');
  const stamp = new Date().toISOString().slice(0, 10);
  return `${baseName || 'dataset'}_workflow_report_${stamp}.pdf`;
}

function ChecklistItem({
  icon: Icon,
  label,
  complete,
  delay = 0,
}: {
  icon: React.ElementType;
  label: string;
  complete: boolean;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className={cn(
        'flex items-center gap-3 rounded-xl border p-3 transition-colors',
        complete
          ? 'border-emerald-500/20 bg-gradient-to-r from-emerald-500/8 to-teal-500/8'
          : 'border-border bg-card',
      )}
    >
      <div className={cn(
        'flex h-9 w-9 items-center justify-center rounded-lg border',
        complete
          ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
          : 'border-border bg-muted/50 text-muted-foreground',
      )}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-[11px] text-muted-foreground">
          {complete ? 'Included in the report flow' : 'Complete this step to enrich the report'}
        </p>
      </div>
      {complete ? (
        <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
      ) : (
        <Circle className="h-4 w-4 text-muted-foreground shrink-0" />
      )}
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Main Report Tab
// ──────────────────────────────────────────────────────────
export default function ReportTab() {
  const {
    rawData, fileName, columns, totalRows, duplicates, memoryUsage,
    cleaningLogs, cleaningDone, cleanedData, cleanedRowCount,
    aiInsights,
    targetColumn, problemType, selectedFeatures, selectedModel, modelMetrics, modelTrained, featureImportance,
    salesForecastResult, uploadedModel, predictionResult, predictionAnalysis, predictionProbabilities, predictionHistory,
    reportGenerated, reportUrl, setReportGenerated, setReportUrl,
    setActiveTab,
  } = useAppStore();

  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);

  const analysisData = cleanedData ?? rawData ?? [];

  // Compute EDA stats for the payload
  const edaStats = useMemo(
    () => computeEdaStats(analysisData, columns),
    [analysisData, columns],
  );

  // Checklist items
  const checklist = useMemo(() => [
    { icon: UploadIcon, label: 'Data Uploaded', complete: !!rawData },
    { icon: Database, label: 'Data Understood', complete: columns.length > 0 },
    { icon: Droplets, label: 'Data Cleaned', complete: cleaningDone },
    { icon: BrainCircuit, label: 'EDA Performed', complete: edaStats.numericColumns.length > 0 || edaStats.correlations.length > 0 },
    { icon: TrendingUp, label: 'Sales Forecast Run', complete: !!salesForecastResult },
    { icon: Target, label: 'Model Trained', complete: modelTrained },
    { icon: TrendingUp, label: 'Prediction Made', complete: predictionResult !== null },
  ] as const, [rawData, columns.length, cleaningDone, edaStats.numericColumns.length, edaStats.correlations.length, salesForecastResult, modelTrained, predictionResult]);

  const completedCount = checklist.filter((c) => c.complete).length;
  const completionPercent = Math.round((completedCount / checklist.length) * 100);

  // Generate PDF
  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    try {
      const payload = {
        fileName: fileName ?? 'Untitled Dataset',
        totalRows: totalRows || (rawData?.length ?? 0),
        columns: columns.map((c) => ({
          name: c.name,
          dtype: c.dtype,
          nonNull: c.nonNull,
          nullCount: c.nullCount,
          uniqueCount: c.uniqueCount,
          role: c.role,
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
        predictionResult,
        predictionAnalysis,
        predictionProbabilities,
        predictionHistory,
        aiInsights,
        uploadedModel,
        salesForecastResult,
        edaStats,
      };

      const response = await apiClient.post('/generate-report', payload, {
        responseType: 'blob',
      });

      const blob = response.data as Blob;
      const url = URL.createObjectURL(blob);
      setReportUrl(url);
      setReportGenerated(true);

      // Trigger download
      const a = document.createElement('a');
      a.href = url;
      a.download = buildReportFileName(fileName);
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      toast({
        title: 'Report generated successfully',
        description: 'Your PDF report has been downloaded.',
      });
    } catch (err) {
      console.error(err);
      toast({
        title: 'Generation failed',
        description: getApiErrorMessage(err, 'Something went wrong while generating the report. Please try again.'),
        variant: 'destructive',
      });
    } finally {
      setGenerating(false);
    }
  }, [
    fileName, totalRows, rawData, columns, duplicates, memoryUsage,
    cleaningLogs, cleaningDone, cleanedData, cleanedRowCount, targetColumn, problemType,
    selectedFeatures, selectedModel, modelMetrics, featureImportance,
    predictionResult, predictionAnalysis, predictionProbabilities, predictionHistory, aiInsights, uploadedModel, salesForecastResult, edaStats,
    setReportUrl, setReportGenerated, toast,
  ]);

  const handleDownloadAgain = useCallback(() => {
    if (!reportUrl) return;
    const a = document.createElement('a');
    a.href = reportUrl;
    a.download = buildReportFileName(fileName);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [reportUrl, fileName]);

  const handleRegenerate = useCallback(() => {
    setReportGenerated(false);
    setReportUrl(null);
    if (reportUrl) URL.revokeObjectURL(reportUrl);
  }, [reportUrl, setReportGenerated, setReportUrl]);

  // Empty state
  if (!rawData) {
    return (
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="flex flex-col items-center justify-center py-20"
      >
        <motion.div
          variants={itemVariants}
          className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 text-emerald-500"
        >
          <FileText className="h-10 w-10" />
        </motion.div>
        <motion.h2 variants={itemVariants} className="mt-6 text-xl font-bold text-foreground">
          No Data Available
        </motion.h2>
        <motion.p variants={itemVariants} className="mt-2 text-sm text-muted-foreground max-w-md text-center">
          Upload a dataset and complete the analysis workflow to generate a comprehensive report.
        </motion.p>
        <motion.div variants={itemVariants}>
          <Button
            onClick={() => setActiveTab('upload')}
            className="mt-6 gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600 shadow-lg shadow-emerald-500/25"
          >
            <Database className="h-4 w-4" />
            Go to Upload
            <ArrowRight className="h-4 w-4" />
          </Button>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      {/* ─── Hero Header ─── */}
      <motion.div variants={itemVariants} className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-600 via-teal-600 to-emerald-700 p-8 text-white">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
        <div className="absolute top-4 right-12 w-24 h-24 bg-white/5 rounded-lg rotate-12" />

        <div className="relative flex items-start gap-5">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 260, damping: 20, delay: 0.1 }}
            className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-white/15 backdrop-blur-sm shadow-lg"
          >
            <FileText className="h-7 w-7" />
          </motion.div>
          <div className="flex-1 min-w-0">
            <motion.h1
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="text-2xl sm:text-3xl font-bold tracking-tight"
            >
              Generate Report
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="mt-2 text-sm sm:text-base text-emerald-100 max-w-lg"
            >
              Create a comprehensive PDF analysis report covering your entire data science workflow — from data upload through predictions.
            </motion.p>
          </div>
        </div>
      </motion.div>

      {/* ─── Report Readiness Checklist ─── */}
      <motion.div variants={itemVariants}>
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base font-semibold flex items-center gap-2">
                  <Shield className="h-5 w-5 text-emerald-500" />
                  Report Readiness
                </CardTitle>
                <CardDescription className="text-xs mt-1">
                  Complete these steps to include all sections in your report
                </CardDescription>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                  {completedCount}/{checklist.length}
                </p>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Steps Done
                </p>
              </div>
            </div>
            <Progress
              value={completionPercent}
              className="mt-3 h-2"
            />
            <div className="flex justify-between mt-1">
              <span className="text-[10px] text-muted-foreground">
                {completionPercent < 100 ? `${100 - completionPercent}% remaining` : 'All steps complete!'}
              </span>
              <span className={cn(
                'text-[10px] font-semibold',
                completionPercent === 100 ? 'text-emerald-600' : 'text-amber-600',
              )}>
                {completionPercent}%
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {checklist.map((item, idx) => (
                <ChecklistItem
                  key={item.label}
                  icon={item.icon}
                  label={item.label}
                  complete={item.complete}
                  delay={0.05 * idx}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ─── Report Content Preview ─── */}
      <motion.div variants={itemVariants} className="space-y-3">
        <div className="flex items-center gap-2 mb-1">
          <Eye className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold text-foreground">Report Content Preview</h3>
          <span className="text-[10px] text-muted-foreground">— Expand sections to preview what will be included</span>
        </div>

        {/* Data Overview */}
        <PreviewSection
          icon={Database}
          title="Data Overview"
          description="File name, row count, column count, memory usage"
          hasData={!!rawData}
          defaultOpen
        >
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-muted/50 px-3 py-2">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">File Name</p>
              <p className="text-sm font-medium truncate">{fileName ?? 'N/A'}</p>
            </div>
            <div className="rounded-lg bg-muted/50 px-3 py-2">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Total Rows</p>
              <p className="text-sm font-medium">{totalRows.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-muted/50 px-3 py-2">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Total Columns</p>
              <p className="text-sm font-medium">{columns.length}</p>
            </div>
            <div className="rounded-lg bg-muted/50 px-3 py-2">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Memory Usage</p>
              <p className="text-sm font-medium">{memoryUsage || 'N/A'}</p>
            </div>
          </div>
        </PreviewSection>

        {/* Column Analysis */}
        <PreviewSection
          icon={Table2}
          title="Column Analysis"
          description={`Column names, types, and roles (${columns.length} columns)`}
          hasData={columns.length > 0}
        >
          <ScrollArea className="max-h-64">
            <div className="space-y-2">
              {columns.slice(0, 10).map((col) => (
                <div key={col.name} className="flex items-center gap-2 text-xs">
                  <span className="font-mono font-medium text-emerald-600 dark:text-emerald-400 truncate max-w-[140px]">
                    {col.name}
                  </span>
                  <Badge variant="outline" className="text-[10px] h-5">
                    {col.dtype}
                  </Badge>
                  <Badge variant="outline" className="text-[10px] h-5">
                    {col.role}
                  </Badge>
                </div>
              ))}
              {columns.length > 10 && (
                <p className="text-xs text-muted-foreground">
                  ... and {columns.length - 10} more columns
                </p>
              )}
            </div>
          </ScrollArea>
        </PreviewSection>

        {/* Cleaning Summary */}
        <PreviewSection
          icon={Droplets}
          title="Cleaning Summary"
          description={`${cleaningDone ? `${cleaningLogs.length} operations performed` : 'No cleaning done yet'}`}
          hasData={cleaningDone && cleaningLogs.length > 0}
        >
          {cleaningLogs.length > 0 ? (
            <div className="space-y-2">
              {cleaningLogs.map((log, idx) => (
                <div key={idx} className="flex items-start gap-2 text-xs rounded-lg bg-muted/50 px-3 py-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">{log.action}</p>
                    <p className="text-muted-foreground">{log.detail}</p>
                  </div>
                  <span className="ml-auto text-[10px] text-muted-foreground shrink-0">{log.timestamp}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-4">
              No cleaning operations recorded. Run data cleaning to populate this section.
            </p>
          )}
        </PreviewSection>

        {/* EDA Findings */}
        <PreviewSection
          icon={BarChart3}
          title="EDA Findings"
          description={`${edaStats.numericColumns.length} numeric, ${edaStats.categoricalColumns.length} categorical columns analyzed`}
          hasData={edaStats.numericColumns.length > 0 || !!aiInsights}
        >
          {edaStats.numericColumns.length > 0 ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Statistical Summary (numeric columns)</p>
                <div className="max-h-56 space-y-1.5 overflow-y-auto pr-1">
                  {edaStats.numericColumns.slice(0, 8).map((col) => {
                    const s = edaStats.stats[col];
                    if (!s) return null;
                    return (
                      <div
                        key={col}
                        className="flex flex-col gap-1 rounded-lg bg-muted/50 px-3 py-2 text-xs sm:flex-row sm:items-center sm:gap-3"
                      >
                        <span className="font-mono font-medium text-emerald-600 dark:text-emerald-400 break-all sm:max-w-[140px]">
                          {col}
                        </span>
                        <div className="flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                          <span>&mu;={s.mean}</span>
                          <span>??={s.std}</span>
                          <span>[{s.min}???{s.max}]</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              {edaStats.correlations.length > 0 && (
                <div className="space-y-2 border-t border-border/50 pt-3">
                  <p className="text-xs font-medium text-muted-foreground">Top Correlations</p>
                  <div className="space-y-1.5">
                    {edaStats.correlations.slice(0, 5).map((c) => (
                      <div
                        key={c.pair}
                        className="flex items-center justify-between gap-3 rounded-lg bg-muted/30 px-3 py-2 text-xs"
                      >
                        <span className="min-w-0 flex-1 break-words">{c.pair}</span>
                        <Badge
                          variant="outline"
                          className={cn(
                            'shrink-0 text-[10px] h-5',
                            c.correlation > 0 ? 'text-emerald-600 border-emerald-300' : 'text-red-600 border-red-300',
                          )}
                        >
                          {c.correlation > 0 ? '+' : ''}{c.correlation}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-4">
              Open the EDA tab to populate this section.
            </p>
          )}
        </PreviewSection>

        {/* Sales Forecast */}
        <PreviewSection
          icon={TrendingUp}
          title="Sales Forecast"
          description={salesForecastResult ? `${salesForecastResult.training_summary.model_name} on ${salesForecastResult.training_summary.total_periods} periods` : 'No sales forecast run yet'}
          hasData={!!salesForecastResult}
        >
          {salesForecastResult ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Date Column</p>
                  <p className="text-sm font-medium">{salesForecastResult.date_column}</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Sales Column</p>
                  <p className="text-sm font-medium">{salesForecastResult.target_column}</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Split</p>
                  <p className="text-sm font-medium">{salesForecastResult.training_summary.train_percentage}% / {salesForecastResult.training_summary.test_percentage}%</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Forecast Horizon</p>
                  <p className="text-sm font-medium">{salesForecastResult.training_summary.forecast_periods}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-lg bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border border-emerald-500/10 px-3 py-2 text-center">
                  <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{salesForecastResult.metrics.mae.toLocaleString()}</p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">MAE</p>
                </div>
                <div className="rounded-lg bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border border-emerald-500/10 px-3 py-2 text-center">
                  <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{salesForecastResult.metrics.rmse.toLocaleString()}</p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">RMSE</p>
                </div>
                <div className="rounded-lg bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border border-emerald-500/10 px-3 py-2 text-center">
                  <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{salesForecastResult.metrics.mape}%</p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">MAPE</p>
                </div>
              </div>
              <div className="rounded-lg bg-muted/50 px-3 py-2">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Forecast Insight</p>
                <p className="text-xs">{salesForecastResult.analysis}</p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-4">
              Run the Sales Forecast tab to include time-series forecasting results in the report.
            </p>
          )}
        </PreviewSection>


        {/* ML Model Results */}
        <PreviewSection
          icon={BrainCircuit}
          title="ML Model Results"
          description={selectedModel ? `${selectedModel} — ${problemType}` : 'No model trained yet'}
          hasData={modelTrained && !!selectedModel}
        >
          {modelTrained && selectedModel ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Target</p>
                  <p className="text-sm font-medium">{targetColumn ?? 'N/A'}</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Problem Type</p>
                  <p className="text-sm font-medium capitalize">{problemType}</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Model</p>
                  <p className="text-sm font-medium">{selectedModel}</p>
                </div>
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Features</p>
                  <p className="text-sm font-medium">{selectedFeatures.length} selected</p>
                </div>
              </div>
              {modelMetrics && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Performance Metrics</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {Object.entries(modelMetrics).map(([key, value]) => (
                      <div key={key} className="rounded-lg bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border border-emerald-500/10 px-3 py-2 text-center">
                        <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                          {typeof value === 'number' ? (value * 100 > 1 ? value.toFixed(2) : (value * 100).toFixed(1) + '%') : String(value)}
                        </p>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{key}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {featureImportance && featureImportance.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1.5">Top Features</p>
                  <ScrollArea className="max-h-32">
                    <div className="space-y-1">
                      {featureImportance.slice(0, 6).map((f, idx) => (
                        <div key={f.name} className="flex items-center gap-2 text-xs">
                          <span className="text-muted-foreground w-4">{idx + 1}.</span>
                          <span className="font-mono truncate max-w-[120px]">{f.name}</span>
                          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-500"
                              style={{ width: `${Math.min(100, (f.importance / (featureImportance[0]?.importance || 1)) * 100)}%` }}
                            />
                          </div>
                          <span className="text-muted-foreground w-12 text-right">{f.importance.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-4">
              Train a machine learning model to populate this section.
            </p>
          )}
        </PreviewSection>

        {/* Prediction Results */}
        <PreviewSection
          icon={TrendingUp}
          title="Prediction Results"
          description={predictionResult !== null ? `Predicted: ${predictionResult}` : 'No predictions made yet'}
          hasData={predictionResult !== null}
        >
          {predictionResult !== null ? (
            <div className="space-y-3">
              <div className="rounded-lg bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 px-4 py-3 text-center">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Predicted Value</p>
                <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                  {typeof predictionResult === 'number' ? predictionResult.toLocaleString() : predictionResult}
                </p>
              </div>
              {uploadedModel && (
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Prediction Model</p>
                  <p className="text-xs">{uploadedModel.name} for {uploadedModel.target} ({uploadedModel.problem})</p>
                </div>
              )}
              {predictionAnalysis && (
                <div className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Analysis</p>
                  <p className="text-xs">{predictionAnalysis}</p>
                </div>
              )}
              {predictionProbabilities && Object.keys(predictionProbabilities).length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Class Probabilities</p>
                  <div className="space-y-1">
                    {Object.entries(predictionProbabilities).slice(0, 5).map(([label, probability]) => (
                      <div key={label} className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 px-3 py-1.5 text-xs">
                        <span>{label}</span>
                        <span className="font-medium">{(probability * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {predictionHistory.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    Prediction History ({predictionHistory.length} entries)
                  </p>
                  <ScrollArea className="max-h-32">
                    <div className="space-y-1">
                      {predictionHistory.slice(-5).map((ph) => (
                        <div key={ph.id} className="flex items-center gap-2 text-xs rounded-lg bg-muted/50 px-3 py-1.5">
                          <span className="font-mono font-medium text-emerald-600 dark:text-emerald-400">
                            {ph.prediction}
                          </span>
                          {ph.confidence !== undefined && (
                            <span className="text-muted-foreground">
                              confidence: {(ph.confidence * 100).toFixed(1)}%
                            </span>
                          )}
                          <span className="ml-auto text-[10px] text-muted-foreground">{ph.timestamp}</span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground text-center py-4">
              Make a prediction to populate this section.
            </p>
          )}
        </PreviewSection>
      </motion.div>

      {/* ─── Generate / Download Buttons ─── */}
      <motion.div variants={itemVariants} className="space-y-3">
        {!reportGenerated ? (
          <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
            <Button
              onClick={handleGenerate}
              disabled={generating || !rawData}
              size="lg"
              className={cn(
                'w-full h-14 text-base font-semibold gap-3 shadow-xl transition-all duration-300',
                'bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-600',
                'hover:from-emerald-600 hover:via-teal-600 hover:to-emerald-700',
                'hover:shadow-2xl hover:shadow-emerald-500/30',
                'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-xl',
              )}
            >
              <AnimatePresence mode="wait">
                {!generating ? (
                  <motion.span
                    key="idle"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex items-center gap-3"
                  >
                    <Sparkles className="h-5 w-5" />
                    Generate Updated PDF Report
                  </motion.span>
                ) : (
                  <motion.span
                    key="loading"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex items-center gap-3"
                  >
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Generating Updated Report...
                  </motion.span>
                )}
              </AnimatePresence>
            </Button>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            {/* Success banner */}
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              className="flex items-center gap-3 rounded-xl bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 p-4"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-emerald-500/20">
                <Award className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">
                  Report Generated Successfully!
                </p>
                <p className="text-xs text-muted-foreground">
                  Your updated workflow analysis PDF has been downloaded.
                </p>
              </div>
            </motion.div>

            <div className="flex gap-3">
              <Button
                onClick={handleDownloadAgain}
                variant="outline"
                className="flex-1 h-12 gap-2 hover:border-emerald-300 hover:text-emerald-600"
              >
                <Download className="h-4 w-4" />
                Download Again
              </Button>
              <Button
                onClick={handleRegenerate}
                variant="outline"
                className="flex-1 h-12 gap-2 hover:border-emerald-300 hover:text-emerald-600"
              >
                <RefreshCw className="h-4 w-4" />
                Regenerate Report
              </Button>
            </div>
          </motion.div>
        )}

        {/* Info note */}
        <div className="flex items-start gap-2 rounded-lg bg-muted/30 px-3 py-2">
          <Zap className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            The report is generated entirely on the server. Your data is processed in-memory and not stored.
            Only the data available from completed workflow steps will be included in the report.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Upload icon alias (not imported at top-level to avoid confusion)
// ──────────────────────────────────────────────────────────
function UploadIcon(props: React.SVGProps<SVGSVGElement> & { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}



