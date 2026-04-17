'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { AlertCircle, BarChart3, Bot, Lightbulb, Loader2, Maximize2, ScanSearch, ShieldCheck, TrendingUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { ColumnInfo, DataRow } from '@/lib/store';

type AdvancedEdaResponse = {
  row_count: number;
  sampled_row_count: number;
  column_count: number;
  missingness: {
    status: 'success' | 'chart' | 'empty' | 'error';
    message: string | null;
    total_missing: number;
    chart_base64: string | null;
    columns_analyzed: string[];
    row_groups: number;
  };
  distributions: {
    status: 'chart' | 'empty' | 'error';
    message: string | null;
    chart_base64: string | null;
    columns_analyzed: string[];
    charts: { column: string; chart_base64: string | null }[];
  };
  categorical: {
    status: 'chart' | 'empty' | 'error';
    message: string | null;
    charts: { column: string; unique_count: number; chart_base64: string | null }[];
    warnings: { column: string; unique_count: number; message: string }[];
  };
  interactions: {
    status: 'chart' | 'empty' | 'error';
    message: string | null;
    plots: { pair: string; correlation: number; chart_base64: string | null }[];
  };
  insights: {
    status: 'success';
    message: string | null;
    insights: string[];
  };
};

const ADVANCED_EDA_CLIENT_SAMPLE_LIMIT = 1200;

function sampleRowsForAdvancedEda(data: DataRow[], limit = ADVANCED_EDA_CLIENT_SAMPLE_LIMIT) {
  if (data.length <= limit) return data;
  const step = Math.max(1, Math.floor(data.length / limit));
  const sampled: DataRow[] = [];
  for (let index = 0; index < data.length && sampled.length < limit; index += step) {
    sampled.push(data[index]);
  }
  return sampled;
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-dashed bg-muted/20 px-6 py-10 text-center">
      <p className="max-w-md text-sm text-muted-foreground">{message}</p>
    </div>
  );
}

function ImageFrame({
  src,
  alt,
  className,
  imageClassName,
}: {
  src: string | null;
  alt: string;
  className?: string;
  imageClassName?: string;
}) {
  if (!src) {
    return <EmptyState message="This visualization could not be generated for the current dataset." />;
  }

  return (
    <div className={cn('overflow-hidden rounded-2xl border border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#f8fafc_100%)] p-3 shadow-sm dark:border-slate-800 dark:bg-[linear-gradient(180deg,#0f172a_0%,#111827_100%)]', className)}>
      <div className="rounded-xl border border-slate-200/80 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
        <img src={src} alt={alt} className={cn('mx-auto block h-auto w-full max-w-full rounded-lg object-contain', imageClassName)} />
      </div>
    </div>
  );
}

function ChartPanel({
  title,
  description,
  badges,
  children,
  chartImage,
}: {
  title: string;
  description: string;
  badges?: ReactNode;
  children: ReactNode;
  chartImage?: { src: string | null; alt: string };
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-2xl border border-border/70 bg-card/60 p-4 shadow-sm">
      <div className="mb-4 flex flex-col gap-3 border-b border-border/70 pb-4">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <h4 className="text-base font-semibold tracking-tight text-foreground">{title}</h4>
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {badges}
            {chartImage?.src ? (
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-full"
                onClick={() => setIsExpanded(true)}
                aria-label={`Expand ${title} chart`}
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            ) : null}
          </div>
        </div>
      </div>
      {children}
      {chartImage?.src ? (
        <Dialog open={isExpanded} onOpenChange={setIsExpanded}>
          <DialogContent
            showCloseButton={false}
            className="h-[92vh] max-h-[92vh] w-[94vw] max-w-[94vw] gap-0 overflow-hidden rounded-[28px] border border-slate-800/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.98),rgba(244,247,251,0.96))] p-0 shadow-[0_26px_90px_-38px_rgba(15,23,42,0.72)] dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.98),rgba(15,23,42,0.95))]"
          >
            <div className="flex items-center justify-between gap-4 border-b border-border/70 px-5 py-4">
              <div className="min-w-0">
                <DialogTitle className="truncate text-base">{title}</DialogTitle>
                <DialogDescription className="mt-1">{description}</DialogDescription>
              </div>
              <Button type="button" variant="outline" size="sm" onClick={() => setIsExpanded(false)}>
                Close
              </Button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <ImageFrame
                src={chartImage.src}
                alt={chartImage.alt}
                className="h-full border-transparent bg-transparent p-0 shadow-none"
                imageClassName="h-full max-h-[calc(92vh-8rem)] w-full"
              />
            </div>
          </DialogContent>
        </Dialog>
      ) : null}
    </div>
  );
}

function AnalysisModeBanner({
  requestMode,
  sampledRowCount,
  totalRowCount,
}: {
  requestMode: 'cached' | 'sampled' | null;
  sampledRowCount: number;
  totalRowCount: number;
}) {
  if (requestMode === 'cached') {
    return (
      <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-primary">
        Advanced EDA is running against the cached backend dataset for full-fidelity analysis.
      </div>
    );
  }

  if (requestMode === 'sampled') {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/25 dark:text-amber-200">
        Advanced EDA is using a capped sample of {sampledRowCount.toLocaleString()} row{sampledRowCount === 1 ? '' : 's'} from {totalRowCount.toLocaleString()} total rows because a cached backend dataset was not available for this session.
      </div>
    );
  }

  return null;
}

export default function EdaAdvancedModules({
  datasetId,
  data,
  columns,
}: {
  datasetId: string | null;
  data: DataRow[];
  columns: ColumnInfo[];
}) {
  const [analysis, setAnalysis] = useState<AdvancedEdaResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestMode, setRequestMode] = useState<'cached' | 'sampled' | null>(null);

  useEffect(() => {
    let isCancelled = false;

    async function runAdvancedEda() {
      if (!data.length || !columns.length) {
        setAnalysis(null);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const sampledRows = sampleRowsForAdvancedEda(data);
        const payload = datasetId
          ? { dataset_id: datasetId, data: [] as DataRow[] }
          : { dataset_id: null, data: sampledRows };
        const response = await apiClient.post<AdvancedEdaResponse>('/eda/advanced', payload);
        if (!isCancelled) {
          setAnalysis(response.data);
          setRequestMode(datasetId ? 'cached' : 'sampled');
        }
      } catch (requestError) {
        if (!isCancelled) {
          setError(getApiErrorMessage(requestError, 'Advanced EDA could not be generated for this dataset.'));
          setAnalysis(null);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }

    void runAdvancedEda();

    return () => {
      isCancelled = true;
    };
  }, [columns, data, datasetId]);

  if (!data.length || !columns.length) {
    return null;
  }

  if (loading && !analysis) {
    return (
      <div className="space-y-6">
        {[
          'Data Quality & Missingness',
          'Distributions & Outliers',
          'Categorical Analysis',
          'Key Variable Interactions',
          'Automated Insights & Recommendations',
        ].map((title) => (
          <Card key={title}>
            <CardHeader>
              <CardTitle>{title}</CardTitle>
              <CardDescription>Preparing advanced EDA outputs for the current dataset.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-dashed bg-muted/20">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing dataset safely...
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (error && !analysis) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Advanced EDA</CardTitle>
          <CardDescription>The appended analytical modules could not be loaded.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed bg-muted/20 px-6 text-center">
            <AlertCircle className="h-8 w-8 text-muted-foreground/60" />
            <p className="max-w-xl text-sm text-muted-foreground">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!analysis) {
    return null;
  }

  return (
    <div className="space-y-6">
      <AnalysisModeBanner
        requestMode={requestMode}
        sampledRowCount={analysis.sampled_row_count}
        totalRowCount={analysis.row_count}
      />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-primary" /> Data Quality & Missingness</CardTitle>
          <CardDescription>
            Missing-value coverage and row-group missingness behavior for the current working dataset.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">{analysis.row_count.toLocaleString()} rows</Badge>
            <Badge variant="secondary">{analysis.column_count.toLocaleString()} columns</Badge>
            <Badge variant="secondary">{analysis.missingness.total_missing.toLocaleString()} missing values</Badge>
          </div>
          {analysis.missingness.status === 'success' ? (
            <div className="rounded-xl border border-primary/20 bg-primary/5 px-5 py-6 text-center">
              <p className="font-medium text-primary">Data quality check passed: no missing values were detected.</p>
            </div>
          ) : analysis.missingness.status === 'chart' ? (
            <>
              {analysis.missingness.message && (
                <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
                  {analysis.missingness.message}
                </div>
              )}
              <ChartPanel
                title="Missingness Intensity Map"
                description="Grouped view of missing-value concentration across the dataset."
                chartImage={{ src: analysis.missingness.chart_base64, alt: 'Missingness heatmap' }}
                badges={
                  <>
                    <Badge variant="outline">{analysis.missingness.columns_analyzed.length} columns shown</Badge>
                    <Badge variant="secondary">{analysis.missingness.row_groups} row groups</Badge>
                  </>
                }
              >
                <ImageFrame src={analysis.missingness.chart_base64} alt="Missingness heatmap" />
              </ChartPanel>
            </>
          ) : (
            <EmptyState message={analysis.missingness.message || 'Missingness analysis is not available for this dataset.'} />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-primary" /> Distributions & Outliers</CardTitle>
          <CardDescription>
            Histogram, KDE, and boxplot screening for the first safe set of numeric columns.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {analysis.distributions.message && (
            <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
              {analysis.distributions.message}
            </div>
          )}
          {analysis.distributions.status === 'chart' ? (
            <div className="grid gap-4 xl:grid-cols-2">
              {analysis.distributions.charts.map((chart) => (
                <ChartPanel
                  key={chart.column}
                  title={chart.column}
                  description="Combined distribution and outlier screening sized to compare side by side on larger screens."
                  chartImage={{ src: chart.chart_base64, alt: `${chart.column} distribution and outlier chart` }}
                  badges={<Badge variant="outline">{chart.column}</Badge>}
                >
                  <ImageFrame src={chart.chart_base64} alt={`${chart.column} distribution and outlier chart`} />
                </ChartPanel>
              ))}
            </div>
          ) : (
            <EmptyState message={analysis.distributions.message || 'No numeric columns available for this analysis.'} />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ScanSearch className="h-4 w-4 text-primary" /> Categorical Analysis</CardTitle>
          <CardDescription>
            Top-category frequency views for categorical fields, capped for stable rendering on wide datasets.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {analysis.categorical.warnings.length > 0 && (
            <div className="space-y-3">
              {analysis.categorical.warnings.map((warning) => (
                <div key={`${warning.column}-${warning.unique_count}`} className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/25 dark:text-amber-200">
                  High cardinality: column '{warning.column}' has {warning.unique_count} unique values. Consider encoding strategies before ML.
                </div>
              ))}
            </div>
          )}
          {analysis.categorical.message && (
            <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
              {analysis.categorical.message}
            </div>
          )}
          {analysis.categorical.status === 'chart' ? (
            <div className="grid gap-4 xl:grid-cols-2">
              {analysis.categorical.charts.map((chart) => (
                <ChartPanel
                  key={chart.column}
                  title={chart.column}
                  description="Top-category frequency distribution scaled to fit the available panel width."
                  chartImage={{ src: chart.chart_base64, alt: `${chart.column} categorical analysis` }}
                  badges={
                    <>
                      <Badge variant="outline">{chart.column}</Badge>
                      <Badge variant="secondary">{chart.unique_count.toLocaleString()} unique values</Badge>
                    </>
                  }
                >
                  <ImageFrame src={chart.chart_base64} alt={`${chart.column} categorical analysis`} />
                </ChartPanel>
              ))}
            </div>
          ) : (
            <EmptyState message={analysis.categorical.message || 'No categorical columns available for this analysis.'} />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><TrendingUp className="h-4 w-4 text-primary" /> Key Variable Interactions</CardTitle>
          <CardDescription>
            Top numeric relationships after dropping constant columns and stabilizing the correlation matrix.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {analysis.interactions.message && (
            <div className="rounded-xl border bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
              {analysis.interactions.message}
            </div>
          )}
          {analysis.interactions.status === 'chart' ? (
            <div className="grid gap-4 xl:grid-cols-2">
              {analysis.interactions.plots.map((plot) => (
                <ChartPanel
                  key={plot.pair}
                  title={plot.pair}
                  description="Scatter view with trend overlay resized to stay readable without overflow."
                  chartImage={{ src: plot.chart_base64, alt: `${plot.pair} interaction chart` }}
                  badges={
                    <>
                      <Badge variant="outline" className={cn('max-w-full truncate')}>{plot.pair}</Badge>
                      <Badge variant="secondary">Corr {plot.correlation.toFixed(2)}</Badge>
                    </>
                  }
                >
                  <ImageFrame src={plot.chart_base64} alt={`${plot.pair} interaction chart`} />
                </ChartPanel>
              ))}
            </div>
          ) : (
            <EmptyState message={analysis.interactions.message || 'Need at least 2 numeric columns.'} />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Bot className="h-4 w-4 text-primary" /> Automated Insights & Recommendations</CardTitle>
          <CardDescription>
            Safe anomaly screening across skewness, multicollinearity, and IQR-based outlier signals.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {analysis.insights.insights.length === 0 ? (
            <div className="rounded-xl border border-primary/20 bg-primary/5 px-5 py-6 text-center">
              <p className="font-medium text-primary">No major statistical anomalies were detected.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {analysis.insights.insights.map((insight) => (
                <div key={insight} className="flex gap-3 rounded-xl border bg-background p-4 shadow-sm">
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-lg shadow-primary/20">
                    <Lightbulb className="h-4 w-4" />
                  </div>
                  <p className="text-sm leading-6 text-foreground">{insight}</p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
