'use client';

import { useCallback, useMemo, useState } from 'react';
import { AlertCircle, BarChart3, Database, Download, Loader2, Sigma, Table as TableIcon, TrendingUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table as ShadTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { buildCorrelationHeatmap, computeEdaStats, formatMetric, heatColor } from '@/lib/eda';
import { useAppStore } from '@/lib/store';
import EdaAdvancedModules, { type AdvancedEdaResponse } from '@/components/tabs/eda-advanced-modules';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderRows(rows: string[][]) {
  return rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('');
}

export default function EdaTab() {
  const { toast } = useToast();
  const rawData = useAppStore((s) => s.rawData);
  const cleanedData = useAppStore((s) => s.cleanedData);
  const columns = useAppStore((s) => s.columns);
  const totalRows = useAppStore((s) => s.totalRows);
  const loadedRowCount = useAppStore((s) => s.loadedRowCount);
  const previewLoaded = useAppStore((s) => s.previewLoaded);
  const datasetId = useAppStore((s) => s.datasetId);
  const fileName = useAppStore((s) => s.fileName);
  const selectedSheets = useAppStore((s) => s.selectedSheets);
  const sheetMergeMode = useAppStore((s) => s.sheetMergeMode);

  const data = cleanedData ?? rawData ?? [];
  const hasData = !!rawData && columns.length > 0;
  const [advancedAnalysis, setAdvancedAnalysis] = useState<AdvancedEdaResponse | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  const stats = useMemo(() => computeEdaStats(data, columns), [data, columns]);
  const heatmap = useMemo(() => buildCorrelationHeatmap(data, stats.numericColumns), [data, stats.numericColumns]);
  const analysisBaseLabel = previewLoaded && totalRows > loadedRowCount ? 'preview dataset sample' : 'current working dataset';
  const numericSummaryScope = stats.numericColumns.length ? `Computed from the ${analysisBaseLabel} for responsive analysis.` : 'Descriptive statistics for numeric columns.';
  const numericRows = useMemo(
    () => stats.numericColumns.map((name) => ({ name, summary: stats.stats[name] })).filter((item) => item.summary),
    [stats.numericColumns, stats.stats]
  );
  const strongestCorrelation = stats.correlations[0];
  const heatmapColumns = stats.numericColumns.slice(0, 5);

  const getBlobErrorMessage = useCallback(async (error: unknown, fallback: string) => {
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
  }, []);

  const handleDownloadReport = useCallback(() => {
    if (!hasData) return;

    setIsDownloading(true);
    try {
      const schemaRows = columns.map((col) => ([
        col.name,
        col.dtype,
        totalRows.toLocaleString(),
        col.nonNull.toLocaleString(),
        col.nullCount.toLocaleString(),
        col.uniqueCount.toLocaleString(),
        `${totalRows ? ((col.uniqueCount / totalRows) * 100).toFixed(1) : '0.0'}%`,
        col.role,
      ]));

      const statsRows = numericRows.map(({ name, summary }) => ([
        name,
        formatMetric(summary.count),
        formatMetric(summary.mean),
        formatMetric(summary.std),
        formatMetric(summary.min),
        formatMetric(summary.q1),
        formatMetric(summary.median),
        formatMetric(summary.q3),
        formatMetric(summary.max),
      ]));

      const relationshipItems = stats.correlations.length
        ? stats.correlations.slice(0, 8).map((item) => `<li><strong>${escapeHtml(item.pair)}</strong><span>${item.correlation >= 0 ? '+' : ''}${item.correlation}</span></li>`).join('')
        : '<li><strong>Relationships</strong><span>Need at least two numeric columns</span></li>';

      const heatmapHeader = heatmapColumns.map((column) => `<th>${escapeHtml(column)}</th>`).join('');
      const heatmapBody = heatmapColumns.map((rowLabel) => {
        const cells = heatmapColumns.map((column) => {
          const cell = heatmap.find((item) => item.x === rowLabel && item.y === column);
          return `<td>${(cell?.value ?? 0).toFixed(2)}</td>`;
        }).join('');
        return `<tr><th>${escapeHtml(rowLabel)}</th>${cells}</tr>`;
      }).join('');

      const advancedSections = advancedAnalysis ? `
        <section>
          <h2>Advanced EDA Modules</h2>
          <div class="chip-row">
            <span class="chip">${advancedAnalysis.row_count.toLocaleString()} rows analyzed</span>
            <span class="chip">${advancedAnalysis.column_count.toLocaleString()} columns analyzed</span>
            <span class="chip">${advancedAnalysis.missingness.total_missing.toLocaleString()} missing values</span>
          </div>
        </section>
        <section>
          <h3>Data Quality & Missingness</h3>
          ${advancedAnalysis.missingness.message ? `<p>${escapeHtml(advancedAnalysis.missingness.message)}</p>` : ''}
          ${advancedAnalysis.missingness.chart_base64 ? `<img src="${advancedAnalysis.missingness.chart_base64}" alt="Missingness heatmap" />` : '<p>No missingness chart available.</p>'}
        </section>
        <section>
          <h3>Distributions & Outliers</h3>
          ${advancedAnalysis.distributions.message ? `<p>${escapeHtml(advancedAnalysis.distributions.message)}</p>` : ''}
          <div class="image-grid">
            ${advancedAnalysis.distributions.charts.map((chart) => `
              <figure>
                <figcaption>${escapeHtml(chart.column)}</figcaption>
                ${chart.chart_base64 ? `<img src="${chart.chart_base64}" alt="${escapeHtml(chart.column)} distribution chart" />` : '<p>No chart available.</p>'}
              </figure>
            `).join('')}
          </div>
        </section>
        <section>
          <h3>Categorical Analysis</h3>
          ${advancedAnalysis.categorical.message ? `<p>${escapeHtml(advancedAnalysis.categorical.message)}</p>` : ''}
          ${advancedAnalysis.categorical.warnings.length ? `<ul>${advancedAnalysis.categorical.warnings.map((warning) => `<li>${escapeHtml(`${warning.column}: ${warning.message}`)}</li>`).join('')}</ul>` : ''}
          <div class="image-grid">
            ${advancedAnalysis.categorical.charts.map((chart) => `
              <figure>
                <figcaption>${escapeHtml(chart.column)} (${chart.unique_count.toLocaleString()} unique)</figcaption>
                ${chart.chart_base64 ? `<img src="${chart.chart_base64}" alt="${escapeHtml(chart.column)} categorical chart" />` : '<p>No chart available.</p>'}
              </figure>
            `).join('')}
          </div>
        </section>
        <section>
          <h3>Key Variable Interactions</h3>
          ${advancedAnalysis.interactions.message ? `<p>${escapeHtml(advancedAnalysis.interactions.message)}</p>` : ''}
          <div class="image-grid">
            ${advancedAnalysis.interactions.plots.map((plot) => `
              <figure>
                <figcaption>${escapeHtml(plot.pair)} (Corr ${plot.correlation.toFixed(2)})</figcaption>
                ${plot.chart_base64 ? `<img src="${plot.chart_base64}" alt="${escapeHtml(plot.pair)} interaction chart" />` : '<p>No chart available.</p>'}
              </figure>
            `).join('')}
          </div>
        </section>
        <section>
          <h3>Automated Insights & Recommendations</h3>
          ${advancedAnalysis.insights.insights.length ? `<ul>${advancedAnalysis.insights.insights.map((insight) => `<li>${escapeHtml(insight)}</li>`).join('')}</ul>` : '<p>No major statistical anomalies were detected.</p>'}
        </section>
      ` : `
        <section>
          <h2>Advanced EDA Modules</h2>
          <p>Advanced EDA charts were not available at download time.</p>
        </section>
      `;

      const reportHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EDA Report - ${escapeHtml(fileName ?? 'dataset')}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    .page { max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }
    h1, h2, h3 { margin: 0 0 12px; }
    p { line-height: 1.6; }
    section { background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 20px; margin-top: 20px; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06); }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 20px; }
    .metric { background: linear-gradient(180deg, #eff6ff, #ffffff); border: 1px solid #bfdbfe; border-radius: 16px; padding: 16px; }
    .metric-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; }
    .metric-value { margin-top: 8px; font-size: 24px; font-weight: 700; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .chip { display: inline-flex; padding: 8px 12px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 999px; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }
    th, td { border: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; vertical-align: top; }
    th { background: #f8fafc; }
    .relationship-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
    .relationship-list li { display: flex; justify-content: space-between; gap: 12px; border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px 14px; background: #fff; }
    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 14px; }
    figure { margin: 0; border: 1px solid #e2e8f0; border-radius: 16px; padding: 12px; background: #fff; }
    figcaption { margin-bottom: 10px; font-weight: 600; }
    img { width: 100%; height: auto; border-radius: 12px; border: 1px solid #e2e8f0; background: #fff; }
  </style>
</head>
<body>
  <div class="page">
    <h1>Exploratory Data Analysis Report</h1>
    <p>Dataset: ${escapeHtml(fileName ?? 'Uploaded dataset')}</p>
    <p>This report exports the EDA tab summaries, tables, relationship analysis, correlation matrix, and available advanced EDA charts.</p>

    <div class="summary">
      <div class="metric"><div class="metric-label">Total Rows</div><div class="metric-value">${totalRows.toLocaleString()}</div></div>
      <div class="metric"><div class="metric-label">Rows In Workspace</div><div class="metric-value">${data.length.toLocaleString()}</div></div>
      <div class="metric"><div class="metric-label">Columns</div><div class="metric-value">${columns.length.toLocaleString()}</div></div>
      <div class="metric"><div class="metric-label">Numeric Columns</div><div class="metric-value">${stats.numericColumns.length.toLocaleString()}</div></div>
    </div>

    <section>
      <h2>Dataset Schema</h2>
      <table>
        <thead>
          <tr><th>Column</th><th>Type</th><th>Count</th><th>Non-Null</th><th>Missing</th><th>Unique</th><th>Unique %</th><th>Role</th></tr>
        </thead>
        <tbody>${renderRows(schemaRows)}</tbody>
      </table>
    </section>

    <section>
      <h2>Statistical Summary</h2>
      ${statsRows.length ? `
        <table>
          <thead>
            <tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th></tr>
          </thead>
          <tbody>${renderRows(statsRows)}</tbody>
        </table>
      ` : '<p>No numeric columns available for descriptive statistics.</p>'}
    </section>

    <section>
      <h2>Relationships</h2>
      <ul class="relationship-list">${relationshipItems}</ul>
    </section>

    <section>
      <h2>Correlation Heatmap</h2>
      ${heatmapColumns.length >= 2 ? `
        <table>
          <thead><tr><th></th>${heatmapHeader}</tr></thead>
          <tbody>${heatmapBody}</tbody>
        </table>
      ` : '<p>Need at least two numeric columns to render the heatmap matrix.</p>'}
    </section>

    ${advancedSections}
  </div>
</body>
</html>`;

      const blob = new Blob([reportHtml], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${(fileName ?? 'dataset').replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '') || 'dataset'}_eda_report.html`;
      link.click();
      URL.revokeObjectURL(url);
    } finally {
      setIsDownloading(false);
    }
  }, [advancedAnalysis, columns, data.length, fileName, hasData, heatmap, heatmapColumns, numericRows, stats, totalRows]);

  const handleDownloadPdf = useCallback(async () => {
    if (!hasData) return;

    setIsDownloadingPdf(true);
    try {
      const response = await apiClient.post('/eda/report', {
        datasetId: datasetId ?? null,
        fileName: fileName ?? 'dataset',
        totalRows,
        loadedRowCount,
        previewLoaded,
        columns: columns.map((column) => ({
          name: column.name,
          dtype: column.dtype,
          nonNull: column.nonNull,
          nullCount: column.nullCount,
          uniqueCount: column.uniqueCount,
          role: column.role,
        })),
        edaStats: stats,
        advancedAnalysis,
      }, {
        responseType: 'blob',
      });

      const blob = response.data as Blob;
      if (!blob || blob.size === 0) {
        throw new Error('The EDA PDF service returned an empty file.');
      }

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      const baseName = (fileName ?? 'dataset').replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9-_ ]+/g, '').trim().replace(/\s+/g, '_');
      anchor.href = url;
      anchor.download = `${baseName || 'dataset'}_eda_report.pdf`;
      anchor.rel = 'noopener';
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      toast({
        title: 'EDA PDF downloaded',
        description: 'The EDA tab functionality, working flow, and advanced features were exported as a PDF.',
      });
    } catch (error) {
      toast({
        title: 'EDA PDF failed',
        description: await getBlobErrorMessage(error, 'Failed to generate the EDA PDF report.'),
        variant: 'destructive',
      });
    } finally {
      setIsDownloadingPdf(false);
    }
  }, [advancedAnalysis, columns, datasetId, fileName, getBlobErrorMessage, hasData, loadedRowCount, previewLoaded, stats, toast, totalRows]);

  if (!hasData) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Exploratory Data Analysis</h2>
          <p className="mt-1 text-muted-foreground">Upload a dataset to explore schema, statistics, relationships, and correlations here.</p>
        </div>
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <p className="font-medium">No dataset available yet</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Exploratory Data Analysis</h2>
          <p className="mt-1 text-muted-foreground">Schema, descriptive statistics, relationship signals, and visual correlation summaries for the active dataset.</p>
          {selectedSheets.length > 0 && (
            <p className="mt-1 text-xs text-muted-foreground">
              Workbook scope: {sheetMergeMode === 'stack' ? 'Stacked sheets' : 'Single sheet'} | {selectedSheets.join(', ')}
            </p>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary" className="gap-1"><Database className="h-3.5 w-3.5" /> {totalRows.toLocaleString()} total rows</Badge>
          <Badge variant="secondary" className="gap-1"><Database className="h-3.5 w-3.5" /> {data.length.toLocaleString()} in workspace</Badge>
          <Badge variant="secondary" className="gap-1"><Sigma className="h-3.5 w-3.5" /> {stats.numericColumns.length} numeric</Badge>
          <Badge variant="secondary" className="gap-1"><TableIcon className="h-3.5 w-3.5" /> {columns.length} columns</Badge>
        </div>
      </div>

      {previewLoaded && totalRows > loadedRowCount && (
        <Card className="border border-amber-300 bg-amber-50">
          <CardContent className="space-y-2 text-sm text-amber-900">
            <p>
              Showing a preview of {loadedRowCount.toLocaleString()} rows from {totalRows.toLocaleString()} total. The full dataset is cached on the backend so cleaning, advanced EDA, forecasting, and training can still use the complete source.
            </p>
            <p>
              On-screen summary cards in this tab use the loaded workspace rows to stay responsive with larger files.
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Dataset Schema</CardTitle>
          <CardDescription>Column names, data types, counts, completeness, uniqueness, and roles.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 overflow-auto rounded-lg border">
            <ShadTable>
              <TableHeader>
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="text-right">Count</TableHead>
                  <TableHead className="text-right">Non-Null</TableHead>
                  <TableHead className="text-right">Missing</TableHead>
                  <TableHead className="text-right">Unique</TableHead>
                  <TableHead className="text-right">Unique %</TableHead>
                  <TableHead>Role</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {columns.map((col) => {
                  const uniquePct = totalRows ? ((col.uniqueCount / totalRows) * 100).toFixed(1) : '0.0';
                  return (
                    <TableRow key={col.name}>
                      <TableCell className="font-medium">{col.name}</TableCell>
                      <TableCell>{col.dtype}</TableCell>
                      <TableCell className="text-right">{totalRows.toLocaleString()}</TableCell>
                      <TableCell className="text-right">{col.nonNull.toLocaleString()}</TableCell>
                      <TableCell className="text-right">{col.nullCount.toLocaleString()}</TableCell>
                      <TableCell className="text-right">{col.uniqueCount.toLocaleString()}</TableCell>
                      <TableCell className="text-right">{uniquePct}%</TableCell>
                      <TableCell className="capitalize">{col.role}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </ShadTable>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Statistical Summary</CardTitle>
          <CardDescription>{numericSummaryScope}</CardDescription>
        </CardHeader>
        <CardContent>
          {numericRows.length ? (
            <div className="overflow-auto rounded-lg border">
              <table className="w-full min-w-[860px] text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="px-3 py-2 text-left">Column</th>
                    <th className="px-3 py-2 text-right">Count</th>
                    <th className="px-3 py-2 text-right">Mean</th>
                    <th className="px-3 py-2 text-right">Std</th>
                    <th className="px-3 py-2 text-right">Min</th>
                    <th className="px-3 py-2 text-right">Q1</th>
                    <th className="px-3 py-2 text-right">Median</th>
                    <th className="px-3 py-2 text-right">Q3</th>
                    <th className="px-3 py-2 text-right">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {numericRows.map(({ name, summary }) => (
                    <tr key={name} className="border-t">
                      <td className="px-3 py-2 font-medium">{name}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.count)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.mean)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.std)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.min)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.q1)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.median)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.q3)}</td>
                      <td className="px-3 py-2 text-right">{formatMetric(summary.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No numeric columns available for descriptive statistics.</p>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card className="h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><TrendingUp className="h-4 w-4 text-primary" /> Relationships</CardTitle>
            <CardDescription>{strongestCorrelation ? `Strongest sampled relationship: ${strongestCorrelation.pair}` : 'Top numeric correlations from the loaded workspace rows.'}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {stats.correlations.length ? stats.correlations.slice(0, 8).map((item) => (
              <div key={item.pair} className="rounded-xl border bg-background p-4 shadow-sm">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-medium">{item.pair}</span>
                  <Badge
                    variant="outline"
                    className={item.correlation >= 0 ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-red-200 bg-red-50 text-red-700'}
                  >
                    {item.correlation >= 0 ? '+' : ''}{item.correlation}
                  </Badge>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(8, Math.abs(item.correlation) * 100)}%`,
                      backgroundColor: item.correlation >= 0 ? '#10b981' : '#ef4444',
                    }}
                  />
                </div>
              </div>
            )) : <p className="text-sm text-muted-foreground">Need at least two numeric columns.</p>}
          </CardContent>
        </Card>

        <Card className="h-full overflow-hidden">
          <CardHeader className="border-b bg-secondary/60">
            <CardTitle className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-primary" /> Correlation Heatmap</CardTitle>
            <CardDescription>Matrix view for up to five numeric columns from the active in-app dataset slice.</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {stats.numericColumns.length >= 2 ? (
              <div className="overflow-auto">
                <div className="grid gap-2 min-w-[620px]" style={{ gridTemplateColumns: `140px repeat(${Math.min(stats.numericColumns.length, 5)}, minmax(96px, 1fr))` }}>
                  <div />
                  {stats.numericColumns.slice(0, 5).map((column) => (
                    <div key={`head-${column}`} className="flex min-h-[72px] items-center justify-center rounded-lg bg-muted/40 px-3 py-3 text-center text-[11px] font-semibold leading-4 text-foreground">
                      <span className="break-words [word-break:break-word]">{column}</span>
                    </div>
                  ))}
                  {stats.numericColumns.slice(0, 5).map((rowLabel) => (
                    <div key={`row-${rowLabel}`} className="contents">
                      <div className="flex min-h-[72px] items-center rounded-lg bg-muted/40 px-3 py-3 text-[11px] font-semibold leading-4 text-foreground">
                        <span className="break-words [word-break:break-word]">{rowLabel}</span>
                      </div>
                      {stats.numericColumns.slice(0, 5).map((column) => {
                        const cell = heatmap.find((item) => item.x === rowLabel && item.y === column);
                        const value = cell?.value ?? 0;
                        const bg = heatColor(value);
                        return <div key={`${rowLabel}-${column}`} className="flex min-h-[72px] items-center justify-center rounded-lg border text-sm font-semibold shadow-sm" style={{ backgroundColor: bg }}>{value.toFixed(2)}</div>;
                      })}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Need at least two numeric columns.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <EdaAdvancedModules datasetId={datasetId} data={data} columns={columns} onAnalysisReady={setAdvancedAnalysis} />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Download className="h-4 w-4 text-primary" /> Download EDA Report</CardTitle>
          <CardDescription>Keep the current standalone EDA download as-is, or export the same EDA functionality and advanced features as a downloadable PDF.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-3xl text-sm text-muted-foreground">
            The current EDA report download remains unchanged. The new PDF option packages the EDA tab working flow, schema, summaries, correlations, advanced charts, and automated insights into a shareable PDF.
          </p>
          <div className="flex flex-col gap-2 self-start sm:flex-row sm:self-auto">
            <Button onClick={handleDownloadReport} disabled={isDownloading} variant="outline" className="gap-2">
              {isDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {isDownloading ? 'Preparing Report...' : 'Download EDA Report'}
            </Button>
            <Button onClick={() => void handleDownloadPdf()} disabled={isDownloadingPdf} className="gap-2">
              {isDownloadingPdf ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {isDownloadingPdf ? 'Preparing PDF...' : 'Download EDA PDF'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
