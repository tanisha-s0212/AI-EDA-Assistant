'use client';

import { useMemo } from 'react';
import { AlertCircle, BarChart3, Database, Sigma, Table as TableIcon, TrendingUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
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
import EdaAdvancedModules from '@/components/tabs/eda-advanced-modules';

export default function EdaTab() {
  const rawData = useAppStore((s) => s.rawData);
  const cleanedData = useAppStore((s) => s.cleanedData);
  const columns = useAppStore((s) => s.columns);
  const totalRows = useAppStore((s) => s.totalRows);
  const loadedRowCount = useAppStore((s) => s.loadedRowCount);
  const previewLoaded = useAppStore((s) => s.previewLoaded);
  const datasetId = useAppStore((s) => s.datasetId);

  const data = cleanedData ?? rawData ?? [];
  const hasData = !!rawData && columns.length > 0;

  const stats = useMemo(() => computeEdaStats(data, columns), [data, columns]);
  const heatmap = useMemo(() => buildCorrelationHeatmap(data, stats.numericColumns), [data, stats.numericColumns]);
  const analysisBaseLabel = previewLoaded && totalRows > loadedRowCount ? 'preview dataset sample' : 'current working dataset';
  const numericSummaryScope = stats.numericColumns.length ? `Computed from the ${analysisBaseLabel} for responsive analysis.` : 'Descriptive statistics for numeric columns.';
  const numericRows = useMemo(
    () => stats.numericColumns.map((name) => ({ name, summary: stats.stats[name] })).filter((item) => item.summary),
    [stats.numericColumns, stats.stats]
  );
  const strongestCorrelation = stats.correlations[0];

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

      <EdaAdvancedModules datasetId={datasetId} data={data} columns={columns} />
    </div>
  );
}
