'use client';

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, Database, Eye, FileText, Info, Sparkles, Table as TableIcon, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import {
  Table as ShadTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { computeColumnMissingPercent, computeProfileCompleteness } from '@/lib/data-quality';
import { useAppStore } from '@/lib/store';

type SalesReadiness = {
  status: string;
  date_column?: string;
  target_column?: string;
  period_label?: string;
  period_count?: number;
  minimum_periods_for_ts?: number;
  date_span_days?: number;
  revenue_column_confidence?: number;
  zero_sales_share?: number;
  notes?: Array<string | null>;
};

type DatasetReview = {
  rows_analyzed: number;
  grain_changed?: boolean;
  grain: { grain: string; confidence?: string };
  coverage: null | { first_month: string | null; last_month: string | null; observed_months: number; usable_months: number; missing_months: string[]; missing_month_count: number; invalid_date_rows: number; invalid_target_rows: number; duplicate_key_rows: number | null; meets_minimum: boolean };
};
type FullQuality = { row_count: number; missing_cells: number; duplicate_rows: number | null; columns: Array<{ name: string; null_count: number; unique_count: number | null }> };

export default function UnderstandingTab() {
  const rawData = useAppStore((s) => s.rawData);
  const cleanedData = useAppStore((s) => s.cleanedData);
  const columns = useAppStore((s) => s.columns);
  const totalRows = useAppStore((s) => s.totalRows);
  const duplicates = useAppStore((s) => s.duplicates);
  const memoryUsage = useAppStore((s) => s.memoryUsage);
  const fileName = useAppStore((s) => s.fileName);
  const selectedSheets = useAppStore((s) => s.selectedSheets);
  const sheetMergeMode = useAppStore((s) => s.sheetMergeMode);
  const datasetId = useAppStore((s) => s.datasetId);
  const previewLoaded = useAppStore((s) => s.previewLoaded);
  const loadedRowCount = useAppStore((s) => s.loadedRowCount);
  const datasetDescription = useAppStore((s) => s.datasetDescription);
  const parseOptions = useAppStore((s) => s.datasetParseOptions);
  const savedMapping = useAppStore((s) => s.confirmedSalesMapping);
  const [salesReadiness, setSalesReadiness] = useState<SalesReadiness | null>(null);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [grain, setGrain] = useState<'auto' | 'monthly' | 'records'>('auto');
  const [dimensions, setDimensions] = useState<string[]>([]);
  const [review, setReview] = useState<DatasetReview | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [fullQuality, setFullQuality] = useState<FullQuality | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [columnSearch, setColumnSearch] = useState('');
  const [page, setPage] = useState(0);
  const [previewMode, setPreviewMode] = useState<'pages' | 'sample'>('pages');

  const data = cleanedData ?? rawData ?? [];
  const hasData = !!rawData && columns.length > 0;
  const displayColumns = useMemo(() => columns.filter((column) => column.name.toLowerCase().includes(columnSearch.toLowerCase())), [columns, columnSearch]);
  const maxPage = Math.max(0, Math.ceil(data.length / 10) - 1);
  const activePage = Math.min(page, maxPage);
  const sampleSize = Math.min(10, data.length);
  const previewIndices = previewMode === 'sample'
    ? Array.from({ length: sampleSize }, (_, index) => sampleSize <= 1 ? 0 : Math.floor(index * (data.length - 1) / (sampleSize - 1)))
    : Array.from({ length: Math.min(10, Math.max(0, data.length - activePage * 10)) }, (_, index) => activePage * 10 + index);
  const previewRows = previewIndices.map((index) => data[index]);
  const duplicatesChecked = data.length <= 10000;
  const missingColumns = columns.filter((col) => col.nullCount > 0);
  const numericColumns = columns.filter((col) => col.role === 'numeric');
  const categoricalColumns = columns.filter((col) => col.role === 'categorical' || col.role === 'boolean');
  const datetimeColumns = columns.filter((col) => col.role === 'datetime');
  const identifierColumns = columns.filter((col) => col.role === 'identifier');
  const highlyUniqueColumns = columns.filter((col) => col.nonNull + col.nullCount > 0 && col.uniqueCount / (col.nonNull + col.nullCount) >= 0.9);

  useEffect(() => {
    setDateColumn(savedMapping?.dateColumn ?? '');
    setTargetColumn(savedMapping?.targetColumn ?? '');
    setDimensions(savedMapping?.dimensions ?? []);
    setGrain(savedMapping?.grain ?? 'auto');
  }, [datasetId, savedMapping]);

  useEffect(() => {
    setReview(null);
    setReviewError(null);
    setFullQuality(null);
    setQualityError(null);
    setPage(0);
    setPreviewMode('pages');
    setColumnSearch('');
  }, [datasetId, rawData, cleanedData]);

  useEffect(() => {
    if (!datasetId) {
      setSalesReadiness(null);
      return;
    }
    let cancelled = false;
    setReadinessError(null);
    apiClient.post('/sales/readiness', { dataset_id: datasetId, date_column: savedMapping?.dateColumn || undefined, target_column: savedMapping?.targetColumn || undefined })
      .then((response) => {
        if (!cancelled) {
          const result = response.data as SalesReadiness;
          setSalesReadiness(result);
          setDateColumn((current) => current || result.date_column || '');
          setTargetColumn((current) => current || result.target_column || '');
        }
      })
      .catch((error) => {
        if (!cancelled) { setSalesReadiness(null); setReadinessError(getApiErrorMessage(error, 'Could not calculate forecast readiness.')); }
      });
    return () => { cancelled = true; };
  }, [datasetId, savedMapping]);
  const sparseColumns = [...missingColumns].sort((a, b) => b.nullCount - a.nullCount).slice(0, 3);
  const completeness = computeProfileCompleteness(columns);
  const profileScopeNote = previewLoaded && loadedRowCount > 0 && loadedRowCount < totalRows
    ? ` Figures use the loaded profile of ${loadedRowCount.toLocaleString()} of ${totalRows.toLocaleString()} rows.`
    : '';
  const qualitySignals = [
    {
      label: `${completeness}% overall completeness across ${columns.length} columns`,
      details: `The current dataset keeps ${completeness}% of all cells populated, which is a quick indicator of how much imputation or review may be needed before deeper analysis.${profileScopeNote}`,
    },
    {
      label: !duplicatesChecked ? 'Duplicate check unavailable for this loaded view' : duplicates > 0 ? `${duplicates.toLocaleString()} duplicate rows in the loaded view` : 'No duplicate rows detected in the loaded view',
      details: !duplicatesChecked ? 'The loaded view exceeded the local duplicate-check limit; no duplicate count was calculated.' : duplicates > 0
        ? `Duplicate records can bias counts, trend summaries, and training outcomes. Reviewing ${duplicates.toLocaleString()} repeated rows helps keep the dataset representative.`
        : 'No repeated full-row records were detected in the loaded view; other rows may not have been checked.',
    },
    {
      label: missingColumns.length > 0 ? `${missingColumns.length} columns contain missing values` : 'No columns with missing values detected',
      details: missingColumns.length > 0
        ? `Missingness is present across ${missingColumns.length} column${missingColumns.length === 1 ? '' : 's'}, which can affect segmentation, aggregates, and model readiness if left untreated.`
        : 'Column completeness looks strong in the current profile, so downstream cleaning can stay focused on structure and consistency.',
    },
  ];
  const structureSignals = [
    `${numericColumns.length} numeric column${numericColumns.length === 1 ? '' : 's'} for measures and trends`,
    `${categoricalColumns.length} categorical/boolean column${categoricalColumns.length === 1 ? '' : 's'} for segments and classes`,
    datetimeColumns.length > 0 ? `${datetimeColumns.length} datetime column${datetimeColumns.length === 1 ? '' : 's'} can support time-based analysis` : 'No datetime columns detected',
  ];
  const modelingSignals = [
    identifierColumns.length > 0 ? `${identifierColumns.length} identifier-like column${identifierColumns.length === 1 ? '' : 's'} should usually be excluded from modeling` : 'No strong identifier columns detected',
    highlyUniqueColumns.length > 0 ? `${highlyUniqueColumns.length} high-cardinality column${highlyUniqueColumns.length === 1 ? '' : 's'} may require encoding care` : 'Column cardinality looks manageable for standard ML workflows',
    sparseColumns.length > 0
      ? `Most incomplete column: ${sparseColumns[0].name} (${computeColumnMissingPercent(sparseColumns[0])}% missing)`
      : 'Missingness risk is low across the current dataset view',
  ];

  const runReview = async () => {
    if (!datasetId || !dateColumn || !targetColumn) { setReviewError('Select both a date column and a numeric target.'); return; }
    const reviewDatasetId = datasetId;
    setReviewLoading(true);
    setReviewError(null);
    try {
      const response = await apiClient.post<DatasetReview>('/dataset-understanding', {
        dataset_id: reviewDatasetId, date_column: dateColumn, target_column: targetColumn, dimensions, grain,
      });
      const current = useAppStore.getState();
      if (current.datasetId !== reviewDatasetId || current.rawData !== rawData || current.cleanedData !== cleanedData) return;
      useAppStore.setState({ confirmedSalesMapping: { dateColumn, targetColumn, dimensions, grain } });
      setReview(response.data);
    } catch (error) {
      setReview(null);
      setReviewError(getApiErrorMessage(error, 'Could not review selected columns.'));
    } finally { setReviewLoading(false); }
  };

  const runFullQuality = async () => {
    if (!datasetId) return;
    const activeId = datasetId;
    setQualityLoading(true);
    setQualityError(null);
    try {
      const response = await apiClient.post<FullQuality>('/dataset-quality-summary', { dataset_id: activeId });
      const current = useAppStore.getState();
      if (current.datasetId === activeId && current.rawData === rawData && current.cleanedData === cleanedData) setFullQuality(response.data);
    } catch (error) {
      setQualityError(getApiErrorMessage(error, 'Could not calculate full-dataset quality.'));
    } finally { setQualityLoading(false); }
  };

  if (!hasData) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Data Understanding</h2>
          <p className="mt-1 text-muted-foreground">Upload a dataset to view data quality and preview details here.</p>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-14 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground/50" />
            <p className="font-medium">No dataset loaded</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Data Understanding</h2>
        <p className="mt-1 text-muted-foreground">Review the uploaded dataset identity, quality checks, explainability signals, and a quick preview before exploratory data analysis and data cleaning.</p>
        {datasetDescription && <p className="mt-1 text-sm text-muted-foreground">Description: {datasetDescription}</p>}
        {parseOptions && <p className="mt-1 text-xs text-muted-foreground">Text parsing: delimiter {parseOptions.separator === '\t' ? 'tab' : parseOptions.separator}, encoding {parseOptions.encoding}, header row {parseOptions.header_row} (zero-based).</p>}
        <p className="mt-1 text-xs text-muted-foreground">Quality and duplicate figures describe {previewLoaded ? `the loaded preview (${loadedRowCount.toLocaleString()} of ${totalRows.toLocaleString()} rows)` : 'the current loaded view'}. Memory usage is an estimate.</p>
        {selectedSheets.length > 0 && (
          <p className="mt-1 text-xs text-muted-foreground">
            Workbook scope: {sheetMergeMode === 'stack' ? 'Stacked sheets' : 'Single sheet'} | {selectedSheets.join(', ')}
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {[
          { label: 'Dataset Name', value: fileName ?? 'Uploaded dataset', Icon: FileText },
          { label: 'Total Records', value: totalRows.toLocaleString(), Icon: Database },
          { label: 'Total Features', value: String(columns.length), Icon: TableIcon },
          { label: 'Duplicates in Loaded View', value: duplicatesChecked ? duplicates.toLocaleString() : 'Not checked', Icon: Sparkles },
          { label: 'Memory Estimate', value: memoryUsage || 'N/A', Icon: Database },
        ].map(({ label, value, Icon }) => (
          <motion.div
            key={label}
            whileHover={{ y: -8, rotateX: 3, rotateY: -3 }}
            transition={{ type: 'spring', stiffness: 260, damping: 18 }}
            className="[transform-style:preserve-3d]"
          >
          <Card className="group relative overflow-hidden border-border/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.72),rgba(248,250,252,0.94))] transition-all duration-500 hover:border-primary/30 hover:shadow-[0_28px_65px_-32px_rgba(37,99,235,0.38)] dark:bg-[linear-gradient(180deg,rgba(30,41,59,0.7),rgba(15,23,42,0.92))]">
            <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
              <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary/70 via-sky-400/65 to-emerald-400/65" />
              <div className="absolute -right-10 top-0 h-24 w-24 rounded-full bg-sky-400/15 blur-2xl" />
              <div className="absolute -left-8 bottom-0 h-20 w-20 rounded-full bg-emerald-400/10 blur-2xl" />
            </div>
            <CardContent className="relative flex items-center gap-3 pt-6">
              <div className="rounded-2xl bg-primary/10 p-3 transition-all duration-500 group-hover:scale-110 group-hover:bg-primary/15 group-hover:shadow-[0_14px_30px_-18px_rgba(37,99,235,0.55)]"><Icon className="h-5 w-5 text-primary transition-transform duration-500 group-hover:rotate-6" /></div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground transition-colors duration-300 group-hover:text-foreground/70">{label}</p>
                <p className="truncate text-lg font-semibold transition-transform duration-500 group-hover:translate-x-0.5">{value}</p>
              </div>
            </CardContent>
          </Card>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-primary/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> AI Explainability</CardTitle>
            <CardDescription>What the uploaded dataset is telling us about quality and readiness.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {qualitySignals.map((signal) => (
              <HoverCard key={signal.label} openDelay={120} closeDelay={100}>
                <HoverCardTrigger asChild>
                  <div className="flex cursor-default items-center justify-between rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                    <p className="pr-3 transition-colors duration-300 group-hover:text-foreground/85">{signal.label}</p>
                    <Info className="h-4 w-4 shrink-0 text-primary/70" />
                  </div>
                </HoverCardTrigger>
                <HoverCardContent align="start" className="w-[320px] rounded-2xl border-border/70 bg-popover/98 p-4 shadow-[0_24px_60px_-32px_rgba(15,23,42,0.35)]">
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-foreground">Explainability insight</p>
                    <p className="text-xs leading-6 text-muted-foreground">{signal.details}</p>
                  </div>
                </HoverCardContent>
              </HoverCard>
            ))}
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-sky-400/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><Database className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> Dataset Structure</CardTitle>
            <CardDescription>How the uploaded dataset is organized for analysis.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {structureSignals.map((signal) => (
              <div key={signal} className="rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                <p className="transition-colors duration-300 group-hover:text-foreground/85">{signal}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-emerald-400/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><TrendingUp className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> Sales Forecast Readiness</CardTitle>
            <CardDescription>Date span, period count, and revenue-column confidence for sales forecasting.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            {salesReadiness ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={salesReadiness.status === 'ready' ? 'default' : 'secondary'}>{salesReadiness.status}</Badge>
                  <span>Date: {salesReadiness.date_column ?? 'N/A'}</span>
                  <span>Target: {salesReadiness.target_column ?? 'N/A'}</span>
                </div>
                <p>Periods: {salesReadiness.period_count ?? 0} {salesReadiness.period_label ?? 'period'} periods (minimum {salesReadiness.minimum_periods_for_ts ?? 24} for multi-model Time Series)</p>
                <p>Date span: {salesReadiness.date_span_days ?? 0} days · Revenue confidence: {salesReadiness.revenue_column_confidence ?? 0} · Zero-sales share: {Math.round((salesReadiness.zero_sales_share ?? 0) * 100)}%</p>
                {(salesReadiness.notes ?? []).filter(Boolean).map((note) => (
                  <p key={String(note)} className="rounded-xl bg-amber-50 px-3 py-2 text-amber-800 dark:bg-amber-950/40 dark:text-amber-200">{note}</p>
                ))}
              </>
            ) : (
              <p role={readinessError ? 'alert' : undefined}>{readinessError ?? 'Calculating forecast readiness...'}</p>
            )}
          </CardContent>
        </Card>

        <Card className="group relative overflow-hidden border-border/70 bg-card/80 transition-all duration-500 hover:-translate-y-1.5 hover:border-primary/35 hover:bg-card hover:shadow-[0_24px_60px_-28px_rgba(37,99,235,0.35)]">
          <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-primary via-sky-400 to-emerald-400" />
            <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-emerald-400/10 blur-2xl" />
          </div>
          <CardHeader className="transition-colors duration-300 group-hover:text-foreground">
            <CardTitle className="flex items-center gap-2"><TableIcon className="h-4 w-4 text-primary transition-transform duration-300 group-hover:scale-110" /> Modeling Readiness</CardTitle>
            <CardDescription>Key explainability cues that can affect downstream ML behavior.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            {modelingSignals.map((signal) => (
              <div key={signal} className="rounded-xl px-3 py-2 transition-all duration-300 hover:bg-primary/5 hover:pl-4 hover:text-foreground/90">
                <p className="transition-colors duration-300 group-hover:text-foreground/85">{signal}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>


      <Card>
        <CardHeader><CardTitle>Full-dataset quality check</CardTitle><CardDescription>Optional exact check of the cached dataset. This may take time and memory for large files.</CardDescription></CardHeader>
        <CardContent className="space-y-3">
          <Button variant="outline" onClick={() => void runFullQuality()} disabled={!datasetId || qualityLoading}>{qualityLoading ? 'Checking full dataset...' : 'Check full dataset'}</Button>
          {qualityError && <p role="alert" className="text-sm text-destructive">{qualityError}</p>}
          {fullQuality && <div className="space-y-2 text-sm" aria-live="polite"><p>{fullQuality.row_count.toLocaleString()} rows checked; {fullQuality.missing_cells.toLocaleString()} missing cells; {fullQuality.duplicate_rows === null ? 'duplicate check unavailable for this data' : `${fullQuality.duplicate_rows.toLocaleString()} duplicate rows`}.</p><div className="max-h-52 overflow-auto"><table className="w-full text-left"><thead><tr><th>Column</th><th>Missing</th><th>Unique non-null</th></tr></thead><tbody>{fullQuality.columns.map((column) => <tr key={column.name}><td>{column.name}</td><td>{column.null_count.toLocaleString()}</td><td>{column.unique_count?.toLocaleString() ?? 'Unavailable'}</td></tr>)}</tbody></table></div></div>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Dataset grain and feature availability</CardTitle><CardDescription>Review the complete cached dataset and confirm the columns used for forecasting. Changing its grain or period column updates dataset metadata and invalidates models trained on an earlier version. Choose up to five dimensions when checking repeated monthly keys.</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <fieldset disabled={reviewLoading} className="space-y-4"><div className="grid gap-3 md:grid-cols-3">
            <label className="text-sm">Date column<select className="mt-1 w-full rounded border bg-background p-2" value={dateColumn} onChange={(event) => { const value = event.target.value; setDateColumn(value); setDimensions((current) => current.filter((item) => item !== value)); setReview(null); }}><option value="">Select date</option>{columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>
            <label className="text-sm">Numeric target<select className="mt-1 w-full rounded border bg-background p-2" value={targetColumn} onChange={(event) => { const value = event.target.value; setTargetColumn(value); setDimensions((current) => current.filter((item) => item !== value)); setReview(null); }}><option value="">Select target</option>{columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>
            <label className="text-sm">Grain<select className="mt-1 w-full rounded border bg-background p-2" value={grain} onChange={(event) => { setGrain(event.target.value as typeof grain); setReview(null); }}><option value="auto">Automatic</option><option value="monthly">Monthly aggregate</option><option value="records">Record data / unknown</option></select></label>
          </div>
          <label className="block text-sm">Dimensions (optional, select up to five)<select multiple className="mt-1 h-28 w-full rounded border bg-background p-2" value={dimensions} onChange={(event) => { setDimensions(Array.from(event.target.selectedOptions, (option) => option.value).slice(0, 5)); setReview(null); }}>{columns.filter((column) => column.name !== dateColumn && column.name !== targetColumn).map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>
          </fieldset>
          <Button onClick={() => void runReview()} disabled={reviewLoading || !datasetId}>{reviewLoading ? 'Reviewing dataset...' : 'Review and confirm mapping'}</Button>
          {reviewError && <p role="alert" className="text-sm text-destructive">{reviewError}</p>}
          {review && <div className="space-y-2 text-sm" aria-live="polite">
            <p>Detected grain: <strong>{review.grain.grain}</strong> ({review.grain.confidence ?? 'unknown'}); {review.rows_analyzed.toLocaleString()} rows reviewed.</p>
            {review.grain_changed && <p className="text-amber-700">Dataset grain metadata changed. Models trained on the previous dataset version must be retrained.</p>}
            {review.coverage && <p>Monthly cadence alone does not prove that rows are aggregated. Confirm what each row represents in the source data.</p>}
            {review.coverage && <>
              <p>Observed: {review.coverage.first_month} to {review.coverage.last_month}; {review.coverage.observed_months} observed months, {review.coverage.usable_months} with numeric target values. {review.coverage.meets_minimum ? 'The inclusive 24-month count is met.' : 'At least 24 usable months are required for multi-model forecasting.'}</p>
              <p>Missing months: {review.coverage.missing_month_count}{review.coverage.missing_months.length ? ` (${review.coverage.missing_months.join(', ')}${review.coverage.missing_month_count > review.coverage.missing_months.length ? ', …' : ''})` : ''}. Invalid dates: {review.coverage.invalid_date_rows}; invalid targets: {review.coverage.invalid_target_rows}.</p>
              <p>{review.coverage.duplicate_key_rows === null ? 'Select dimensions to check repeated monthly keys.' : `${review.coverage.duplicate_key_rows} rows repeat a month and selected dimensions.`}</p>
              {review.coverage.missing_month_count > 0 && <p className="text-amber-700">Missing months were not counted as observed. Review gaps before forecasting; existing forecast preparation may fill them.</p>}
            </>}
            <p><strong>EDA and comparisons:</strong> available for suitable columns. <strong>Forecasting:</strong> {review.coverage ? (review.coverage.meets_minimum && !review.coverage.missing_month_count ? '24-month count met; model and data quality checks still apply.' : 'review month coverage before forecasting.') : 'depends on the selected date, target, frequency and available history.'}</p>
            <p><strong>Customer analytics:</strong> {review.coverage ? 'monthly aggregates do not establish individual purchase histories needed for next purchase, churn, recommendations, or individual customer segmentation.' : 'requires suitable individual customer histories.'} <strong>Profit:</strong> requires verified financial measures and confirmed assumptions.</p>
          </div>}
        </CardContent>
      </Card>

      <Card className="group overflow-hidden border-border/70 bg-card/80 shadow-[0_18px_48px_-36px_rgba(15,23,42,0.22)] transition-all duration-500 hover:border-primary/30 hover:shadow-[0_28px_65px_-34px_rgba(37,99,235,0.3)]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4 text-primary transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3" /> Data Preview</CardTitle>
          <CardDescription>{previewMode === 'sample' ? `${previewRows.length} evenly spaced rows from the loaded view` : `Rows ${data.length ? activePage * 10 + 1 : 0}–${Math.min((activePage + 1) * 10, data.length)} of ${data.length.toLocaleString()} loaded rows`}{previewLoaded ? ` (preview of ${totalRows.toLocaleString()} total rows)` : ''}.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-3 flex flex-wrap items-center gap-2"><label className="text-sm">Find columns <input className="ml-2 rounded border bg-background p-2" value={columnSearch} onChange={(event) => setColumnSearch(event.target.value)} placeholder="Column name" /></label><Button variant="outline" size="sm" onClick={() => setPreviewMode('pages')}>Pages</Button><Button variant="outline" size="sm" onClick={() => setPreviewMode('sample')}>Spread sample</Button><Button variant="outline" size="sm" disabled={previewMode === 'sample' || activePage === 0} onClick={() => setPage(activePage - 1)}>Previous</Button><span className="text-sm">Page {activePage + 1} of {maxPage + 1}</span><Button variant="outline" size="sm" disabled={previewMode === 'sample' || activePage >= maxPage} onClick={() => setPage(activePage + 1)}>Next</Button></div>
          <div className="max-h-[420px] overflow-auto rounded-2xl border border-border/70 bg-background/70 shadow-inner transition-all duration-500 group-hover:border-primary/20">
            <ShadTable>
              <TableHeader>
                <TableRow>
                  <TableHead>SNo</TableHead>
                  {displayColumns.map((col) => <TableHead key={col.name}>{col.name}</TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewRows.map((row, rowIdx) => (
                  <TableRow key={rowIdx} className="transition-all duration-300 hover:bg-primary/5 hover:shadow-[inset_3px_0_0_rgba(37,99,235,0.8)]">
                    <TableCell className="font-medium text-muted-foreground transition-colors duration-300 group-hover:text-foreground/80">{previewIndices[rowIdx] + 1}</TableCell>
                    {displayColumns.map((col) => <TableCell key={`${rowIdx}-${col.name}`} className="transition-colors duration-300 hover:text-foreground">{row[col.name] == null ? '-' : String(row[col.name])}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </ShadTable>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
