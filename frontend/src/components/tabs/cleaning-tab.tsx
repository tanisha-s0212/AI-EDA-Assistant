'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import {
  Copy,
  Droplets,
  Calendar,
  Type,
  Sparkles,
  CheckCircle2,
  Loader2,
  Database,
  ChevronDown,
  ChevronRight,
  Download,
  ShieldCheck,
  Clock,
  Trash2,
  ArrowRight,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
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
import { useAppStore, type DataRow, type ColumnInfo, type CleaningLog } from '@/lib/store';
import { toast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';

// ─────────────────────────────────────────────
// Animation variants
// ─────────────────────────────────────────────
const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 16, scale: 0.97 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { type: 'spring', stiffness: 260, damping: 22 },
  },
};

// ─────────────────────────────────────────────
// Cleaning operation definitions
// ─────────────────────────────────────────────
interface CleaningOperation {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  enabled: boolean;
  effectText: string;
  hoverDetails: string[];
}

// ─────────────────────────────────────────────
// Cleaning functions
// ─────────────────────────────────────────────
const removeDuplicates = (data: DataRow[]): { cleaned: DataRow[]; log: CleaningLog } => {
  const seen = new Set<string>();
  const cleaned = data.filter((row) => {
    const key = JSON.stringify(row);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return {
    cleaned,
    log: {
      action: 'Remove Duplicates',
      detail: `Removed ${data.length - cleaned.length} duplicate rows`,
      timestamp: new Date().toLocaleTimeString(),
    },
  };
};

const inferDtype = (values: unknown[]): ColumnInfo['dtype'] => {
  const nonEmpty = values.filter((value) => value !== null && value !== undefined && value !== '');
  if (nonEmpty.length === 0) return 'string';

  const boolValues = new Set(nonEmpty.map((value) => String(value).toLowerCase()));
  if ([...boolValues].every((value) => ['true', 'false', '1', '0', 'yes', 'no'].includes(value)) && boolValues.size <= 2) {
    return 'boolean';
  }

  const dateRegex = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/;
  const dateCount = nonEmpty.filter((value) => dateRegex.test(String(value))).length;
  if (dateCount / nonEmpty.length > 0.8) return 'datetime';

  const numericCount = nonEmpty.filter((value) => typeof value === 'number' || (!Number.isNaN(Number(value)) && String(value).trim() !== '')).length;
  if (numericCount / nonEmpty.length > 0.8) return 'number';

  return 'string';
};

const inferRole = (dtype: ColumnInfo['dtype'], name: string, uniqueCount: number, totalRows: number): ColumnInfo['role'] => {
  if (name.toLowerCase().includes('id') || (uniqueCount === totalRows && totalRows > 1)) return 'identifier';
  if (dtype === 'number') return 'numeric';
  if (dtype === 'boolean') return 'boolean';
  if (dtype === 'datetime') return 'datetime';
  return 'categorical';
};

const buildColumnInfo = (data: DataRow[]): ColumnInfo[] => {
  if (!data.length) return [];
  const columnNames = Object.keys(data[0]);
  const totalRows = data.length;

  return columnNames.map((name) => {
    const values = data.map((row) => row[name]);
    const nonNull = values.filter((value) => value !== null && value !== undefined && value !== '').length;
    const uniqueCount = new Set(values.map((value) => String(value))).size;
    const dtype = inferDtype(values);
    const role = inferRole(dtype, name, uniqueCount, totalRows);

    return {
      name,
      dtype,
      nonNull,
      nullCount: totalRows - nonNull,
      uniqueCount,
      role,
      sample: values.slice(0, 5).map((value) => String(value)),
    };
  });
};

const handleMissingValues = (
  data: DataRow[],
  columns: ColumnInfo[]
): { cleaned: DataRow[]; log: CleaningLog } => {
  const cleaned = data.map((row) => ({ ...row }));
  const details: string[] = [];

  columns.forEach((col) => {
    if (col.nullCount > 0) {
      const values = data
        .map((r) => r[col.name])
        .filter((v) => v !== null && v !== undefined && v !== '');
      if (values.length === 0) return;

      const fillValue =
        col.role === 'numeric'
          ? values.sort((a, b) => Number(a) - Number(b))[Math.floor(values.length / 2)]
          : values
              .sort((a, b) => String(a).localeCompare(String(b)))
              .reduce((a, b, _i, arr) => {
                const count = arr.filter((v) => v === a).length;
                const countB = arr.filter((v) => v === b).length;
                return count >= countB ? a : b;
              }, values[0]);

      cleaned.forEach((row) => {
        if (row[col.name] === null || row[col.name] === undefined || row[col.name] === '') {
          row[col.name] = fillValue;
        }
      });
      details.push(`\`${col.name}\`: filled ${col.nullCount} values`);
    }
  });

  return {
    cleaned,
    log: {
      action: 'Handle Missing Values',
      detail: details.length > 0 ? details.join('; ') : 'No missing values found',
      timestamp: new Date().toLocaleTimeString(),
    },
  };
};

const convertDates = (
  data: DataRow[],
  columns: ColumnInfo[]
): { cleaned: DataRow[]; log: CleaningLog } => {
  const dateCols = columns.filter(
    (c) => c.role === 'datetime' || (c.dtype === 'string' && c.sample.length > 0)
  );
  const converted: string[] = [];

  dateCols.forEach((col) => {
    const sample = col.sample[0];
    if (sample && !isNaN(Date.parse(sample)) && col.role !== 'datetime') {
      converted.push(col.name);
    }
  });

  return {
    cleaned: data,
    log: {
      action: 'Convert Dates',
      detail:
        converted.length > 0
          ? `Identified ${converted.length} date columns: ${converted.join(', ')}`
          : 'No additional date columns detected',
      timestamp: new Date().toLocaleTimeString(),
    },
  };
};

const standardizeNames = (
  data: DataRow[],
  columns: ColumnInfo[]
): { cleaned: DataRow[]; log: CleaningLog } => {
  const nameMap: Record<string, string> = {};
  const cleaned = data.map((row) => {
    const newRow: DataRow = {};
    Object.keys(row).forEach((key) => {
      const newKey = key
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_|_$/g, '');
      nameMap[key] = newKey;
      newRow[newKey] = row[key];
    });
    return newRow;
  });

  return {
    cleaned,
    log: {
      action: 'Standardize Names',
      detail: `Renamed ${Object.keys(nameMap).length} columns to snake_case`,
      timestamp: new Date().toLocaleTimeString(),
    },
  };
};

// ─────────────────────────────────────────────
// Typewriter component
// ─────────────────────────────────────────────
function TypewriterDisplay({ text, speed = 4 }: { text: string | null; speed?: number }) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!text) return;
    let index = 0;
    const loop = setInterval(() => {
      index += 2;
      if (index >= text.length) {
        setDisplayed(text);
        setDone(true);
        clearInterval(loop);
      } else {
        setDisplayed(text.slice(0, index));
      }
    }, speed);
    return () => clearInterval(loop);
  }, [text, speed]);

  return (
    <div className="relative">
      {!done && (
        <span className="ml-0.5 inline-block h-5 w-0.5 animate-pulse align-middle bg-primary" />
      )}
      <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-primary prose-p:text-muted-foreground prose-li:text-muted-foreground prose-strong:text-foreground">
        <ReactMarkdown>{displayed}</ReactMarkdown>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────
export default function CleaningTab() {
  const datasetId = useAppStore((s) => s.datasetId);
  const fileName = useAppStore((s) => s.fileName);
  const previewLoaded = useAppStore((s) => s.previewLoaded);
  const loadedRowCount = useAppStore((s) => s.loadedRowCount);
  const cleanedRowCount = useAppStore((s) => s.cleanedRowCount);
  const rawData = useAppStore((s) => s.rawData);
  const columns = useAppStore((s) => s.columns);
  const totalRows = useAppStore((s) => s.totalRows);
  const duplicates = useAppStore((s) => s.duplicates);
  const cleanedData = useAppStore((s) => s.cleanedData);
  const cleaningLogs = useAppStore((s) => s.cleaningLogs);
  const cleaningDone = useAppStore((s) => s.cleaningDone);

  // Operation enable states
  const [ops, setOps] = useState<Record<string, boolean>>({
    removeDuplicates: true,
    handleMissing: true,
    convertDates: true,
    standardizeNames: true,
  });

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);

  // AI justification state
  const [aiJustification, setAiJustification] = useState<string | null>(null);
  const [isGeneratingAI, setIsGeneratingAI] = useState(false);
  const [justificationKey, setJustificationKey] = useState(0);

  const hasData = rawData !== null && columns.length > 0;

  // ── Compute operation effect texts
  const columnsWithMissing = useMemo(
    () => columns.filter((c) => c.nullCount > 0),
    [columns]
  );

  const dateColumnsDetected = useMemo(() => {
    return columns.filter(
      (c) =>
        (c.dtype === 'string' && c.sample.length > 0 && !isNaN(Date.parse(c.sample[0]))) ||
        c.role === 'datetime'
    );
  }, [columns]);

  const needsStandardizing = useMemo(() => {
    return columns.some(
      (c) => c.name !== c.name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
    );
  }, [columns]);

  // Operation cards data
  const operationCards: CleaningOperation[] = useMemo(
    () => [
      {
        id: 'removeDuplicates',
        title: 'Remove Duplicates',
        description: 'Detect and remove exact duplicate rows from the dataset to prevent skewed analysis.',
        icon: Copy,
        enabled: ops.removeDuplicates,
        effectText:
          duplicates > 0
            ? `Will remove ${duplicates} duplicate row${duplicates !== 1 ? 's' : ''}`
            : 'No duplicates detected',
        hoverDetails: duplicates > 0
          ? [
              `${duplicates.toLocaleString()} repeated row${duplicates !== 1 ? 's are' : ' is'} currently present in the uploaded dataset.`,
              'Removing exact duplicates helps keep counts, averages, and model learning from being biased by repeated records.',
            ]
          : [
              'No repeated full-row records were found in the current dataset sample.',
              'Leaving this enabled is still safe when a new dataset is uploaded into the same workspace.',
            ],
      },
      {
        id: 'handleMissing',
        title: 'Handle Missing Values',
        description: 'Impute missing values using mode for categorical and median for numeric columns.',
        icon: Droplets,
        enabled: ops.handleMissing,
        effectText:
          columnsWithMissing.length > 0
            ? `Will fill values in ${columnsWithMissing.length} column${columnsWithMissing.length !== 1 ? 's' : ''}`
            : 'No missing values detected',
        hoverDetails: columnsWithMissing.length > 0
          ? [
              `${columnsWithMissing.length.toLocaleString()} column${columnsWithMissing.length !== 1 ? 's have' : ' has'} missing values in scope.`,
              'Numeric fields use a median-style fill, while categorical fields use the most frequent observed value.',
            ]
          : [
              'No null or empty values were detected in the active cleaning scope.',
              'This option can remain enabled without changing already complete columns.',
            ],
      },
      {
        id: 'convertDates',
        title: 'Convert Date Columns',
        description: 'Identify and parse date-like string columns into proper datetime format.',
        icon: Calendar,
        enabled: ops.convertDates,
        effectText:
          dateColumnsDetected.length > 0
            ? `${dateColumnsDetected.length} date column${dateColumnsDetected.length !== 1 ? 's' : ''} detected`
            : 'No date columns detected',
        hoverDetails: dateColumnsDetected.length > 0
          ? [
              `${dateColumnsDetected.length.toLocaleString()} column${dateColumnsDetected.length !== 1 ? 's appear' : ' appears'} to contain calendar or timestamp values.`,
              'Converting them improves sorting, time-based grouping, forecasting, and downstream feature engineering.',
            ]
          : [
              'No strong date-like patterns were detected from the profiled values.',
              'If later uploads contain time fields, this option helps standardize them automatically.',
            ],
      },
      {
        id: 'standardizeNames',
        title: 'Standardize Column Names',
        description: 'Remove spaces, special characters, and convert to consistent snake_case format.',
        icon: Type,
        enabled: ops.standardizeNames,
        effectText: needsStandardizing ? 'Column name changes available' : 'Names already standardized',
        hoverDetails: needsStandardizing
          ? [
              'Some column names still contain spaces, casing differences, or special characters.',
              'Standardizing to snake_case reduces errors in analysis code, filters, and model feature mapping.',
            ]
          : [
              'The current column names already follow a consistent machine-friendly format.',
              'Keeping this enabled preserves naming consistency for future uploads in the workspace.',
            ],
      },
    ],
    [ops, duplicates, columnsWithMissing, dateColumnsDetected, needsStandardizing]
  );

  const toggleOp = (id: string) => {
    setOps((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const enabledCount = Object.values(ops).filter(Boolean).length;
  const uploadContextSummary = useMemo(() => {
    const datasetLabel = fileName ?? 'uploaded dataset';
    const loadedRows = loadedRowCount || rawData?.length || 0;
    const missingCells = columns.reduce((sum, col) => sum + col.nullCount, 0);
    const activeOps = operationCards.filter((op) => op.enabled).map((op) => op.title);

    return {
      datasetLabel,
      loadedRows,
      missingCells,
      activeOps,
      scopeText:
        previewLoaded && totalRows > loadedRows
          ? `${datasetLabel} was uploaded with ${totalRows.toLocaleString()} total rows. This page is previewing ${loadedRows.toLocaleString()} rows while the backend cleaning run still targets the full dataset.`
          : `${datasetLabel} was uploaded with ${totalRows.toLocaleString()} rows and is being cleaned directly from the active workspace.`,
    };
  }, [columns, fileName, loadedRowCount, operationCards, previewLoaded, rawData?.length, totalRows]);

  // ── Run cleaning
  const handleClean = useCallback(async () => {
    if (!rawData || isProcessing) return;

    setIsProcessing(true);

    try {
      if (datasetId) {
        const response = await apiClient.post('/clean-dataset', {
          dataset_id: datasetId,
          remove_duplicates: ops.removeDuplicates,
          handle_missing: ops.handleMissing,
          convert_dates: ops.convertDates,
          standardize_names: ops.standardizeNames,
        });

        const result = response.data as {
          data: DataRow[];
          columns: ColumnInfo[];
          logs: CleaningLog[];
          rowCount: number;
          loadedRowCount: number;
          previewLoaded: boolean;
          duplicates: number;
        };

        useAppStore.setState({
          rawData: result.data,
          cleanedData: result.data,
          columns: result.columns,
          cleaningLogs: result.logs,
          cleaningDone: true,
          cleanedRowCount: result.rowCount,
          loadedRowCount: result.loadedRowCount ?? result.data.length,
          previewLoaded: !!result.previewLoaded,
          duplicates: result.duplicates ?? 0,
          targetColumn: null,
          selectedFeatures: [],
          selectedModel: null,
          modelId: null,
          modelMetrics: null,
          modelTrained: false,
          featureImportance: null,
          predictionResult: null,
          predictionAnalysis: null,
          predictionProbabilities: null,
          predictionHistory: [],
          timeSeriesForecastResult: null,
          mlForecastResult: null,
          reportGenerated: false,
          reportUrl: null,
          aiInsights: null,
        });

        setLogsOpen(true);
        toast({
          title: 'Cleaning Complete',
          description: `${result.logs.length} operation${result.logs.length !== 1 ? 's' : ''} applied across ${result.rowCount.toLocaleString()} rows on the full dataset.`,
        });
        return;
      }

      const logs: CleaningLog[] = [];
      let currentData = [...rawData];

      await new Promise((r) => setTimeout(r, 300));

      if (ops.removeDuplicates) {
        const result = removeDuplicates(currentData);
        currentData = result.cleaned;
        logs.push(result.log);
      }

      await new Promise((r) => setTimeout(r, 200));

      if (ops.handleMissing) {
        const result = handleMissingValues(currentData, columns);
        currentData = result.cleaned;
        logs.push(result.log);
      }

      await new Promise((r) => setTimeout(r, 200));

      if (ops.convertDates) {
        const result = convertDates(currentData, columns);
        currentData = result.cleaned;
        logs.push(result.log);
      }

      await new Promise((r) => setTimeout(r, 200));

      if (ops.standardizeNames) {
        const result = standardizeNames(currentData, columns);
        currentData = result.cleaned;
        logs.push(result.log);
      }

      useAppStore.setState({
        rawData: currentData,
        cleanedData: currentData,
        columns: buildColumnInfo(currentData),
        cleaningLogs: logs,
        cleaningDone: true,
        cleanedRowCount: currentData.length,
        duplicates: ops.removeDuplicates ? 0 : duplicates,
        targetColumn: null,
        selectedFeatures: [],
        selectedModel: null,
        modelId: null,
        modelMetrics: null,
        modelTrained: false,
        featureImportance: null,
        predictionResult: null,
        predictionAnalysis: null,
        predictionProbabilities: null,
        predictionHistory: [],
        timeSeriesForecastResult: null,
        mlForecastResult: null,
        reportGenerated: false,
        reportUrl: null,
        aiInsights: null,
      });

      setLogsOpen(true);
      toast({
        title: 'Cleaning Complete',
        description: `${logs.length} operation${logs.length !== 1 ? 's' : ''} applied successfully. ${currentData.length.toLocaleString()} rows remaining.`,
      });
    } catch (error) {
      toast({
        title: 'Cleaning failed',
        description: getApiErrorMessage(error, 'The cleaning workflow could not be completed for this dataset.'),
        variant: 'destructive',
      });
    } finally {
      setIsProcessing(false);
    }
  }, [rawData, columns, ops, isProcessing, datasetId, duplicates]);

  // ── AI Justification
  const handleGenerateJustification = useCallback(async () => {
    if (cleaningLogs.length === 0 || isGeneratingAI) return;

    setIsGeneratingAI(true);
    setAiJustification(null);

    try {
      const response = await apiClient.post('/cleaning-justification', {
        logs: cleaningLogs,
        totalRows: cleanedRowCount ?? totalRows ?? rawData?.length ?? 0,
        totalColumns: columns.length,
        fileName,
        loadedRowCount: loadedRowCount ?? rawData?.length ?? 0,
        previewLoaded,
      });

      const data = response.data;
      setAiJustification(data.justification);
      setJustificationKey((k) => k + 1);
    } catch {
      // Fallback to local generation if API fails
      const fallbackJustification = generateFallbackJustification(
        cleaningLogs,
        cleanedRowCount ?? totalRows ?? rawData?.length ?? 0,
        columns.length,
        {
          fileName,
          loadedRowCount: loadedRowCount ?? rawData?.length ?? 0,
          previewLoaded,
        }
      );
      setAiJustification(fallbackJustification);
      setJustificationKey((k) => k + 1);
    } finally {
      setIsGeneratingAI(false);
    }
  }, [cleaningLogs, rawData, columns, isGeneratingAI, cleanedRowCount, totalRows, fileName, loadedRowCount, previewLoaded]);

  // ── Download CSV
  const handleDownload = useCallback(() => {
    if (!cleanedData || cleanedData.length === 0) return;

    const exportingPreviewOnly = previewLoaded && loadedRowCount < (cleanedRowCount ?? cleanedData.length);

    const colKeys = Object.keys(cleanedData[0]);
    const csvHeader = colKeys.join(',');
    const csvRows = cleanedData.map((row) =>
      colKeys
        .map((key) => {
          const val = row[key];
          const strVal = val === null || val === undefined ? '' : String(val);
          // Escape values that contain commas or quotes
          if (strVal.includes(',') || strVal.includes('"') || strVal.includes('\n')) {
            return `"${strVal.replace(/"/g, '""')}"`;
          }
          return strVal;
        })
        .join(',')
    );
    const csvContent = [csvHeader, ...csvRows].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'cleaned_data.csv';
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: 'Download Started',
      description: exportingPreviewOnly ? `Exported ${cleanedData.length.toLocaleString()} preview rows as CSV.` : `Exported ${cleanedData.length.toLocaleString()} rows as CSV.`,
    });
  }, [cleanedData, previewLoaded, loadedRowCount, cleanedRowCount]);

  // ── Empty state
  if (!hasData) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Data Cleaning</h2>
          <p className="text-muted-foreground mt-1">
            Clean and preprocess your dataset with intelligent automation.
          </p>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Database className="h-8 w-8 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold">No Data Loaded</p>
              <p className="text-sm text-muted-foreground mt-1">
                Upload a dataset in the Upload tab to access cleaning operations.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Cleaned data preview columns
  const previewRows = cleanedData ? cleanedData.slice(0, 5) : [];
  const previewColumns = cleanedData && cleanedData.length > 0 ? Object.keys(cleanedData[0]) : columns.map((c) => c.name);
  const processedRowCount = cleanedRowCount ?? cleanedData?.length ?? rawData?.length ?? 0;
  const removedRowCount = Math.max(0, totalRows - processedRowCount);
  const showingPreviewOnly = previewLoaded && loadedRowCount < processedRowCount;

  return (
    <motion.div
      className="space-y-6"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* Section Header */}
      <motion.div variants={itemVariants}>
        <h2 className="text-2xl font-bold tracking-tight">Data Cleaning</h2>
        <p className="text-muted-foreground mt-1">
          Clean and preprocess your dataset with intelligent automation. Toggle operations and click Clean Data.
        </p>
        {datasetId && (
          <p className="mt-2 text-xs text-primary">
            Cleaning runs on the full dataset on the backend. This page shows a preview only.
          </p>
        )}
      </motion.div>

      <motion.div variants={itemVariants}>
        <Card className="border-primary/15 bg-primary/5">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15">
                <ShieldCheck className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle className="text-lg">Cleaning Explainability</CardTitle>
                <CardDescription>Why the selected cleaning actions matter for this uploaded dataset</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 pt-0">
            <div className="rounded-xl border border-primary/15 bg-background/70 px-4 py-3 text-sm leading-6 text-muted-foreground">
              {uploadContextSummary.scopeText}
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-xl border bg-background/70 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Upload Quality Snapshot</p>
                <p className="mt-2 text-sm text-foreground">{duplicates.toLocaleString()} duplicate rows, {uploadContextSummary.missingCells.toLocaleString()} missing cells, and {columns.length.toLocaleString()} profiled columns are currently in scope.</p>
              </div>
              <div className="rounded-xl border bg-background/70 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Selected Actions</p>
                <p className="mt-2 text-sm text-foreground">
                  {uploadContextSummary.activeOps.length > 0
                    ? `${uploadContextSummary.activeOps.join(', ')} will be applied to stabilize the uploaded dataset before downstream analysis.`
                    : 'No cleaning actions are currently selected.'}
                </p>
              </div>
              <div className="rounded-xl border bg-background/70 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Why It Helps</p>
                <p className="mt-2 text-sm text-foreground">These steps reduce avoidable noise so EDA, forecasting, and ML training reflect the dataset’s real structure instead of upload artifacts.</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ───── 1. Cleaning Operations Panel ───── */}
      <motion.div
        className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4"
        variants={containerVariants}
      >
        {operationCards.map((op) => {
          const hasEffect =
            (op.id === 'removeDuplicates' && duplicates > 0) ||
            (op.id === 'handleMissing' && columnsWithMissing.length > 0) ||
            (op.id === 'convertDates' && dateColumnsDetected.length > 0) ||
            (op.id === 'standardizeNames' && needsStandardizing);

          return (
            <motion.div key={op.id} variants={itemVariants}>
              <HoverCard openDelay={160} closeDelay={120}>
              <HoverCardTrigger asChild>
              <Card
                className={`relative overflow-hidden transition-all duration-300 ${
                  op.enabled
                    ? 'border-primary/30 shadow-sm shadow-primary/10 hover:-translate-y-1 hover:shadow-[0_20px_50px_-30px_rgba(37,99,235,0.35)]'
                    : 'border-border/50 opacity-70 hover:border-border hover:opacity-100'
                }`}
              >
                {/* Active indicator bar */}
                <AnimatePresence>
                  {op.enabled && (
                    <motion.div
                      initial={{ scaleY: 0 }}
                      animate={{ scaleY: 1 }}
                      exit={{ scaleY: 0 }}
                      className="absolute top-0 left-0 right-0 h-[2px] bg-primary origin-left"
                    />
                  )}
                </AnimatePresence>

                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <div
                        className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
                          op.enabled
                            ? 'bg-primary/15'
                            : 'bg-muted'
                        }`}
                      >
                        <op.icon
                          className={`h-5 w-5 transition-colors ${
                            op.enabled
                              ? 'text-primary'
                              : 'text-muted-foreground'
                          }`}
                        />
                      </div>
                      <div>
                        <CardTitle className="text-sm font-semibold">{op.title}</CardTitle>
                      </div>
                    </div>
                    <Switch
                      checked={op.enabled}
                      onCheckedChange={() => toggleOp(op.id)}
                      className="data-[state=checked]:bg-primary"
                    />
                  </div>
                </CardHeader>
                <CardContent className="pt-0 space-y-3">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {op.description}
                  </p>
                  <div
                    className={`flex items-center gap-1.5 text-xs font-medium rounded-full px-2.5 py-1 w-fit ${
                      hasEffect
                        ? 'bg-secondary text-secondary-foreground'
                        : 'bg-primary/10 text-primary'
                    }`}
                  >
                    {hasEffect ? (
                      <ShieldCheck className="h-3 w-3" />
                    ) : (
                      <CheckCircle2 className="h-3 w-3" />
                    )}
                    {op.effectText}
                  </div>
                </CardContent>
              </Card>
              </HoverCardTrigger>
              <HoverCardContent align="start" className="w-[320px] rounded-2xl border-border/70 bg-popover/98 p-4 shadow-[0_24px_60px_-32px_rgba(15,23,42,0.35)]">
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-foreground">{op.title}</p>
                  {op.hoverDetails.map((detail) => (
                    <p key={detail} className="text-xs leading-6 text-muted-foreground">
                      {detail}
                    </p>
                  ))}
                </div>
              </HoverCardContent>
              </HoverCard>
            </motion.div>
          );
        })}
      </motion.div>

      {/* ───── 2. Clean Data Button ───── */}
      <motion.div variants={itemVariants} className="flex justify-center">
        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button
            onClick={handleClean}
            disabled={isProcessing || enabledCount === 0}
            size="lg"
            className={`relative gap-2.5 px-8 py-6 text-base font-semibold shadow-lg transition-all ${
              isProcessing
                ? 'bg-primary/80 cursor-wait'
                : cleaningDone
                  ? 'bg-primary hover:bg-primary/90 shadow-primary/25'
                  : 'bg-primary hover:bg-primary/90 shadow-primary/20'
            } text-primary-foreground`}
          >
            <AnimatePresence mode="wait">
              {isProcessing ? (
                <motion.span
                  key="loading"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex items-center gap-2.5"
                >
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Cleaning Data...
                </motion.span>
              ) : cleaningDone ? (
                <motion.span
                  key="done"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex items-center gap-2.5"
                >
                  <CheckCircle2 className="h-5 w-5" />
                  Re-clean Data ({enabledCount} operation{enabledCount !== 1 ? 's' : ''})
                </motion.span>
              ) : (
                <motion.span
                  key="start"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex items-center gap-2.5"
                >
                  <Sparkles className="h-5 w-5" />
                  Clean Data ({enabledCount} operation{enabledCount !== 1 ? 's' : ''})
                </motion.span>
              )}
            </AnimatePresence>
          </Button>
        </motion.div>
      </motion.div>

      {/* Success Banner */}
      <AnimatePresence>
        {cleaningDone && !isProcessing && cleanedData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 shrink-0">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-primary">
                    Data Cleaned Successfully
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {cleaningLogs.length} operations applied &middot;{' '}
                    <span className="font-semibold">{processedRowCount.toLocaleString()}</span> rows remaining
                    {removedRowCount > 0 && (
                      <> &middot; <span className="font-semibold text-foreground">{removedRowCount.toLocaleString()} rows removed</span></>
                    )}
                    {showingPreviewOnly && (
                      <> &middot; Previewing first {loadedRowCount.toLocaleString()} cleaned rows</>
                    )}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => useAppStore.getState().setActiveTab('forecast_ts')}
                  className="gap-1.5 border-primary/30 text-primary hover:bg-primary/10 shrink-0"
                >
                  Next Step
                  <ArrowRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ───── 3. Cleaning Logs Panel ───── */}
      <motion.div variants={itemVariants}>
        <Collapsible open={logsOpen} onOpenChange={setLogsOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader
                className="pb-3 cursor-pointer hover:bg-muted/30 transition-colors rounded-t-lg"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500/15">
                      <Clock className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                    </div>
                    <div className="text-left">
                      <CardTitle className="text-lg">Cleaning Logs</CardTitle>
                      <CardDescription className="mt-0.5">
                        {cleaningLogs.length > 0
                          ? `${cleaningLogs.length} operation${cleaningLogs.length !== 1 ? 's' : ''} recorded`
                          : 'No operations performed yet'}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {cleaningLogs.length > 0 && (
                      <Badge variant="secondary" className="font-normal">
                        {cleaningLogs.length}
                      </Badge>
                    )}
                    <motion.div
                      animate={{ rotate: logsOpen ? 90 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    </motion.div>
                  </div>
                </div>
              </CardHeader>
            </CollapsibleTrigger>

            <CollapsibleContent>
              <CardContent className="pt-0">
                <Separator className="mb-4" />
                {cleaningLogs.length > 0 ? (
                  <div className="max-h-64 overflow-y-auto pr-2">
                    <div className="space-y-3">
                      {cleaningLogs.map((log, idx) => (
                        <motion.div
                          key={`${log.action}-${log.timestamp}-${idx}`}
                          initial={{ opacity: 0, x: -12 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="flex items-start gap-3 rounded-lg border border-border/50 bg-muted/20 p-3"
                        >
                          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/15">
                            <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-sm font-semibold">{log.action}</span>
                              <span className="text-[11px] text-muted-foreground font-mono">
                                {log.timestamp}
                              </span>
                            </div>
                            <p className="mt-1 break-words text-xs leading-relaxed text-muted-foreground">
                              {log.detail}
                            </p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-8 text-center">
                    <Trash2 className="h-8 w-8 text-muted-foreground/40 mb-2" />
                    <p className="text-sm text-muted-foreground">
                      No cleaning operations have been performed yet.
                    </p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Enable operations above and click &quot;Clean Data&quot; to get started.
                    </p>
                  </div>
                )}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </motion.div>

      {/* ───── 4. AI Cleaning Justification ───── */}
      <AnimatePresence>
        {cleaningDone && cleaningLogs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            variants={itemVariants}
          >
            <Card className="overflow-hidden border-primary/20">
              <div className="absolute top-0 left-0 right-0 h-[2px] bg-primary" />
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15">
                      <Sparkles className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">AI Cleaning Justification</CardTitle>
                      <CardDescription className="mt-0.5">
                        Understand why each cleaning step was important
                      </CardDescription>
                    </div>
                  </div>
                  <Button
                    onClick={handleGenerateJustification}
                    disabled={isGeneratingAI}
                    variant="outline"
                    size="sm"
                    className="gap-1.5 border-primary/30 text-primary hover:bg-primary/10"
                  >
                    {isGeneratingAI ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Analyzing...
                      </>
                    ) : aiJustification ? (
                      <>
                        <Sparkles className="h-3.5 w-3.5" />
                        Regenerate
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-3.5 w-3.5" />
                        Generate Justification
                      </>
                    )}
                  </Button>
                </div>
              </CardHeader>

              {aiJustification && (
                <CardContent className="pt-0">
                  <div className="max-h-[300px] overflow-y-auto rounded-lg border border-border bg-muted/30 p-5 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:rounded-full [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-primary/30">
                    <TypewriterDisplay key={justificationKey} text={aiJustification} />
                  </div>
                </CardContent>
              )}
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ───── 5. Cleaned Data Preview ───── */}
      <AnimatePresence>
        {cleanedData && cleanedData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.4, delay: 0.3 }}
            variants={itemVariants}
          >
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-secondary">
                      <Database className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">Cleaned Data Preview</CardTitle>
                      <CardDescription className="mt-0.5">
                        First {Math.min(5, loadedRowCount || cleanedData.length)} of {processedRowCount.toLocaleString()} cleaned rows{showingPreviewOnly ? ' (preview)' : ''}
                      </CardDescription>
                    </div>
                  </div>
                  <Button
                    onClick={handleDownload}
                    variant="outline"
                    size="sm"
                    className="gap-1.5 border-primary/30 text-primary hover:bg-primary/10"
                  >
                    <Download className="h-3.5 w-3.5" />
                    {showingPreviewOnly ? 'Download Preview CSV' : 'Download Cleaned Data'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="rounded-lg border max-h-[320px] overflow-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:rounded-full [&::-webkit-scrollbar-thumb]:bg-zinc-300/60 dark:[&::-webkit-scrollbar-thumb]:bg-zinc-700/60 [&::-webkit-scrollbar-thumb]:rounded-full">
                  <ShadTable>
                    <TableHeader>
                      <TableRow className="bg-muted/40 hover:bg-muted/40">
                        <TableHead className="font-semibold text-muted-foreground w-12 text-center">
                          SNo
                        </TableHead>
                        {previewColumns.map((col) => (
                          <TableHead key={col} className="font-semibold whitespace-nowrap">
                            {col}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {previewRows.map((row, rowIdx) => (
                        <TableRow key={rowIdx}>
                          <TableCell className="text-center text-muted-foreground text-xs tabular-nums">
                            {rowIdx + 1}
                          </TableCell>
                          {previewColumns.map((col) => {
                            const val = row[col];
                            const display =
                              val === null || val === undefined
                                ? '—'
                                : String(val).length > 40
                                  ? String(val).slice(0, 40) + '…'
                                  : String(val);
                            return (
                              <TableCell
                                key={col}
                                className={
                                  val === null || val === undefined
                                    ? 'text-muted-foreground italic'
                                    : 'tabular-nums'
                                }
                                title={
                                  val !== null && val !== undefined ? String(val) : undefined
                                }
                              >
                                {display}
                              </TableCell>
                            );
                          })}
                        </TableRow>
                      ))}
                    </TableBody>
                  </ShadTable>
                </div>
                {processedRowCount > 5 && (
                  <div className="flex items-center justify-center pt-3">
                    <Badge variant="secondary" className="font-normal text-muted-foreground">
                      Showing 5 of {processedRowCount.toLocaleString()} rows{showingPreviewOnly ? ' from the cleaned data preview' : ''}
                    </Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ─────────────────────────────────────────────
// Fallback AI justification generator
// ─────────────────────────────────────────────
function generateFallbackJustification(
  logs: CleaningLog[],
  totalRows: number,
  totalColumns: number,
  context?: {
    fileName?: string | null;
    loadedRowCount?: number;
    previewLoaded?: boolean;
  }
): string {
  const sections: string[] = [];
  const datasetLabel = context?.fileName || 'uploaded dataset';
  const loadedRows = context?.loadedRowCount ?? totalRows;
  const scopeLine = context?.previewLoaded && totalRows > loadedRows
    ? `The dataset "${datasetLabel}" was uploaded with ${totalRows} rows, and this page is previewing ${loadedRows} rows while the backend cleaning workflow still applies to the full dataset.\n`
    : `The dataset "${datasetLabel}" was uploaded with ${totalRows} rows available for direct cleaning review.\n`;

  sections.push(
    `### Why These Cleaning Steps Matter\n\n${scopeLine}This dataset (${totalRows} rows × ${totalColumns} columns) required cleaning to ensure reliable analysis and modeling results.\n`
  );

  for (const log of logs) {
    if (log.action === 'Remove Duplicates') {
      const removedCount = parseInt(log.detail.match(/\d+/)?.[0] || '0');
      if (removedCount > 0) {
        sections.push(
          `**Remove Duplicates:** Duplicate entries can inflate statistical measures, bias model training, and lead to overfitting. Removing ${removedCount} duplicates ensures each data point is unique and representative.\n`
        );
      }
    } else if (log.action === 'Handle Missing Values') {
      sections.push(
        `**Handle Missing Values:** Missing values can cause errors in statistical calculations and machine learning algorithms. Imputing with median (numeric) and mode (categorical) preserves the data distribution while maintaining the full dataset for analysis.\n`
      );
    } else if (log.action === 'Convert Dates') {
      sections.push(
        `**Convert Dates:** Proper date formatting enables temporal analysis, time-series modeling, and correct chronological sorting. String dates limit the types of analysis you can perform.\n`
      );
    } else if (log.action === 'Standardize Names') {
      sections.push(
        `**Standardize Column Names:** Consistent naming conventions (snake_case) improve code readability, reduce errors from typos, and make the dataset compatible with a wider range of analysis tools and libraries.\n`
      );
    }
  }

  sections.push(
    `\n> ✅ These steps collectively improve the uploaded dataset quality, reduce noise introduced during ingestion, and prepare the data for accurate exploratory analysis and machine learning modeling.`
  );

  return sections.join('\n');
}




