'use client';

import React, { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import {
  Upload,
  FileSpreadsheet,
  FileText,
  File,
  Database,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Bot,
  ArrowRight,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import { useAppStore, type ColumnInfo, type DataRow, type DatasetWorkspaceDraft } from '@/lib/store';

// ─── Helpers ───────────────────────────────────────────────────────────────────

const MAX_FILE_SIZE = 512 * 1024 * 1024; // 512MB
const MAX_FILE_SIZE_LABEL = '512MB';
const COLUMN_ANALYSIS_SAMPLE_SIZE = 5000;
const DUPLICATE_CHECK_LIMIT = 10000;
const MEMORY_ESTIMATE_SAMPLE_SIZE = 200;
const DATASET_PREVIEW_ROW_LIMIT = 20000;

function normalizeExcelValue(value: unknown): string | number | boolean | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  if (value instanceof Date) return value.toISOString();

  if (Array.isArray(value)) {
    const parts = value
      .map((item) => normalizeExcelValue(item))
      .filter((item): item is string | number | boolean => item !== null);
    return parts.length > 0 ? parts.join(', ') : null;
  }

  if (typeof value === 'object') {
    const candidate = value as {
      text?: string;
      hyperlink?: string;
      result?: unknown;
      richText?: Array<{ text?: string }>;
    };

    if (typeof candidate.text === 'string') return candidate.text;
    if (typeof candidate.hyperlink === 'string') return candidate.hyperlink;
    if (candidate.result !== undefined) return normalizeExcelValue(candidate.result);
    if (Array.isArray(candidate.richText)) {
      const text = candidate.richText.map((part) => part.text ?? '').join('');
      return text || null;
    }
  }

  return String(value);
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function inferDtype(values: (string | number | boolean | null)[]): string {
  const nonEmpty = values.filter((v) => v !== null && v !== undefined && v !== '');
  if (nonEmpty.length === 0) return 'string';

  // Check boolean
  const boolSet = new Set(nonEmpty.map(String).map((v) => v.toLowerCase()));
  if ([...boolSet].every((v) => ['true', 'false', '1', '0', 'yes', 'no'].includes(v)) && boolSet.size <= 2) {
    return 'boolean';
  }

  // Check date
  const dateRegex = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/;
  const dateCount = nonEmpty.filter((v) => dateRegex.test(String(v))).length;
  if (dateCount / nonEmpty.length > 0.8) return 'date';

  // Check number
  const numericCount = nonEmpty.filter((v) => typeof v === 'number' || (!isNaN(Number(v)) && String(v).trim() !== '')).length;
  if (numericCount / nonEmpty.length > 0.8) return 'number';

  return 'string';
}

function inferRole(dtype: string, name: string, uniqueCount: number, totalRows: number): ColumnInfo['role'] {
  if (name.toLowerCase().includes('id') || (uniqueCount === totalRows && totalRows > 1)) return 'identifier';
  if (dtype === 'number') return 'numeric';
  if (dtype === 'boolean') return 'boolean';
  if (dtype === 'date') return 'datetime';
  if (dtype === 'string' && uniqueCount / totalRows < 0.5) return 'categorical';
  return 'categorical';
}

function getAnalysisRows(data: DataRow[], limit = COLUMN_ANALYSIS_SAMPLE_SIZE): DataRow[] {
  if (data.length <= limit) return data;

  const step = Math.max(1, Math.floor(data.length / limit));
  const sampled: DataRow[] = [];
  for (let index = 0; index < data.length && sampled.length < limit; index += step) {
    sampled.push(data[index]);
  }
  return sampled;
}

function buildColumnInfo(data: DataRow[]): ColumnInfo[] {
  if (!data.length) return [];

  const colNames = Object.keys(data[0]);
  const total = data.length;
  const analysisRows = getAnalysisRows(data);

  return colNames.map((name) => {
    const sampleValues = analysisRows.map((row) => row[name]);
    const nonNullSample = sampleValues.filter((value) => value !== null && value !== undefined && value !== '');
    const estimatedNonNull = Math.round((nonNullSample.length / Math.max(sampleValues.length, 1)) * total);
    const unique = new Set(nonNullSample.map(String)).size;
    const dtype = inferDtype(sampleValues);
    const role = inferRole(dtype, name, unique, total);

    return {
      name,
      dtype,
      nonNull: Math.min(total, estimatedNonNull),
      nullCount: Math.max(0, total - estimatedNonNull),
      uniqueCount: unique,
      role,
      sample: nonNullSample.slice(0, 5).map(String),
    };
  });
}

function enrichColumnInfo(columns: ColumnInfo[], data: DataRow[]): ColumnInfo[] {
  if (!columns.length) return columns;
  const analysisRows = getAnalysisRows(data);

  return columns.map((column) => {
    const existingSample = Array.isArray(column.sample) ? column.sample.filter(Boolean) : [];
    if (existingSample.length > 0) return column;

    const derivedSample = analysisRows
      .map((row) => row[column.name])
      .filter((value) => value !== null && value !== undefined && value !== '')
      .slice(0, 5)
      .map(String);

    return {
      ...column,
      sample: derivedSample,
    };
  });
}

function countDuplicates(data: DataRow[]): number {
  if (data.length > DUPLICATE_CHECK_LIMIT) {
    return 0;
  }

  const seen = new Set<string>();
  let dupes = 0;
  for (const row of data) {
    const key = JSON.stringify(row);
    if (seen.has(key)) dupes++;
    else seen.add(key);
  }
  return dupes;
}

function estimateMemory(data: DataRow[]): string {
  if (!data.length) return '0 Bytes';

  const sampleRows = data.slice(0, Math.min(data.length, MEMORY_ESTIMATE_SAMPLE_SIZE));
  const jsonStr = JSON.stringify(sampleRows);
  const avgBytesPerRow = jsonStr.length > 0 ? (jsonStr.length * 2) / sampleRows.length : 0;
  const bytes = avgBytesPerRow * data.length + data.length * 32;
  return formatBytes(bytes);
}

// ─── Sample Data Generator ─────────────────────────────────────────────────────

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

const cardHover = {
  scale: 1.02,
  boxShadow: '0 8px 30px rgba(16, 185, 129, 0.12)',
};

// ─── Component ─────────────────────────────────────────────────────────────────

function buildFreshDatasetState(
  data: DataRow[],
  fileName: string,
  columns: ColumnInfo[],
  options?: { datasetId?: string | null; totalRows?: number; previewLoaded?: boolean; loadedRowCount?: number },
): DatasetWorkspaceDraft {
  return {
    fileName,
    datasetId: options?.datasetId ?? null,
    previewLoaded: options?.previewLoaded ?? false,
    loadedRowCount: options?.loadedRowCount ?? data.length,
    cleanedRowCount: null,
    rawData: data,
    cleanedData: null,
    columns,
    totalRows: options?.totalRows ?? data.length,
    duplicates: countDuplicates(data),
    memoryUsage: estimateMemory(data),
    cleaningLogs: [],
    cleaningDone: false,
    targetColumn: null,
    problemType: 'regression' as const,
    selectedFeatures: [],
    selectedModel: null,
    modelId: null,
    modelMetrics: null,
    modelTrained: false,
    featureImportance: null,
    uploadedModel: null,
    predictionResult: null,
    predictionAnalysis: null,
    predictionProbabilities: null,
    predictionHistory: [],
    timeSeriesForecastResult: null,
    mlForecastResult: null,
    reportGenerated: false,
    reportUrl: null,
    aiInsights: null,
    aiChatHistory: [],
  };
}

export default function UploadTab() {
  const { setActiveTab, addDataset, activeTab, uploadPickerRequestId } = useAppStore();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handledUploadRequestRef = useRef(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const dragCounter = useRef(0);

  // ── Push data into store ──────────────────────────────────────────────────

  const pushToStore = useCallback(
    async (
      data: DataRow[],
      fileName: string,
      options?: { datasetId?: string | null; totalRows?: number; previewLoaded?: boolean; loadedRowCount?: number; columnsOverride?: ColumnInfo[] },
    ) => {
      const columns = options?.columnsOverride ? enrichColumnInfo(options.columnsOverride, data) : buildColumnInfo(data);
      let resolvedOptions = options;

      if (!resolvedOptions?.datasetId && data.length > 0) {
        try {
          const { data: cached } = await apiClient.post('/cache-dataset', {
            file_name: fileName,
            data,
          });
          resolvedOptions = {
            ...resolvedOptions,
            datasetId: cached.datasetId ?? null,
            totalRows: cached.rowCount ?? data.length,
            loadedRowCount: cached.loadedRowCount ?? data.length,
            previewLoaded: !!cached.previewLoaded,
          };
        } catch (error) {
          toast({
            title: 'Dataset cache warning',
            description: getApiErrorMessage(error, 'The dataset could not be cached on the backend. Training may be slower for this upload.'),
            variant: 'destructive',
          });
        }
      }

      window.setTimeout(() => {
        addDataset(buildFreshDatasetState(data, fileName, columns, resolvedOptions));

        setUploadedFileName(fileName);
        const totalRows = resolvedOptions?.totalRows ?? data.length;
        toast({
          title: 'Dataset loaded successfully',
          description: `${totalRows.toLocaleString()} rows x ${columns.length} columns from ${fileName}`,
        });

        setTimeout(() => setActiveTab('understanding'), 800);
      }, 0);
    },
    [addDataset, setActiveTab, toast],
  );

  // ── File Processing ──────────────────────────────────────────────────────

  const processFile = useCallback(
    async (file: File) => {
      if (file.size > MAX_FILE_SIZE) {
        toast({
          title: 'File too large',
          description: `Maximum file size is ${MAX_FILE_SIZE_LABEL}. Your file is ${formatBytes(file.size)}.`,
          variant: 'destructive',
        });
        return;
      }

      setIsProcessing(true);
      const ext = file.name.split('.').pop()?.toLowerCase();

      const supportedExtensions = ['csv', 'tsv', 'xlsx', 'xls', 'parquet'];
      if (!ext || !supportedExtensions.includes(ext)) {
        toast({
          title: 'Unsupported format',
          description: 'Please upload a CSV, TSV, Excel (.xlsx/.xls), or Parquet file.',
          variant: 'destructive',
        });
        setIsProcessing(false);
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      try {
        const { data: result } = await apiClient.post('/parse-dataset', formData);

        if (!result) {
          toast({
            title: 'Upload parse error',
            description: 'Failed to parse the dataset file.',
            variant: 'destructive',
          });
          setIsProcessing(false);
          return;
        }

        if (result.data && result.data.length > 0) {
          await pushToStore(result.data as DataRow[], file.name, {
            datasetId: result.datasetId ?? null,
            totalRows: result.rowCount,
            previewLoaded: !!result.previewLoaded,
            loadedRowCount: result.loadedRowCount ?? result.data.length,
            columnsOverride: Array.isArray(result.columnInfo) ? (result.columnInfo as ColumnInfo[]) : undefined,
          });

          if (result.previewLoaded) {
            toast({
              title: 'Dataset preview loaded',
              description: `Loaded the first ${(result.loadedRowCount ?? DATASET_PREVIEW_ROW_LIMIT).toLocaleString()} of ${Number(result.rowCount ?? result.loadedRowCount ?? DATASET_PREVIEW_ROW_LIMIT).toLocaleString()} rows from ${file.name}. Full-scale data understanding, exploratory data analysis, data cleaning, and training can still use the cached backend dataset.`,
            });
          }
        } else {
          toast({
            title: 'Empty file',
            description: 'No data rows found in the uploaded file.',
            variant: 'destructive',
          });
        }
      } catch (error) {
        toast({
          title: 'Upload parsing error',
          description: getApiErrorMessage(error, 'Failed to send file to server for parsing.'),
          variant: 'destructive',
        });
      }
      setIsProcessing(false);
    },
    [pushToStore, toast],
  );

  // ── Demo Data ────────────────────────────────────────────────────────────

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      dragCounter.current = 0;

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        processFile(files[0]);
      }
    },
    [processFile],
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        processFile(files[0]);
      }
      // Reset input so the same file can be re-selected
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [processFile],
  );

  const handleBrowseClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  useEffect(() => {
    if (activeTab !== 'upload') return;
    if (uploadPickerRequestId === 0 || uploadPickerRequestId === handledUploadRequestRef.current) return;

    handledUploadRequestRef.current = uploadPickerRequestId;
    window.setTimeout(() => {
      fileInputRef.current?.click();
    }, 120);
  }, [activeTab, uploadPickerRequestId]);

  // ── Format Cards Data ────────────────────────────────────────────────────

  const formatCards = useMemo(
    () => [
      {
        title: 'CSV Files',
        icon: FileText,
        color: 'text-primary',
        bgColor: 'bg-primary/10',
        description: 'Comma-separated datasets with automatic type detection and backend preview scaling for larger row counts.',
        badge: '.csv',
      },
      {
        title: 'TSV Files',
        icon: FileText,
        color: 'text-primary',
        bgColor: 'bg-secondary',
        description: 'Tab-separated datasets for export pipelines that preserve commas inside values more cleanly than CSV.',
        badge: '.tsv',
      },
      {
        title: 'Excel Workbooks',
        icon: FileSpreadsheet,
        color: 'text-primary',
        bgColor: 'bg-secondary',
        description: 'Excel workbooks in .xlsx and .xls format, parsed on the backend with first-sheet preview support.',
        badge: '.xlsx/.xls',
      },
      {
        title: 'Parquet Files',
        icon: Database,
        color: 'text-primary',
        bgColor: 'bg-secondary',
        description: 'Parquet columnar datasets with efficient backend parsing for wide schemas and higher row volumes.',
        badge: '.parquet',
      },
    ],
    [],
  );

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="min-w-0 space-y-8"
    >
      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants} className="relative min-w-0 overflow-hidden rounded-2xl border border-border bg-card p-6 sm:p-8 md:p-12">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/8 blur-3xl" />
          <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-secondary blur-3xl" />
        </div>

        <div className="relative flex min-w-0 flex-col items-center gap-6 md:flex-row md:gap-10">
          {/* Animated Robot Icon */}
          <motion.div
            className="flex-shrink-0"
            animate={{
              y: [0, -8, 0],
              rotate: [0, 2, -2, 0],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            <div className="relative">
              <div className="flex h-24 w-24 items-center justify-center rounded-3xl bg-primary shadow-2xl shadow-primary/25 md:h-32 md:w-32">
                <Bot className="h-12 w-12 text-primary-foreground md:h-16 md:w-16" />
              </div>
              <motion.div
                className="absolute -top-1 -right-1 h-6 w-6 rounded-full bg-yellow-400 border-2 border-white dark:border-gray-900 flex items-center justify-center"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Sparkles className="h-3 w-3 text-yellow-800" />
              </motion.div>
            </div>
          </motion.div>

          {/* Hero Text */}
          <div className="min-w-0 flex-1 text-center md:text-left">
            <motion.h1
              className="text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <span className="text-primary">Load Your Dataset</span>
            </motion.h1>
            <motion.p
              className="mt-3 max-w-full text-base text-muted-foreground md:max-w-xl md:text-lg"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.35 }}
            >
              Ingest CSV, Excel, TSV, or Parquet data into a workflow built for data understanding, exploratory data analysis, data cleaning, forecasting, and model development.
              Drag and drop a file or browse from your device to begin.
            </motion.p>
            <motion.div
              className="mt-4 flex flex-wrap justify-center md:justify-start gap-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
                CSV
              </Badge>
              <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
                TSV
              </Badge>
              <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
                Excel (.xlsx/.xls)
              </Badge>
              <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
                Parquet
              </Badge>
              <Badge variant="outline" className="text-muted-foreground">
                Max {MAX_FILE_SIZE_LABEL}
              </Badge>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* ── Upload Zone ───────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <AnimatePresence mode="wait">
          {uploadedFileName && !isProcessing ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.4 }}
              className="flex flex-col items-center justify-center rounded-2xl border-2 border-primary/30 bg-primary/5 p-8 text-center md:p-12"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.1 }}
                className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-primary/10"
              >
                <CheckCircle2 className="h-10 w-10 text-primary" />
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">Dataset Loaded</h3>
              <p className="text-sm text-muted-foreground mb-6">
                <span className="font-medium text-foreground">{uploadedFileName}</span> has been
                loaded successfully. Redirecting to Data Understanding...
              </p>
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                <span>Proceeding to analysis</span>
                <motion.span
                  animate={{ x: [0, 4, 0] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  <ArrowRight className="h-4 w-4" />
                </motion.span>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="dropzone"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <motion.div
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={handleBrowseClick}
                className={`
                  relative min-w-0 cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out
                  ${
                    isDragging
                      ? 'scale-[1.01] border-primary bg-primary/5 shadow-lg shadow-primary/10'
                      : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-accent/50'
                  }
                  ${isProcessing ? 'pointer-events-none opacity-70' : ''}
                `}
              >
                {/* Pulse ring on drag */}
                <AnimatePresence>
                  {isDragging && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="absolute inset-0 rounded-2xl border-2 border-primary/30 pointer-events-none"
                      style={{
                        animation: 'pulse 1.5s ease-in-out infinite',
                      }}
                    />
                  )}
                </AnimatePresence>

                <div className="flex min-w-0 flex-col items-center justify-center px-4 py-14 sm:px-6 md:px-12 md:py-20">
                  <AnimatePresence mode="wait">
                    {isProcessing ? (
                      <motion.div
                        key="processing"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="flex flex-col items-center gap-4"
                      >
                        <Loader2 className="h-14 w-14 animate-spin text-primary" />
                        <div className="text-center">
                          <p className="text-lg font-semibold">Parsing your file...</p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Analyzing columns and data types
                          </p>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="idle"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex flex-col items-center gap-4"
                      >
                        {/* File Icon */}
                        <motion.div
                          animate={isDragging ? { scale: 1.15, y: -4 } : { scale: 1, y: 0 }}
                          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                          className="flex h-20 w-20 items-center justify-center rounded-2xl border border-border bg-secondary"
                        >
                          <motion.div
                            animate={isDragging ? { y: -6 } : { y: 0 }}
                            transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }}
                          >
                            <Upload className="h-10 w-10 text-primary" />
                          </motion.div>
                        </motion.div>

                        {/* Instructions */}
                        <div className="max-w-md text-center">
                          <h3 className="text-xl font-semibold mb-1">
                            {isDragging ? 'Drop your file here' : 'Drag & drop your file here'}
                          </h3>
                          <p className="text-sm text-muted-foreground">
                            or <span className="font-medium text-primary underline underline-offset-2">click to browse</span> from your computer
                          </p>
                        </div>

                        {/* Supported Formats */}
                        <div className="mt-2 flex max-w-full flex-wrap items-center justify-center gap-2 text-center">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">CSV</span>
                          <span className="text-muted-foreground/30">·</span>
                          <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">TSV | Excel (.xlsx/.xls)</span>
                          <span className="text-muted-foreground/30">·</span>
                          <Database className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">Parquet</span>
                        </div>

                        {/* Size limit */}
                        <p className="text-xs text-muted-foreground/60 mt-1">
                          Maximum file size: {MAX_FILE_SIZE_LABEL}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>

              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.xlsx,.xls,.parquet"
                className="hidden"
                onChange={handleFileInputChange}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* ── Demo Data Button ──────────────────────────────────────────────── */}
      {/* ── Format Info Cards ─────────────────────────────────────────────── */}
      <motion.div variants={itemVariants}>
        <div className="flex items-center gap-2 mb-4">
          <File className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-lg font-semibold">Supported Formats</h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {formatCards.map((fmt, idx) => {
            const Icon = fmt.icon;
            return (
              <motion.div
                key={fmt.title}
                whileHover={cardHover}
                transition={{ duration: 0.2 }}
              >
                <Card className="h-full overflow-hidden border-border/50 transition-colors group hover:border-primary/20">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${fmt.bgColor}`}>
                        <Icon className={`h-5 w-5 ${fmt.color}`} />
                      </div>
                      <Badge variant="outline" className="text-xs font-mono">
                        {fmt.badge}
                      </Badge>
                    </div>
                    <CardTitle className="mt-3 text-base transition-colors group-hover:text-primary">
                      {fmt.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-xs leading-relaxed">
                      {fmt.description}
                    </CardDescription>
                    {(fmt.badge === '.parquet' || fmt.badge === '.tsv') && (
                      <div className="mt-3 flex items-center gap-1.5 text-xs text-primary">
                        <Sparkles className="h-3 w-3" />
                        <span>{fmt.badge === '.tsv' ? 'Backend parsing with tab separator detection' : 'Server-side parsing with full type detection'}</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* ── Info Tip ──────────────────────────────────────────────────────── */}
      <motion.div
        variants={itemVariants}
        className="flex flex-col gap-3 rounded-xl border border-border/50 bg-muted/30 p-4 sm:flex-row sm:items-start"
      >
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div>
          <p className="text-sm font-medium">What happens after upload?</p>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            CSV, TSV, Excel, and Parquet files are parsed on the backend so larger datasets can be previewed, cached, and analyzed without overloading the browser. Once the dataset is loaded, the app profiles column roles, null patterns, uniqueness, and potential duplicates, then carries you into data understanding, exploratory data analysis, data cleaning, forecasting, and ML without disrupting the current session state.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}



