'use client';

import React, { useCallback, useRef, useState, useMemo } from 'react';
import ExcelJS from 'exceljs';
import Papa from 'papaparse';
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
import { useAppStore, type ColumnInfo, type DataRow } from '@/lib/store';

// ─── Helpers ───────────────────────────────────────────────────────────────────

const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB
const COLUMN_ANALYSIS_SAMPLE_SIZE = 5000;
const DUPLICATE_CHECK_LIMIT = 10000;
const MEMORY_ESTIMATE_SAMPLE_SIZE = 200;
const PARQUET_PREVIEW_ROW_LIMIT = 20000;

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

function parseExcelWorkbook(worksheet: ExcelJS.Worksheet): DataRow[] {
  const headerRow = worksheet.getRow(1);
  const columnCount = headerRow.cellCount || worksheet.columnCount;

  if (columnCount === 0) return [];

  const headers = Array.from({ length: columnCount }, (_, index) => {
    const headerValue = normalizeExcelValue(headerRow.getCell(index + 1).value);
    return (typeof headerValue === 'string' && headerValue.trim()) || `column_${index + 1}`;
  });

  const rows: DataRow[] = [];

  for (let rowNumber = 2; rowNumber <= worksheet.rowCount; rowNumber++) {
    const row = worksheet.getRow(rowNumber);
    const record: DataRow = {};
    let hasValue = false;

    headers.forEach((header, index) => {
      const value = normalizeExcelValue(row.getCell(index + 1).value);
      record[header] = value;
      if (value !== null && value !== '') {
        hasValue = true;
      }
    });

    if (hasValue) {
      rows.push(record);
    }
  }

  return rows;
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
) {
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
    salesForecastResult: null,
    reportGenerated: false,
    reportUrl: null,
    aiInsights: null,
    aiChatHistory: [],
  };
}

export default function UploadTab() {
  const { setActiveTab } = useAppStore();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
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
        const previousReportUrl = useAppStore.getState().reportUrl;
        if (previousReportUrl) {
          URL.revokeObjectURL(previousReportUrl);
        }

        useAppStore.setState(buildFreshDatasetState(data, fileName, columns, resolvedOptions));

        setUploadedFileName(fileName);
        const totalRows = resolvedOptions?.totalRows ?? data.length;
        toast({
          title: 'Dataset loaded successfully',
          description: `${totalRows.toLocaleString()} rows x ${columns.length} columns from ${fileName}`,
        });

        setTimeout(() => setActiveTab('understanding'), 800);
      }, 0);
    },
    [setActiveTab, toast],
  );

  // ── File Processing ──────────────────────────────────────────────────────

  const processFile = useCallback(
    async (file: File) => {
      if (file.size > MAX_FILE_SIZE) {
        toast({
          title: 'File too large',
          description: `Maximum file size is 200MB. Your file is ${formatBytes(file.size)}.`,
          variant: 'destructive',
        });
        return;
      }

      setIsProcessing(true);
      const ext = file.name.split('.').pop()?.toLowerCase();

      if (ext === 'csv') {
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          complete: (results) => {
            if (results.errors.length > 0) {
              console.warn('CSV parse warnings:', results.errors);
            }
            if (results.data.length === 0) {
              toast({
                title: 'Empty file',
                description: 'No data rows found in the CSV file.',
                variant: 'destructive',
              });
              setIsProcessing(false);
              return;
            }
            void pushToStore(results.data as DataRow[], file.name).finally(() => setIsProcessing(false));
          },
          error: (err) => {
            toast({
              title: 'Parse error',
              description: `Failed to parse CSV: ${err.message}`,
              variant: 'destructive',
            });
            setIsProcessing(false);
          },
        });
      } else if (ext === 'xlsx') {
        try {
          const buffer = await file.arrayBuffer();
          const workbook = new ExcelJS.Workbook();
          await workbook.xlsx.load(buffer);

          const worksheet = workbook.worksheets[0];
          if (!worksheet) {
            toast({
              title: 'Empty file',
              description: 'No worksheet found in the Excel file.',
              variant: 'destructive',
            });
            setIsProcessing(false);
            return;
          }

          const jsonData = parseExcelWorkbook(worksheet);

          if (jsonData.length === 0) {
            toast({
              title: 'Empty file',
              description: 'No data rows found in the Excel file.',
              variant: 'destructive',
            });
            setIsProcessing(false);
            return;
          }

          await pushToStore(jsonData, file.name);
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : 'Unknown error';
          toast({
            title: 'Parse error',
            description: `Failed to parse Excel file: ${message}`,
            variant: 'destructive',
          });
        }
        setIsProcessing(false);
      } else if (ext === 'parquet') {
        // Send to backend for parsing
        const formData = new FormData();
        formData.append('file', file);

        try {
          const { data: result } = await apiClient.post('/parse-parquet', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });

          if (!result) {
            toast({
              title: 'Parquet parse error',
              description: 'Failed to parse Parquet file.',
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
              columnsOverride: Array.isArray(result.columnInfo) ? result.columnInfo as ColumnInfo[] : undefined,
            });

            if (result.previewLoaded) {
              toast({
                title: 'Parquet preview loaded',
                description: `Loaded the first ${(result.loadedRowCount ?? PARQUET_PREVIEW_ROW_LIMIT).toLocaleString()} of ${Number(result.rowCount ?? result.loadedRowCount ?? PARQUET_PREVIEW_ROW_LIMIT).toLocaleString()} rows from ${file.name}. Training can still use the full cached parquet dataset on the backend.`,
              });
            }
          } else {
            toast({
              title: 'Empty file',
              description: 'No data rows found in the Parquet file.',
              variant: 'destructive',
            });
          }
        } catch (error) {
          toast({
            title: 'Parquet error',
            description: getApiErrorMessage(error, 'Failed to send file to server for parsing.'),
            variant: 'destructive',
          });
        }
        setIsProcessing(false);
      } else {
        toast({
          title: 'Unsupported format',
          description: 'Please upload a CSV, Excel (.xlsx), or Parquet file.',
          variant: 'destructive',
        });
        setIsProcessing(false);
      }
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

  // ── Format Cards Data ────────────────────────────────────────────────────

  const formatCards = useMemo(
    () => [
      {
        title: 'CSV Files',
        icon: FileText,
        color: 'text-emerald-600 dark:text-emerald-400',
        bgColor: 'bg-emerald-500/10',
        description: 'Comma-separated values with automatic type detection. Most universal format for data exchange.',
        badge: '.csv',
      },
      {
        title: 'Excel Workbooks',
        icon: FileSpreadsheet,
        color: 'text-teal-600 dark:text-teal-400',
        bgColor: 'bg-teal-500/10',
        description: 'Multi-sheet .xlsx workbooks with cell formatting support. Reads the first sheet automatically.',
        badge: '.xlsx',
      },
      {
        title: 'Parquet Files',
        icon: Database,
        color: 'text-cyan-600 dark:text-cyan-400',
        bgColor: 'bg-cyan-500/10',
        description: 'High-performance columnar storage format used in data lakes. Parsed server-side with full type support.',
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
      className="space-y-8"
    >
      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <motion.div variants={itemVariants} className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-500/5 via-teal-500/5 to-cyan-500/5 border border-emerald-500/10 p-8 md:p-12">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-emerald-500/5 blur-3xl" />
          <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-teal-500/5 blur-3xl" />
        </div>

        <div className="relative flex flex-col md:flex-row items-center gap-6 md:gap-10">
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
              <div className="h-24 w-24 md:h-32 md:w-32 rounded-3xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-2xl shadow-emerald-500/30">
                <Bot className="h-12 w-12 md:h-16 md:w-16 text-white" />
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
          <div className="text-center md:text-left flex-1">
            <motion.h1
              className="text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 dark:from-emerald-400 dark:via-teal-400 dark:to-cyan-400 bg-clip-text text-transparent">
                Upload Your Dataset
              </span>
            </motion.h1>
            <motion.p
              className="mt-3 text-base md:text-lg text-muted-foreground max-w-xl"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.35 }}
            >
              Begin your AI-assisted exploratory data analysis by uploading a dataset.
              Drag & drop your file or browse from your computer to get started.
            </motion.p>
            <motion.div
              className="mt-4 flex flex-wrap justify-center md:justify-start gap-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20">
                CSV
              </Badge>
              <Badge variant="secondary" className="bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/20">
                Excel (.xlsx)
              </Badge>
              <Badge variant="secondary" className="bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border-cyan-500/20">
                Parquet
              </Badge>
              <Badge variant="outline" className="text-muted-foreground">
                Max 200MB
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
              className="rounded-2xl border-2 border-emerald-500/30 bg-emerald-500/5 p-8 md:p-12 flex flex-col items-center justify-center text-center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.1 }}
                className="flex h-20 w-20 items-center justify-center rounded-full bg-emerald-500/10 mb-6"
              >
                <CheckCircle2 className="h-10 w-10 text-emerald-500" />
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">Dataset Loaded</h3>
              <p className="text-sm text-muted-foreground mb-6">
                <span className="font-medium text-foreground">{uploadedFileName}</span> has been
                loaded successfully. Redirecting to Data Understanding...
              </p>
              <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 text-sm font-medium">
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
                  relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out
                  ${
                    isDragging
                      ? 'border-emerald-500 bg-emerald-500/5 shadow-lg shadow-emerald-500/10 scale-[1.01]'
                      : 'border-muted-foreground/25 hover:border-emerald-500/50 hover:bg-accent/50'
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
                      className="absolute inset-0 rounded-2xl border-2 border-emerald-500/30 pointer-events-none"
                      style={{
                        animation: 'pulse 1.5s ease-in-out infinite',
                      }}
                    />
                  )}
                </AnimatePresence>

                <div className="flex flex-col items-center justify-center py-16 px-6 md:py-20 md:px-12">
                  <AnimatePresence mode="wait">
                    {isProcessing ? (
                      <motion.div
                        key="processing"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="flex flex-col items-center gap-4"
                      >
                        <Loader2 className="h-14 w-14 text-emerald-500 animate-spin" />
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
                          className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/10"
                        >
                          <motion.div
                            animate={isDragging ? { y: -6 } : { y: 0 }}
                            transition={{ duration: 0.8, repeat: Infinity, ease: 'easeInOut' }}
                          >
                            <Upload className="h-10 w-10 text-emerald-500" />
                          </motion.div>
                        </motion.div>

                        {/* Instructions */}
                        <div className="text-center max-w-md">
                          <h3 className="text-xl font-semibold mb-1">
                            {isDragging ? 'Drop your file here' : 'Drag & drop your file here'}
                          </h3>
                          <p className="text-sm text-muted-foreground">
                            or <span className="text-emerald-600 dark:text-emerald-400 font-medium underline underline-offset-2">click to browse</span> from your computer
                          </p>
                        </div>

                        {/* Supported Formats */}
                        <div className="flex items-center gap-2 mt-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">CSV</span>
                          <span className="text-muted-foreground/30">·</span>
                          <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">Excel (.xlsx)</span>
                          <span className="text-muted-foreground/30">·</span>
                          <Database className="h-4 w-4 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">Parquet</span>
                        </div>

                        {/* Size limit */}
                        <p className="text-xs text-muted-foreground/60 mt-1">
                          Maximum file size: 200MB
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
                accept=".csv,.xlsx,.parquet"
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
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {formatCards.map((fmt, idx) => {
            const Icon = fmt.icon;
            return (
              <motion.div
                key={fmt.title}
                whileHover={cardHover}
                transition={{ duration: 0.2 }}
              >
                <Card className="h-full border-border/50 hover:border-emerald-500/20 transition-colors overflow-hidden group">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${fmt.bgColor}`}>
                        <Icon className={`h-5 w-5 ${fmt.color}`} />
                      </div>
                      <Badge variant="outline" className="text-xs font-mono">
                        {fmt.badge}
                      </Badge>
                    </div>
                    <CardTitle className="text-base mt-3 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                      {fmt.title}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-xs leading-relaxed">
                      {fmt.description}
                    </CardDescription>
                    {fmt.badge === '.parquet' && (
                      <div className="mt-3 flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
                        <Sparkles className="h-3 w-3" />
                        <span>Server-side parsing with full type detection</span>
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
        className="rounded-xl border border-border/50 bg-muted/30 p-4 flex items-start gap-3"
      >
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 mt-0.5">
          <Sparkles className="h-4 w-4 text-emerald-500" />
        </div>
        <div>
          <p className="text-sm font-medium">What happens after upload?</p>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            CSV and Excel files are parsed in the browser. Parquet files are sent to the backend parser so we can preserve their schema and types. After loading, we detect column types, count unique values and nulls, and identify potential duplicate rows before guiding you through understanding, cleaning, EDA, and ML modeling.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}



