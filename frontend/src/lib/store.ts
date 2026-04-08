import { create } from 'zustand';

export type TabId = 'upload' | 'understanding' | 'cleaning' | 'eda' | 'ml' | 'sales_forecast' | 'prediction' | 'report';

export interface ColumnInfo {
  name: string;
  dtype: string;
  nonNull: number;
  nullCount: number;
  uniqueCount: number;
  role: 'identifier' | 'numeric' | 'categorical' | 'boolean' | 'datetime' | 'unknown';
  sample: string[];
}

export interface DataRow {
  [key: string]: string | number | boolean | null;
}

export interface CleaningLog {
  action: string;
  detail: string;
  timestamp: string;
}


export interface SalesForecastPoint {
  period: string;
  actual?: number;
  predicted: number;
}

export interface SalesForecastResult {
  date_column: string;
  target_column: string;
  frequency?: string;
  period_label?: string;
  history: { period: string; actual: number }[];
  test_forecast: { period: string; actual: number; predicted: number }[];
  future_forecast: SalesForecastPoint[];
  metrics: { mae: number; rmse: number; mape: number };
  training_summary: {
    model_name: string;
    total_periods: number;
    train_periods: number;
    test_periods: number;
    train_percentage: number;
    test_percentage: number;
    forecast_periods: number;
    lag_periods: number;
    train_start: string;
    train_end: string;
    test_start: string;
    test_end: string;
    last_observed_period: string;
  };
  analysis: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  target: string;
  problem: string;
  trainedAt: string;
  metrics: Record<string, number>;
  features: string[];
}

export interface AppState {
  // Navigation
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
  mlWorkflowStep: number;
  setMlWorkflowStep: (step: number) => void;

  // Data
  fileName: string | null;
  datasetId: string | null;
  previewLoaded: boolean;
  loadedRowCount: number;
  cleanedRowCount: number | null;
  rawData: DataRow[] | null;
  cleanedData: DataRow[] | null;
  columns: ColumnInfo[];
  totalRows: number;
  duplicates: number;
  memoryUsage: string;

  // Cleaning
  cleaningLogs: CleaningLog[];
  cleaningDone: boolean;

  // ML
  targetColumn: string | null;
  problemType: 'regression' | 'classification';
  selectedFeatures: string[];
  selectedModel: string | null;
  modelId: string | null;
  modelMetrics: Record<string, number> | null;
  modelTrained: boolean;
  featureImportance: { name: string; importance: number }[] | null;

  // Prediction
  uploadedModel: ModelInfo | null;
  predictionResult: number | string | null;
  predictionAnalysis: string | null;
  predictionProbabilities: Record<string, number> | null;
  predictionHistory: { id: string; prediction: number | string; confidence?: number; probabilities?: Record<string, number>; features: Record<string, string | number>; timestamp: string }[];

  // Sales Forecast
  salesForecastResult: SalesForecastResult | null;

  // Report
  reportGenerated: boolean;
  reportUrl: string | null;
  setReportGenerated: (v: boolean) => void;
  setReportUrl: (v: string | null) => void;

  // AI
  aiInsights: string | null;
  aiChatHistory: { role: 'user' | 'assistant'; content: string }[];
}

export const useAppStore = create<AppState>((set) => ({
  // Navigation
  activeTab: 'upload',
  setActiveTab: (tab) => set({ activeTab: tab }),
  mlWorkflowStep: 1,
  setMlWorkflowStep: (step) => set({ mlWorkflowStep: Math.max(1, Math.min(6, step)) }),

  // Data
  fileName: null,
  datasetId: null,
  previewLoaded: false,
  loadedRowCount: 0,
  cleanedRowCount: null,
  rawData: null,
  cleanedData: null,
  columns: [],
  totalRows: 0,
  duplicates: 0,
  memoryUsage: '',

  // Cleaning
  cleaningLogs: [],
  cleaningDone: false,

  // ML
  targetColumn: null,
  problemType: 'regression',
  selectedFeatures: [],
  selectedModel: null,
  modelId: null,
  modelMetrics: null,
  modelTrained: false,
  featureImportance: null,

  // Prediction
  uploadedModel: null,
  predictionResult: null,
  predictionAnalysis: null,
  predictionProbabilities: null,
  predictionHistory: [],

  // Sales Forecast
  salesForecastResult: null,

  // Report
  reportGenerated: false,
  reportUrl: null,
  setReportGenerated: (v) => set({ reportGenerated: v }),
  setReportUrl: (v) => set({ reportUrl: v }),

  // AI
  aiInsights: null,
  aiChatHistory: [],
}));

