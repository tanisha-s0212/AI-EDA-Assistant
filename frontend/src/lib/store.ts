import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type TabId = 'upload' | 'understanding' | 'cleaning' | 'eda' | 'forecast_ts' | 'forecast_ml' | 'ml' | 'prediction' | 'report';

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

export interface ForecastPoint {
  period: string;
  actual?: number;
  predicted: number;
  lower?: number | null;
  upper?: number | null;
}

export interface ForecastFeatureImportance {
  name: string;
  importance: number;
}

export interface DatasetProfile {
  detected_frequency: string;
  usable_periods: number;
  volatility: number;
  zero_value_share: number;
}

export interface StationarityCheck {
  test_name: string;
  p_value: number;
  verdict: string;
  note: string;
}

export interface ForecastTrainingSummary {
  model_name: string;
  total_periods: number;
  train_periods: number;
  test_periods: number;
  train_percentage: number;
  test_percentage: number;
  forecast_periods: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  last_observed_period: string;
}

export interface TimeSeriesForecastResult {
  date_column: string;
  target_column: string;
  frequency?: string;
  period_label?: string;
  dataset_profile: DatasetProfile;
  stationarity_check: StationarityCheck;
  history: { period: string; actual: number }[];
  test_forecast: ForecastPoint[];
  future_forecast: ForecastPoint[];
  metrics: { mae: number; rmse: number; mape: number };
  training_summary: ForecastTrainingSummary;
  recommended_models?: { model_type: string; model_name: string; recommendation_reason: string; recommended?: boolean }[];
  model_details?: { model_type: string; model_name: string; rationale?: string };
  analysis: string;
}

export interface MlForecastResult {
  date_column: string;
  target_column: string;
  frequency?: string;
  period_label?: string;
  dataset_profile: DatasetProfile;
  generated_features: string[];
  feature_preview_rows: Record<string, string | number | null>[];
  history: { period: string; actual: number }[];
  test_forecast: ForecastPoint[];
  future_forecast: ForecastPoint[];
  metrics: { mae: number; rmse: number; mape: number };
  training_summary: ForecastTrainingSummary & { lag_periods: number };
  shap_feature_importance: ForecastFeatureImportance[];
  recommended_models?: { model_type: string; model_name: string; recommendation_reason: string; recommended?: boolean }[];
  model_details?: { model_type: string; model_name: string; rationale?: string };
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
  resetWorkspace: () => void;
  hasHydrated: boolean;
  setHasHydrated: (value: boolean) => void;
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

  // Forecasting
  timeSeriesForecastResult: TimeSeriesForecastResult | null;
  mlForecastResult: MlForecastResult | null;

  // Report
  reportGenerated: boolean;
  reportUrl: string | null;
  setReportGenerated: (v: boolean) => void;
  setReportUrl: (v: string | null) => void;

  // AI
  aiInsights: string | null;
  aiChatHistory: { role: 'user' | 'assistant'; content: string }[];
}

type PersistedAppSlice = Pick<
  AppState,
  | 'activeTab'
  | 'mlWorkflowStep'
  | 'fileName'
  | 'datasetId'
  | 'previewLoaded'
  | 'loadedRowCount'
  | 'cleanedRowCount'
  | 'rawData'
  | 'cleanedData'
  | 'columns'
  | 'totalRows'
  | 'duplicates'
  | 'memoryUsage'
  | 'cleaningLogs'
  | 'cleaningDone'
  | 'targetColumn'
  | 'problemType'
  | 'selectedFeatures'
  | 'selectedModel'
  | 'modelId'
  | 'modelMetrics'
  | 'modelTrained'
  | 'featureImportance'
  | 'uploadedModel'
  | 'predictionResult'
  | 'predictionAnalysis'
  | 'predictionProbabilities'
  | 'predictionHistory'
  | 'timeSeriesForecastResult'
  | 'mlForecastResult'
  | 'reportGenerated'
  | 'aiInsights'
  | 'aiChatHistory'
>;

const STORE_PERSIST_KEY = 'ai-eda-workspace-v1';

const initialPersistedState: PersistedAppSlice = {
  activeTab: 'upload',
  mlWorkflowStep: 1,
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
  cleaningLogs: [],
  cleaningDone: false,
  targetColumn: null,
  problemType: 'regression',
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
  aiInsights: null,
  aiChatHistory: [],
};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      ...initialPersistedState,
      reportUrl: null,
      hasHydrated: false,
      setActiveTab: (tab) => set({ activeTab: tab }),
      resetWorkspace: () => {
        const previousReportUrl = get().reportUrl;
        if (previousReportUrl && typeof URL !== 'undefined') {
          URL.revokeObjectURL(previousReportUrl);
        }
        set({
          ...initialPersistedState,
          reportUrl: null,
          hasHydrated: true,
        });
      },
      setHasHydrated: (value) => set({ hasHydrated: value }),
      setMlWorkflowStep: (step) => set({ mlWorkflowStep: Math.max(1, Math.min(6, step)) }),
      setReportGenerated: (v) => set({ reportGenerated: v }),
      setReportUrl: (v) => set({ reportUrl: v }),
    }),
    {
      name: STORE_PERSIST_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        activeTab: state.activeTab,
        mlWorkflowStep: state.mlWorkflowStep,
        fileName: state.fileName,
        datasetId: state.datasetId,
        previewLoaded: state.previewLoaded,
        loadedRowCount: state.loadedRowCount,
        cleanedRowCount: state.cleanedRowCount,
        rawData: state.rawData,
        cleanedData: state.cleanedData,
        columns: state.columns,
        totalRows: state.totalRows,
        duplicates: state.duplicates,
        memoryUsage: state.memoryUsage,
        cleaningLogs: state.cleaningLogs,
        cleaningDone: state.cleaningDone,
        targetColumn: state.targetColumn,
        problemType: state.problemType,
        selectedFeatures: state.selectedFeatures,
        selectedModel: state.selectedModel,
        modelId: state.modelId,
        modelMetrics: state.modelMetrics,
        modelTrained: state.modelTrained,
        featureImportance: state.featureImportance,
        uploadedModel: state.uploadedModel,
        predictionResult: state.predictionResult,
        predictionAnalysis: state.predictionAnalysis,
        predictionProbabilities: state.predictionProbabilities,
        predictionHistory: state.predictionHistory,
        timeSeriesForecastResult: state.timeSeriesForecastResult,
        mlForecastResult: state.mlForecastResult,
        reportGenerated: state.reportGenerated,
        aiInsights: state.aiInsights,
        aiChatHistory: state.aiChatHistory,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);

