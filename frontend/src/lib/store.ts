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

export interface DatasetWorkspaceState {
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
  cleaningLogs: CleaningLog[];
  cleaningDone: boolean;
  targetColumn: string | null;
  problemType: 'regression' | 'classification';
  selectedFeatures: string[];
  selectedModel: string | null;
  modelId: string | null;
  modelMetrics: Record<string, number> | null;
  modelTrained: boolean;
  featureImportance: { name: string; importance: number }[] | null;
  uploadedModel: ModelInfo | null;
  predictionResult: number | string | null;
  predictionAnalysis: string | null;
  predictionProbabilities: Record<string, number> | null;
  predictionHistory: { id: string; prediction: number | string; confidence?: number; probabilities?: Record<string, number>; features: Record<string, string | number>; timestamp: string }[];
  timeSeriesForecastResult: TimeSeriesForecastResult | null;
  mlForecastResult: MlForecastResult | null;
  reportGenerated: boolean;
  reportUrl: string | null;
  aiInsights: string | null;
  aiChatHistory: { role: 'user' | 'assistant'; content: string }[];
}

export interface DatasetWorkspace extends DatasetWorkspaceState {
  key: string;
  createdAt: string;
}

export type DatasetWorkspaceDraft = DatasetWorkspaceState;

export interface AppState extends DatasetWorkspaceState {
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
  resetWorkspace: () => void;
  hasHydrated: boolean;
  setHasHydrated: (value: boolean) => void;
  mlWorkflowStep: number;
  setMlWorkflowStep: (step: number) => void;
  datasets: Record<string, DatasetWorkspace>;
  datasetOrder: string[];
  activeDatasetKey: string | null;
  addDataset: (dataset: DatasetWorkspaceDraft, options?: { key?: string; activate?: boolean }) => string;
  selectDataset: (key: string) => void;
  setReportGenerated: (v: boolean) => void;
  setReportUrl: (v: string | null) => void;
}

type PersistedAppSlice = Pick<
  AppState,
  | 'activeTab'
  | 'mlWorkflowStep'
  | 'hasHydrated'
  | 'datasets'
  | 'datasetOrder'
  | 'activeDatasetKey'
  | keyof DatasetWorkspaceState
>;

const STORE_PERSIST_KEY = 'ai-eda-workspace-v2';

function createEmptyDatasetState(): DatasetWorkspaceState {
  return {
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
    reportUrl: null,
    aiInsights: null,
    aiChatHistory: [],
  };
}

const datasetStateKeys: Array<keyof DatasetWorkspaceState> = [
  'fileName',
  'datasetId',
  'previewLoaded',
  'loadedRowCount',
  'cleanedRowCount',
  'rawData',
  'cleanedData',
  'columns',
  'totalRows',
  'duplicates',
  'memoryUsage',
  'cleaningLogs',
  'cleaningDone',
  'targetColumn',
  'problemType',
  'selectedFeatures',
  'selectedModel',
  'modelId',
  'modelMetrics',
  'modelTrained',
  'featureImportance',
  'uploadedModel',
  'predictionResult',
  'predictionAnalysis',
  'predictionProbabilities',
  'predictionHistory',
  'timeSeriesForecastResult',
  'mlForecastResult',
  'reportGenerated',
  'reportUrl',
  'aiInsights',
  'aiChatHistory',
];

function getDatasetSnapshot(state: DatasetWorkspaceState): DatasetWorkspaceState {
  return Object.fromEntries(datasetStateKeys.map((key) => [key, state[key]])) as unknown as DatasetWorkspaceState;
}

function buildDatasetStatePatch(dataset: DatasetWorkspaceState) {
  return Object.fromEntries(datasetStateKeys.map((key) => [key, dataset[key]])) as unknown as DatasetWorkspaceState;
}

function buildDatasetKey(dataset: DatasetWorkspaceDraft, preferredKey?: string) {
  if (preferredKey) return preferredKey;
  const base = (dataset.datasetId ?? dataset.fileName ?? 'dataset')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
  return `${base || 'dataset'}-${Date.now().toString(36)}`;
}

const initialPersistedState: PersistedAppSlice = {
  activeTab: 'upload',
  mlWorkflowStep: 1,
  hasHydrated: false,
  datasets: {},
  datasetOrder: [],
  activeDatasetKey: null,
  ...createEmptyDatasetState(),
};

const store = create<AppState>()(
  persist(
    (set, get) => ({
      ...initialPersistedState,
      setActiveTab: (tab) => set({ activeTab: tab }),
      resetWorkspace: () => {
        const previousReportUrl = get().reportUrl;
        if (previousReportUrl && typeof URL !== 'undefined') {
          URL.revokeObjectURL(previousReportUrl);
        }
        set({
          ...initialPersistedState,
          hasHydrated: true,
        });
      },
      setHasHydrated: (value) => set({ hasHydrated: value }),
      setMlWorkflowStep: (step) => set({ mlWorkflowStep: Math.max(1, Math.min(6, step)) }),
      addDataset: (dataset, options) => {
        const nextKey = buildDatasetKey(dataset, options?.key);
        const nextDataset: DatasetWorkspace = {
          key: nextKey,
          createdAt: new Date().toISOString(),
          ...createEmptyDatasetState(),
          ...dataset,
        };

        set((state) => {
          const previousReportUrl = state.reportUrl;
          if (previousReportUrl && previousReportUrl !== nextDataset.reportUrl && typeof URL !== 'undefined') {
            URL.revokeObjectURL(previousReportUrl);
          }

          return {
            datasets: {
              ...state.datasets,
              [nextKey]: nextDataset,
            },
            datasetOrder: state.datasetOrder.includes(nextKey) ? state.datasetOrder : [nextKey, ...state.datasetOrder],
            activeDatasetKey: options?.activate === false ? state.activeDatasetKey : nextKey,
            ...(options?.activate === false ? {} : buildDatasetStatePatch(nextDataset)),
          };
        });

        return nextKey;
      },
      selectDataset: (key) => {
        const state = get();
        const nextDataset = state.datasets[key];
        if (!nextDataset) return;

        const previousReportUrl = state.reportUrl;
        if (previousReportUrl && previousReportUrl !== nextDataset.reportUrl && typeof URL !== 'undefined') {
          URL.revokeObjectURL(previousReportUrl);
        }

        set({
          activeDatasetKey: key,
          ...buildDatasetStatePatch(nextDataset),
        });
      },
      setReportGenerated: (v) => set({ reportGenerated: v }),
      setReportUrl: (v) => set({ reportUrl: v }),
    }),
    {
      name: STORE_PERSIST_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        activeTab: state.activeTab,
        mlWorkflowStep: state.mlWorkflowStep,
        hasHydrated: state.hasHydrated,
        datasets: Object.fromEntries(
          Object.entries(state.datasets).map(([key, dataset]) => [
            key,
            {
              ...dataset,
              reportUrl: null,
            },
          ])
        ),
        datasetOrder: state.datasetOrder,
        activeDatasetKey: state.activeDatasetKey,
        ...buildDatasetStatePatch({
          ...getDatasetSnapshot(state),
          reportUrl: null,
        }),
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);

let isSyncingDatasetRegistry = false;

store.subscribe((state) => {
  if (isSyncingDatasetRegistry || !state.activeDatasetKey) return;
  const activeDataset = state.datasets[state.activeDatasetKey];
  if (!activeDataset) return;

  const nextSnapshot = getDatasetSnapshot(state);
  const changed = datasetStateKeys.some((key) => activeDataset[key] !== nextSnapshot[key]);
  if (!changed) return;

  isSyncingDatasetRegistry = true;
  store.setState((currentState) => ({
    datasets: {
      ...currentState.datasets,
      [state.activeDatasetKey as string]: {
        ...currentState.datasets[state.activeDatasetKey as string],
        ...nextSnapshot,
      },
    },
  }));
  isSyncingDatasetRegistry = false;
});

export const useAppStore = store;
