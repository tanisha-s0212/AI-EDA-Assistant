import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { apiClient, getApiErrorMessage } from '@/lib/api';
import type { LossForecastResult, LossForecastSummary, ProfitForecastResult, ScenarioComparison, SegmentBreakdown } from '@/types/forecast';

export type TabId = 'upload' | 'understanding' | 'cleaning' | 'eda' | 'forecast_ts' | 'forecast_ml' | 'loss_forecast' | 'profit_forecast' | 'ml' | 'prediction' | 'report';

// # AGENTIC LAYER START
export type AgenticStepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface Recommendation {
  step: string;
  reason: string;
  findings: string[];
}
// # AGENTIC LAYER END

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

export interface DatasetSheetSummary {
  name: string;
  rowCount: number;
  columnCount: number;
  columns?: string[];
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
  lossForecast: LossForecastResult[] | null;
  profitForecast: ProfitForecastResult[] | null;
  lossSegments: SegmentBreakdown[] | null;
  lossSummary: LossForecastSummary | null;
  scenarios: ScenarioComparison | null;
  breakevenPeriod: string | null;
  periodsToBreakeven: number | null;
  lossLoading: boolean;
  profitLoading: boolean;
  lossError: string | null;
  profitError: string | null;
  reportGenerated: boolean;
  reportUrl: string | null;
  aiInsights: string | null;
  aiChatHistory: { role: 'user' | 'assistant'; content: string }[];
  availableSheets: DatasetSheetSummary[];
  selectedSheets: string[];
  sheetMergeMode: 'single' | 'stack';
}

export interface DatasetWorkspace extends DatasetWorkspaceState {
  key: string;
  createdAt: string;
}

export type DatasetWorkspaceDraft = DatasetWorkspaceState;

export interface AuthenticatedUser {
  userId: string;
  username: string;
  email: string;
  profileImageDataUrl?: string | null;
  createdAt: string;
  updatedAt: string;
  lastLoginAt: string;
}

export interface AppState extends DatasetWorkspaceState {
  activeTab: TabId;
  uploadPickerRequestId: number;
  uploadPickerSourceTab: TabId | null;
  currentUser: AuthenticatedUser | null;
  isAuthenticated: boolean;
  // # AGENTIC LAYER START
  agenticSessionId: string | null;
  agenticStepStatuses: Record<string, AgenticStepStatus>;
  agenticRecommendations: Recommendation[];
  agenticLastSyncedAt: number | null;
  agenticEnabled: boolean;
  setAgenticSessionId: (sessionId: string | null) => void;
  setAgenticStepStatus: (step: string, status: AgenticStepStatus) => void;
  setAgenticRecommendations: (recommendations: Recommendation[]) => void;
  setAgenticLastSyncedAt: (value: number | null) => void;
  // # AGENTIC LAYER END
  setActiveTab: (tab: TabId) => void;
  requestUploadPicker: (sourceTab?: TabId) => void;
  resetWorkspace: () => void;
  setCurrentUser: (user: AuthenticatedUser) => void;
  logoutUser: () => void;
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
  runLossForecast: (sessionId: string, periods: number) => Promise<void>;
  runProfitForecast: (sessionId: string, periods: number) => Promise<void>;
  fetchLossSegments: (sessionId: string) => Promise<void>;
  fetchBreakeven: (sessionId: string) => Promise<void>;
  resetForecasts: () => void;
}

type PersistedAppSlice = Pick<
  AppState,
  | 'activeTab'
  | 'mlWorkflowStep'
  | 'hasHydrated'
  | 'currentUser'
  | 'isAuthenticated'
  | 'datasets'
  | 'datasetOrder'
  | 'activeDatasetKey'
  // # AGENTIC LAYER START
  | 'agenticSessionId'
  | 'agenticStepStatuses'
  | 'agenticRecommendations'
  | 'agenticLastSyncedAt'
  | 'agenticEnabled'
  // # AGENTIC LAYER END
  | keyof DatasetWorkspaceState
>;

const STORE_PERSIST_KEY = 'ai-eda-workspace-v2';
const STORE_PERSIST_VERSION = 4;

// # AGENTIC LAYER START
const AGENTIC_ENABLED = process.env.NEXT_PUBLIC_AGENTIC_ENABLED === 'true';
// # AGENTIC LAYER END

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
    lossForecast: null,
    profitForecast: null,
    lossSegments: null,
    lossSummary: null,
    scenarios: null,
    breakevenPeriod: null,
    periodsToBreakeven: null,
    lossLoading: false,
    profitLoading: false,
    lossError: null,
    profitError: null,
    reportGenerated: false,
    reportUrl: null,
    aiInsights: null,
    aiChatHistory: [],
    availableSheets: [],
    selectedSheets: [],
    sheetMergeMode: 'single',
  };
}

function stripTransientDatasetState<T extends DatasetWorkspaceState>(dataset: T): T {
  return {
    ...dataset,
    rawData: null,
    cleanedData: null,
    reportUrl: null,
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
  'lossForecast',
  'profitForecast',
  'lossSegments',
  'lossSummary',
  'scenarios',
  'breakevenPeriod',
  'periodsToBreakeven',
  'lossLoading',
  'profitLoading',
  'lossError',
  'profitError',
  'reportGenerated',
  'reportUrl',
  'aiInsights',
  'aiChatHistory',
  'availableSheets',
  'selectedSheets',
  'sheetMergeMode',
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
  currentUser: null,
  isAuthenticated: false,
  datasets: {},
  datasetOrder: [],
  activeDatasetKey: null,
  // # AGENTIC LAYER START
  agenticSessionId: null,
  agenticStepStatuses: {},
  agenticRecommendations: [],
  agenticLastSyncedAt: null,
  agenticEnabled: AGENTIC_ENABLED,
  // # AGENTIC LAYER END
  ...createEmptyDatasetState(),
};

const store = create<AppState>()(
  persist(
    (set, get) => ({
      ...initialPersistedState,
      uploadPickerRequestId: 0,
      uploadPickerSourceTab: null,
      // # AGENTIC LAYER START
      setAgenticSessionId: (sessionId) => set({ agenticSessionId: sessionId }),
      setAgenticStepStatus: (step, status) =>
        set((state) => ({
          agenticStepStatuses: {
            ...state.agenticStepStatuses,
            [step]: status,
          },
          agenticLastSyncedAt: Date.now(),
        })),
      setAgenticRecommendations: (recommendations) =>
        set({
          agenticRecommendations: recommendations,
          agenticLastSyncedAt: Date.now(),
        }),
      setAgenticLastSyncedAt: (value) => set({ agenticLastSyncedAt: value }),
      // # AGENTIC LAYER END
      setActiveTab: (tab) => set({ activeTab: tab }),
      requestUploadPicker: (sourceTab) =>
        set((state) => ({
          uploadPickerRequestId: state.uploadPickerRequestId + 1,
          uploadPickerSourceTab: sourceTab ?? state.activeTab,
        })),
      setCurrentUser: (user) => set({ currentUser: user, isAuthenticated: true }),
      logoutUser: () => set({ currentUser: null, isAuthenticated: false }),
      resetWorkspace: () => {
        const previousReportUrl = get().reportUrl;
        const { currentUser, isAuthenticated } = get();
        if (previousReportUrl && typeof URL !== 'undefined') {
          URL.revokeObjectURL(previousReportUrl);
        }
        set({
          ...initialPersistedState,
          uploadPickerRequestId: 0,
          uploadPickerSourceTab: null,
          currentUser,
          isAuthenticated,
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
      runLossForecast: async (sessionId, periods) => {
        set({ lossLoading: true, lossError: null });
        try {
          const response = await apiClient.post('/loss-forecast/run', { session_id: sessionId, forecast_periods: periods });
          set({
            lossForecast: response.data.loss_forecast ?? [],
            lossSegments: response.data.segments ?? [],
            lossSummary: response.data.summary ?? null,
            lossLoading: false,
            lossError: null,
          });
          if (!response.data.segments?.length) {
            await get().fetchLossSegments(sessionId);
          }
        } catch (error) {
          set({ lossLoading: false, lossError: getApiErrorMessage(error, 'Loss forecast failed.') });
          throw error;
        }
      },
      runProfitForecast: async (sessionId, periods) => {
        set({ profitLoading: true, profitError: null });
        try {
          const response = await apiClient.post('/profit-forecast/run', { session_id: sessionId, forecast_periods: periods });
          const scenarios = response.data.scenarios as ScenarioComparison;
          set({
            scenarios,
            profitForecast: scenarios?.baseline ?? [],
            breakevenPeriod: response.data.breakeven?.breakeven_period ?? null,
            periodsToBreakeven: response.data.breakeven?.periods_to_breakeven ?? null,
            profitLoading: false,
            profitError: null,
          });
        } catch (error) {
          set({ profitLoading: false, profitError: getApiErrorMessage(error, 'Profit forecast failed.') });
          throw error;
        }
      },
      fetchLossSegments: async (sessionId) => {
        const response = await apiClient.get(`/loss-forecast/segments/${sessionId}`);
        set({ lossSegments: response.data.segments ?? [] });
      },
      fetchBreakeven: async (sessionId) => {
        const response = await apiClient.get(`/profit-forecast/breakeven/${sessionId}`);
        set({
          breakevenPeriod: response.data.breakeven_period ?? null,
          periodsToBreakeven: response.data.periods_to_breakeven ?? null,
        });
      },
      resetForecasts: () => set({
        lossForecast: null,
        profitForecast: null,
        lossSegments: null,
        lossSummary: null,
        scenarios: null,
        breakevenPeriod: null,
        periodsToBreakeven: null,
        lossLoading: false,
        profitLoading: false,
        lossError: null,
        profitError: null,
      }),
    }),
    {
      name: STORE_PERSIST_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        activeTab: state.activeTab,
        mlWorkflowStep: state.mlWorkflowStep,
        hasHydrated: state.hasHydrated,
        currentUser: state.currentUser,
        isAuthenticated: state.isAuthenticated,
        datasets: Object.fromEntries(
          Object.entries(state.datasets).map(([key, dataset]) => [
            key,
            stripTransientDatasetState(dataset),
          ])
        ),
        datasetOrder: state.datasetOrder,
        activeDatasetKey: state.activeDatasetKey,
        // # AGENTIC LAYER START
        agenticSessionId: state.agenticSessionId,
        agenticStepStatuses: state.agenticStepStatuses,
        agenticRecommendations: state.agenticRecommendations,
        agenticLastSyncedAt: state.agenticLastSyncedAt,
        agenticEnabled: AGENTIC_ENABLED,
        // # AGENTIC LAYER END
        ...buildDatasetStatePatch(stripTransientDatasetState(getDatasetSnapshot(state))),
      }),
      version: STORE_PERSIST_VERSION,
      migrate: (persistedState) => {
        const state = persistedState as PersistedAppSlice | undefined;
        if (!state) return initialPersistedState;

        return {
          ...state,
          currentUser: state.currentUser ?? null,
          isAuthenticated: Boolean(state.currentUser ?? state.isAuthenticated),
          datasets: Object.fromEntries(
            Object.entries(state.datasets ?? {}).map(([key, dataset]) => [
              key,
              stripTransientDatasetState({
                ...createEmptyDatasetState(),
                ...(dataset as DatasetWorkspace),
              }),
            ])
          ),
          // # AGENTIC LAYER START
          agenticSessionId: state.agenticSessionId ?? null,
          agenticStepStatuses: state.agenticStepStatuses ?? {},
          agenticRecommendations: state.agenticRecommendations ?? [],
          agenticLastSyncedAt: state.agenticLastSyncedAt ?? null,
          agenticEnabled: AGENTIC_ENABLED,
          // # AGENTIC LAYER END
          ...buildDatasetStatePatch(
            stripTransientDatasetState({
              ...createEmptyDatasetState(),
              ...state,
            })
          ),
        } satisfies PersistedAppSlice;
      },
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
