import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import AgenticWorkspace from './agentic-workspace';

type MockState = {
  agenticSessionId: string | null;
  agenticStepStatuses: Record<string, string>;
  agenticRecommendations: Array<{ step: string; reason: string; findings?: string[] }>;
  agenticLastSyncedAt: number | null;
  datasetId: string | null;
  fileName: string | null;
  rawData: Array<Record<string, unknown>> | null;
  cleanedData: Array<Record<string, unknown>> | null;
  columns: Array<{ name: string; role: string; uniqueCount?: number }>;
  totalRows: number;
  duplicates: number;
  cleanedRowCount: number | null;
  cleaningDone: boolean;
  cleaningLogs: string[];
  timeSeriesForecastResult: unknown;
  mlForecastResult: unknown;
  selectedModel: string | null;
  lossSummary: Record<string, unknown> | null;
  lossForecast: Array<Record<string, unknown>> | null;
  predictionResult: unknown;
  setAgenticSessionId: (sessionId: string | null) => void;
  setAgenticStepStatus: (step: string, status: string) => void;
  setAgenticRecommendations: (recommendations: MockState['agenticRecommendations']) => void;
  setAgenticLastSyncedAt: (value: number | null) => void;
};

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}));

let state: MockState;

vi.mock('axios', () => ({
  default: {
    create: () => ({
      get: getMock,
      post: postMock,
    }),
  },
}));

vi.mock('@/lib/api', () => ({
  getApiErrorMessage: (error: unknown, fallback: string) => error instanceof Error ? error.message : fallback,
}));

vi.mock('@/lib/store', () => ({
  useAppStore: Object.assign(
    () => state,
    {
      getState: () => state,
      setState: (patch: Partial<MockState>) => {
        state = { ...state, ...patch };
      },
    },
  ),
}));

function resetState(patch: Partial<MockState> = {}) {
  state = {
    agenticSessionId: null,
    agenticStepStatuses: {},
    agenticRecommendations: [],
    agenticLastSyncedAt: null,
    datasetId: null,
    fileName: null,
    rawData: [{ amount: 10, date: '2026-01-01' }],
    cleanedData: null,
    columns: [
      { name: 'date', role: 'datetime', uniqueCount: 1 },
      { name: 'amount', role: 'numeric', uniqueCount: 1 },
    ],
    totalRows: 1,
    duplicates: 0,
    cleanedRowCount: null,
    cleaningDone: false,
    cleaningLogs: [],
    timeSeriesForecastResult: null,
    mlForecastResult: null,
    selectedModel: null,
    lossSummary: null,
    lossForecast: null,
    predictionResult: null,
    setAgenticSessionId: (sessionId) => {
      state = { ...state, agenticSessionId: sessionId };
    },
    setAgenticStepStatus: (step, status) => {
      state = {
        ...state,
        agenticStepStatuses: { ...state.agenticStepStatuses, [step]: status },
      };
    },
    setAgenticRecommendations: (recommendations) => {
      state = { ...state, agenticRecommendations: recommendations };
    },
    setAgenticLastSyncedAt: (value) => {
      state = { ...state, agenticLastSyncedAt: value };
    },
    ...patch,
  };
}

function mockHealth() {
  getMock.mockImplementation((url: string) => {
    if (url === '/health') {
      return Promise.resolve({ data: { status: 'ok', agentic_enabled: true, db_connected: true, db_fallback_active: false } });
    }
    if (url.startsWith('/session/')) {
      return Promise.resolve({
        data: {
          steps: {},
          recommendations: [],
          results: {},
          last_result: null,
        },
      });
    }
    return Promise.reject(new Error(`unexpected GET ${url}`));
  });
}

beforeEach(() => {
  vi.useRealTimers();
  getMock.mockReset();
  postMock.mockReset();
  mockHealth();
  resetState();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('AgenticWorkspace', () => {
  test('step panel renders with status from backend response', async () => {
    resetState({ agenticSessionId: 'session-1' });
    getMock.mockImplementation((url: string) => {
      if (url === '/health') {
        return Promise.resolve({ data: { status: 'ok', agentic_enabled: true, db_connected: true, db_fallback_active: false } });
      }
      if (url === '/session/session-1/status') {
        return Promise.resolve({
          data: {
            steps: { EDA: 'completed' },
            recommendations: [],
            results: {
              EDA: {
                step_id: 'step-1',
                status: 'completed',
                executed_at: new Date().toISOString(),
                result: { output_summary: 'EDA completed by backend' },
              },
            },
          },
        });
      }
      return Promise.reject(new Error(`unexpected GET ${url}`));
    });

    render(<AgenticWorkspace datasetId={null} fileName="sales.csv" />);

    expect(await screen.findByText('Backend Step Result')).toBeInTheDocument();
    expect(screen.getByText('EDA completed by backend')).toBeInTheDocument();
  });

  test('clicking execute calls execute-step only', async () => {
    resetState({
      agenticSessionId: 'session-2',
      agenticRecommendations: [{ step: 'EDA', reason: 'Run EDA next.' }],
    });
    postMock.mockResolvedValue({
      data: {
        status: 'completed',
        step_name: 'EDA',
        output_summary: 'EDA completed on backend',
        next_recommendations: [],
      },
    });

    render(<AgenticWorkspace datasetId={null} fileName="sales.csv" />);

    fireEvent.click(screen.getByRole('button', { name: /accept & continue/i }));

    await waitFor(() => expect(postMock).toHaveBeenCalled());
    expect(postMock).toHaveBeenCalledTimes(1);
    expect(postMock.mock.calls[0][0]).toBe('/execute-step');
    expect(postMock.mock.calls.some(([url]) => url === '/decision')).toBe(false);
  });

  test('polling stops when backend returns completed status', async () => {
    vi.useFakeTimers();
    const setIntervalSpy = vi.spyOn(window, 'setInterval');
    const clearIntervalSpy = vi.spyOn(window, 'clearInterval');
    resetState({
      agenticSessionId: 'session-3',
      agenticRecommendations: [{ step: 'EDA', reason: 'Run EDA next.' }],
    });
    let resolvePost: (value: unknown) => void = () => {};
    postMock.mockReturnValue(new Promise((resolve) => {
      resolvePost = resolve;
    }));
    getMock.mockImplementation((url: string) => {
      if (url === '/health') {
        return Promise.resolve({ data: { status: 'ok', agentic_enabled: true, db_connected: true, db_fallback_active: false } });
      }
      if (url === '/session/session-3/status') {
        return Promise.resolve({
          data: {
            steps: { EDA: 'completed' },
            recommendations: [],
            results: {
              EDA: {
                step_id: 'step-3',
                status: 'completed',
                executed_at: new Date().toISOString(),
                result: { output_summary: 'done' },
              },
            },
          },
        });
      }
      return Promise.reject(new Error(`unexpected GET ${url}`));
    });

    render(<AgenticWorkspace datasetId={null} fileName="sales.csv" />);
    fireEvent.click(screen.getByRole('button', { name: /accept & continue/i }));

    await act(async () => {
      await Promise.resolve();
    });
    expect(setIntervalSpy).toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(clearIntervalSpy).toHaveBeenCalled();
    resolvePost({
      data: {
        status: 'completed',
        step_name: 'EDA',
        output_summary: 'done',
        next_recommendations: [],
      },
    });
  });
});
