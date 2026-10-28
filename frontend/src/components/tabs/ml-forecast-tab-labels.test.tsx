import React from 'react';
import { render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import MlForecastTab from './ml-forecast-tab';
import { useAppStore } from '@/lib/store';

const { postMock } = vi.hoisted(() => ({ postMock: vi.fn() }));
vi.mock('@/lib/api', () => ({
  apiClient: { post: postMock, get: vi.fn() },
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
}));

function makeRows(dates: string[]) {
  return dates.flatMap((doc_date, index) => [
    { doc_date, item: 'A', sales: index * 10 + 10 },
    { doc_date, item: 'B', sales: index * 10 + 5 },
  ]);
}

function configureDataset(dates: string[]) {
  useAppStore.setState({
    datasetId: 'forecast-label-test',
    rawData: makeRows(dates),
    cleanedData: null,
    columns: [
      { name: 'doc_date', dtype: 'datetime64', nonNull: dates.length * 2, nullCount: 0, uniqueCount: dates.length, role: 'datetime', sample: [] },
      { name: 'sales', dtype: 'float64', nonNull: dates.length * 2, nullCount: 0, uniqueCount: dates.length * 2, role: 'numeric', sample: [] },
    ],
    confirmedSalesMapping: null,
    mlForecastResult: null,
  });
}

beforeEach(() => {
  postMock.mockReset();
  postMock.mockResolvedValue({ data: { period_count: 24 } });
});

it('uses month wording for a monthly series without changing the selected horizon', async () => {
  configureDataset(Array.from({ length: 24 }, (_, index) => {
    const date = new Date(Date.UTC(2023, 3 + index, 1));
    return date.toISOString().slice(0, 10);
  }));

  render(<MlForecastTab />);

  expect(await screen.findByText('Usable Months')).toBeTruthy();
  expect(screen.getByText('Future Months')).toBeTruthy();
  expect(screen.getByText('3 months')).toBeTruthy();
  expect(screen.getByText('Target: sales')).toBeTruthy();
  expect(screen.getByText('11 generated features')).toBeTruthy();
});

it('keeps period wording for a daily series', async () => {
  configureDataset(Array.from({ length: 24 }, (_, index) => {
    const date = new Date(Date.UTC(2025, 0, 1 + index));
    return date.toISOString().slice(0, 10);
  }));

  render(<MlForecastTab />);

  expect(await screen.findByText('Usable Periods')).toBeTruthy();
  expect(screen.getByText('Future Periods')).toBeTruthy();
  expect(screen.getByText('3 periods')).toBeTruthy();
});
