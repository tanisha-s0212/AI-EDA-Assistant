import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import UnderstandingTab from './understanding-tab';

const { postMock, state } = vi.hoisted(() => ({
  postMock: vi.fn(),
  state: {
    datasetId: 'sample', fileName: 'sales.csv', totalRows: 30, loadedRowCount: 15,
    previewLoaded: true, duplicates: 0, memoryUsage: '1 KB', selectedSheets: [], sheetMergeMode: 'single',
    cleanedData: null, datasetDescription: '', datasetParseOptions: null, confirmedSalesMapping: null,
    rawData: Array.from({ length: 15 }, (_, index) => ({ month: `2024-${String(index + 1).padStart(2, '0')}-01`, sales: index })),
    columns: [
      { name: 'month', dtype: 'date', role: 'datetime', nonNull: 15, nullCount: 0, uniqueCount: 15, sample: [] },
      { name: 'sales', dtype: 'number', role: 'numeric', nonNull: 15, nullCount: 0, uniqueCount: 15, sample: [] },
    ],
  },
}));

vi.mock('@/lib/store', () => ({
  useAppStore: Object.assign((selector: (value: typeof state) => unknown) => selector(state), {
    getState: () => state,
    setState: (patch: Partial<typeof state>) => Object.assign(state, patch),
  }),
}));
vi.mock('@/lib/api', () => ({
  apiClient: { post: postMock },
  getApiErrorMessage: (error: unknown, fallback: string) => error instanceof Error ? error.message : fallback,
}));

describe('Data Understanding', () => {
  beforeEach(() => {
    postMock.mockReset();
    postMock.mockImplementation((path: string) => {
      if (path === '/sales/readiness') return Promise.resolve({ data: { status: 'ready', date_column: 'month', target_column: 'sales', period_count: 24, period_label: 'month' } });
      if (path === '/dataset-quality-summary') return Promise.resolve({ data: { row_count: 30, missing_cells: 2, duplicate_rows: 1, columns: [{ name: 'sales', null_count: 2, unique_count: 28 }] } });
      return Promise.resolve({ data: { rows_analyzed: 30, grain: { grain: 'monthly_aggregate', confidence: 'validated' }, coverage: { first_month: '2023-01', last_month: '2024-12', observed_months: 24, usable_months: 24, missing_months: [], missing_month_count: 0, invalid_date_rows: 0, invalid_target_rows: 0, duplicate_key_rows: null, meets_minimum: true } } });
    });
  });

  it('labels the preview, pages loaded rows, and reviews confirmed columns', async () => {
    render(<UnderstandingTab />);
    expect(screen.getByText(/preview of 30 total rows/i)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText(/Rows 11–15 of 15 loaded rows/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Spread sample' }));
    expect(screen.getByText(/10 evenly spaced rows from the loaded view/)).toBeTruthy();
    await waitFor(() => expect((screen.getByLabelText('Date column') as HTMLSelectElement).value).toBe('month'));
    fireEvent.click(screen.getByRole('button', { name: 'Review and confirm mapping' }));
    expect(await screen.findByText(/24 observed months/)).toBeTruthy();
    expect(postMock).toHaveBeenCalledWith('/dataset-understanding', expect.objectContaining({ dataset_id: 'sample', date_column: 'month', target_column: 'sales' }));
    fireEvent.click(screen.getByRole('button', { name: 'Check full dataset' }));
    expect(await screen.findByText(/30 rows checked; 2 missing cells; 1 duplicate rows/)).toBeTruthy();
  });
});
