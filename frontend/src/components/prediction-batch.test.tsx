import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PredictionBatch } from './prediction-batch';

const mocks = vi.hoisted(() => ({ post: vi.fn(), clearStale: vi.fn() }));
vi.mock('@/lib/api', () => ({ apiClient: { post: mocks.post }, getApiErrorMessage: () => 'Batch failed' }));
vi.mock('@/lib/model-lifecycle', () => ({ clearStaleModelState: mocks.clearStale }));

describe('batch prediction results', () => {
  beforeEach(() => { vi.clearAllMocks(); });
  afterEach(cleanup);

  it('shows rejected rows and warning details alongside successful predictions', async () => {
    mocks.post.mockResolvedValue({ data: {
      summary: { total: 2, succeeded: 1, failed: 1, with_warnings: 1 },
      results: [
        { row: 1, status: 'success', prediction: 12.3456, warnings: ['Unseen category'] },
        { row: 2, status: 'failed', error: 'amount must be a finite number' },
      ],
    } });
    const { container } = render(<PredictionBatch modelId="model" datasetId="dataset" />);
    fireEvent.change(container.querySelector('input[type=file]')!, { target: { files: [new File(['amount\n1\n'], 'data.csv')] } });
    expect(await screen.findByText(/2 rows: 1 succeeded, 1 rejected/)).toBeTruthy();
    expect(screen.getByText('12.3456')).toBeTruthy();
    expect(screen.getByText('Unseen category')).toBeTruthy();
    expect(screen.getByText('amount must be a finite number')).toBeTruthy();
    expect(screen.getByText('Export all results and errors')).toBeTruthy();
  });

  it('ignores a stale error from a component belonging to the previous model', async () => {
    let reject!: (reason: unknown) => void;
    mocks.post.mockReturnValue(new Promise((_, fail) => { reject = fail; }));
    const { container, unmount } = render(<PredictionBatch modelId="old-model" datasetId="dataset" />);
    fireEvent.change(container.querySelector('input[type=file]')!, { target: { files: [new File(['a\n1'], 'data.csv')] } });
    unmount();
    await act(async () => { reject({ response: { status: 409 } }); });
    expect(mocks.clearStale).not.toHaveBeenCalled();
  });
});
