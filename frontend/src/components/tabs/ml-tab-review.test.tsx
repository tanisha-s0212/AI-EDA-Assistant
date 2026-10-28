import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import MlTab from './ml-tab';
import { useAppStore } from '@/lib/store';

const { postMock } = vi.hoisted(() => ({ postMock: vi.fn() }));
vi.mock('@/lib/api', () => ({
  apiClient: { post: postMock, get: vi.fn() },
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
  getApiErrorDetail: () => undefined,
}));

beforeEach(() => {
  postMock.mockReset();
  useAppStore.setState({
    datasetId: 'review-test', fileName: 'synthetic.csv', totalRows: 4, loadedRowCount: 4,
    rawData: [{ x: 1, outcome: 2 }, { x: 2, outcome: 5 }, { x: 3, outcome: 7 }, { x: 4, outcome: 9 }],
    cleanedData: null,
    columns: [
      { name: 'x', dtype: 'number', role: 'numeric', nonNull: 4, nullCount: 0, uniqueCount: 4, sample: [] },
      { name: 'outcome', dtype: 'number', role: 'numeric', nonNull: 4, nullCount: 0, uniqueCount: 4, sample: [] },
    ],
  });
});

it('shows the chosen target and features before confirming assistant training', async () => {
  let assistantRequests = 0;
  postMock.mockImplementation((path: string) => {
    if (path === '/ml/feature-review') return Promise.resolve({ data: { warnings: [], excluded: {} } });
    assistantRequests += 1;
    return Promise.resolve({ data: assistantRequests === 1
      ? { status: 'needs_confirmation', answer: 'Review the target and features before training.',
          ml_plan: { target: 'outcome', task: 'regression', features: ['x'], review_token: 'review-token', warnings: [] } }
      : { status: 'success', answer: 'Training completed.' } });
  });
  render(<MlTab />);
  fireEvent.change(screen.getByPlaceholderText(/^Example:/), { target: { value: 'Predict outcome from x' } });
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Problem' }));
  await screen.findByText('Target: outcome (regression)');
  expect(screen.getByText('Features: x')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and train' }));
  await waitFor(() => expect(assistantRequests).toBe(2));
  const requests = postMock.mock.calls.filter(([path]) => path === '/assistant/ask');
  expect(requests[1][1]).toMatchObject({ confirm_training: true, review_token: 'review-token', feature_columns: ['x'] });
});

it('shows feature leakage warnings before manual training', async () => {
  postMock.mockResolvedValue({ data: { features: ['x'], warnings: ['The target may be derived from selected inputs.'], excluded: {} } });
  render(<MlTab />);
  fireEvent.click(screen.getByRole('tab', { name: 'Train Manually' }));
  await screen.findByText('The target may be derived from selected inputs.');
  const acknowledgement = screen.getByRole('checkbox', { name: /reviewed the feature warnings/i }) as HTMLInputElement;
  expect(acknowledgement.checked).toBe(false);
  fireEvent.click(acknowledgement);
  expect(acknowledgement.checked).toBe(true);
});
