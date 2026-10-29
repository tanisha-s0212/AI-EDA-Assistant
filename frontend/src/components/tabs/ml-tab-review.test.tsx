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

it('reveals the target and feature review UI after analyzing a supervised question', async () => {
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
  expect(screen.queryByRole('tab', { name: 'Ask Assistant' })).toBeNull();
  expect(screen.queryByRole('tab', { name: 'Train Manually' })).toBeNull();
  expect(screen.queryByRole('combobox', { name: 'Target column' })).toBeNull();
  fireEvent.change(screen.getByPlaceholderText(/^Example:/), { target: { value: 'Predict outcome from x' } });
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Question' }));
  const target = await screen.findByRole('combobox', { name: 'Target column' });
  expect(target).toHaveTextContent('outcome');
  expect(screen.getByRole('checkbox', { name: /^x/ })).toBeChecked();
  expect(screen.getByText(/question identified a supervised prediction task/i)).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Confirm and train' })).toBeNull();
  await waitFor(() => expect(assistantRequests).toBe(1));
  expect(postMock.mock.calls.filter(([path]) => path === '/assistant/ask')).toHaveLength(1);
});

it('shows feature leakage warnings in the question-driven training review', async () => {
  postMock.mockImplementation((path: string) => path === '/ml/feature-review'
    ? Promise.resolve({ data: { features: ['x'], warnings: ['The target may be derived from selected inputs.'], excluded: {} } })
    : Promise.resolve({ data: {
      status: 'needs_confirmation', answer: 'Review the target and features before training.',
      ml_plan: { target: 'outcome', task: 'regression', features: ['x'], review_token: 'review-token', warnings: [] },
    } }));
  render(<MlTab />);
  fireEvent.change(screen.getByPlaceholderText(/^Example:/), { target: { value: 'Estimate outcome from x' } });
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Question' }));
  await screen.findByText('The target may be derived from selected inputs.');
  const acknowledgement = screen.getByRole('checkbox', { name: /reviewed the feature warnings/i }) as HTMLInputElement;
  expect(acknowledgement.checked).toBe(false);
  fireEvent.click(acknowledgement);
  expect(acknowledgement.checked).toBe(true);
});
