import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import MlTab from './ml-tab';
import { useAppStore } from '@/lib/store';

Element.prototype.scrollIntoView = vi.fn();

const { postMock } = vi.hoisted(() => ({ postMock: vi.fn() }));
vi.mock('@/lib/api', () => ({
  apiClient: { post: postMock, get: vi.fn() },
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
  getApiErrorDetail: () => undefined,
}));

beforeEach(() => {
  postMock.mockReset();
  postMock.mockResolvedValue({ data: { warnings: [], excluded: {} } });
  const features = Array.from({ length: 12 }, (_, index) => `feature_${index}`);
  const names = ['sales', 'qty', ...features];
  useAppStore.setState({
    datasetId: 'human-review-test', mlWorkflowStep: 1,
    activeTab: 'ml', modelId: null, modelTrained: false, targetColumn: null,
    rawData: Array.from({ length: 4 }, (_, row) => Object.fromEntries(names.map((name, index) => [name, row + index + 1]))),
    cleanedData: null,
    columns: names.map(name => ({ name, dtype: 'number', role: 'numeric' as const, nonNull: 4, nullCount: 0, uniqueCount: 4, sample: [] })),
  });
});

it('keeps manual feature and problem-type choices and reviews features beyond the initial suggestion', async () => {
  render(<MlTab />);
  fireEvent.click(screen.getByRole('tab', { name: 'Train Manually' }));
  await screen.findByText('Suggested target: sales');
  const firstFeature = screen.getByRole('checkbox', { name: /^feature_0/ });
  const laterFeature = screen.getByRole('checkbox', { name: /^feature_11/ });
  expect(firstFeature).toBeChecked();
  expect(laterFeature).not.toBeChecked();

  fireEvent.click(firstFeature);
  fireEvent.click(laterFeature);
  fireEvent.click(screen.getByRole('button', { name: /Classification Predict categories/ }));

  await waitFor(() => {
    expect(firstFeature).not.toBeChecked();
    expect(laterFeature).toBeChecked();
    expect(screen.getByRole('button', { name: /Classification Predict categories/ })).toHaveClass('bg-secondary');
    const reviews = postMock.mock.calls.filter(([path]) => path === '/ml/feature-review');
    const lastReview = reviews[reviews.length - 1][1];
    expect(lastReview.target_column).toBe('sales');
    expect(lastReview.feature_columns).toContain('feature_11');
    expect(lastReview.feature_columns).not.toContain('feature_0');
  });
});

it('updates the review for a manually chosen target while preserving valid feature choices', async () => {
  render(<MlTab />);
  fireEvent.click(screen.getByRole('tab', { name: 'Train Manually' }));
  await screen.findByText('Suggested target: sales');
  fireEvent.click(screen.getByRole('checkbox', { name: /^feature_11/ }));
  fireEvent.keyDown(screen.getByRole('combobox', { name: 'Target column' }), { key: 'ArrowDown' });
  fireEvent.click(await screen.findByRole('option', { name: /qty/ }));

  await waitFor(() => {
    expect(screen.getByRole('combobox', { name: 'Target column' })).toHaveTextContent('qty');
    expect(screen.getByRole('checkbox', { name: /^feature_11/ })).toBeChecked();
    const reviews = postMock.mock.calls.filter(([path]) => path === '/ml/feature-review');
    const lastReview = reviews[reviews.length - 1][1];
    expect(lastReview.target_column).toBe('qty');
    expect(lastReview.feature_columns).toContain('feature_11');
    expect(lastReview.feature_columns).not.toContain('qty');
  });
  expect(screen.getByText('Suggested target: sales')).toBeInTheDocument();
});

it('labels an assistant training result separately from the manual target suggestion', async () => {
  postMock.mockImplementation((path: string) => path === '/ml/feature-review'
    ? Promise.resolve({ data: { warnings: [], excluded: {} } })
    : Promise.resolve({ data: {
      status: 'success', intent: 'regression', routing: { intent: 'regression', confidence: 83 },
      model: 'Gradient Boosting', model_id: 'model-1', target: 'qty', features: ['feature_0'],
      metrics: { RMSE: 5, R2: 0.1 }, evaluation: { rows: 10, method: 'holdout split' },
      warnings: ['Chronological holdout was used.'],
      answer: 'Full training details and chronological holdout warning.',
    } }));

  render(<MlTab />);
  fireEvent.change(screen.getByPlaceholderText(/^Example:/), { target: { value: 'Predict qty' } });
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Problem' }));

  expect(await screen.findByText('Assistant training result')).toBeVisible();
  expect(screen.getByText(/target: qty/)).toBeVisible();
  expect(screen.queryByText('Suggested target: sales')).toBeNull();
  expect(screen.getByText(/intent match score: 83% \(not prediction confidence\)/)).toBeVisible();
  expect(screen.getByText('Chronological holdout was used.')).toBeVisible();
  expect(screen.getByText('No single-row prediction was made. Provide input values in the Prediction tab to score a row.')).toBeVisible();
  expect(screen.getByText('Full assistant response').closest('details')).not.toHaveAttribute('open');
  fireEvent.click(screen.getByRole('tab', { name: 'Train Manually' }));
  await waitFor(() => expect(screen.getByText('Suggested target: sales')).toBeVisible());
  expect(screen.queryByText('Assistant training result')).toBeNull();
  fireEvent.click(screen.getByRole('tab', { name: 'Ask Assistant' }));
  expect(await screen.findByText('Assistant training result')).toBeVisible();
  expect(screen.queryByText('Suggested target: sales')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Use this model in Prediction' }));
  expect(useAppStore.getState().modelId).toBe('model-1');
  expect(useAppStore.getState().targetColumn).toBe('qty');
  expect(useAppStore.getState().selectedFeatures).toEqual(['feature_0']);
  expect(useAppStore.getState().modelTrained).toBe(true);
  expect(useAppStore.getState().activeTab).toBe('prediction');
});
