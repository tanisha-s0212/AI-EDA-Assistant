import React from 'react';
import { render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import StepNavigator from './step-navigator';
import { useAppStore } from '@/lib/store';

beforeEach(() => {
  useAppStore.setState({
    activeTab: 'forecast_ts',
    rawData: [{ period: '2025-01-01', amount: 12 }],
    cleaningDone: true,
    modelTrained: true,
  });
});

it('hides the currency selector while retaining forecast navigation', () => {
  render(<StepNavigator />);

  expect(screen.queryByText('Currency')).toBeNull();
  expect(screen.queryByRole('button', { name: 'USD' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'INR' })).toBeNull();
  expect(screen.getAllByRole('button', { name: /Time Series Forecast/ })).toHaveLength(2);
  expect(screen.getByRole('button', { name: /Next: Machine Learning Forecast/ })).toBeTruthy();
});
