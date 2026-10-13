import { describe, expect, test } from 'vitest';
import {
  formatAxisCurrency,
  formatCompactCurrency,
  formatCurrency,
  formatNumber,
  inferCurrencyFromColumns,
} from './currency-format';

describe('currency-format', () => {
  test('formats USD currency with US grouping', () => {
    expect(formatCurrency(1094538, 'USD')).toBe('$1,094,538');
  });

  test('formats INR currency with Indian grouping', () => {
    expect(formatCurrency(1094538, 'INR')).toBe('₹10,94,538');
  });

  test('formats compact USD suffixes', () => {
    expect(formatCompactCurrency(1_500_000, 'USD')).toBe('$1.5M');
    expect(formatCompactCurrency(12_000, 'USD')).toBe('$12K');
  });

  test('formats compact INR suffixes', () => {
    expect(formatCompactCurrency(1_50_00_000, 'INR')).toBe('₹1.5 Cr');
    expect(formatCompactCurrency(2_50_000, 'INR')).toBe('₹2.5 L');
  });

  test('formats axis labels per currency', () => {
    expect(formatAxisCurrency(1_500_000, 'USD')).toBe('$1.5M');
    expect(formatAxisCurrency(2_50_000, 'INR')).toBe('₹2.5L');
  });

  test('formats plain numbers using locale', () => {
    expect(formatNumber(1094538.33, 'USD', 2)).toBe('1,094,538.33');
    expect(formatNumber(1094538.33, 'INR', 2)).toBe('10,94,538.33');
  });

  test('infers USD from column names containing usd', () => {
    expect(inferCurrencyFromColumns([
      { name: 'net_sales' },
      { name: 'avg_selling_price_usd' },
      { name: 'backlog_usd' },
    ])).toBe('USD');
  });

  test('returns null when no USD hint exists', () => {
    expect(inferCurrencyFromColumns([
      { name: 'net_sales' },
      { name: 'plant_region' },
    ])).toBeNull();
  });
});
