import { describe, expect, test } from 'vitest';
import {
  shouldEnableSalesPreset,
  scoreSalesRevenueColumn,
  pickPreferredSalesDateColumn,
  scoreSalesDateColumn,
  isDatePartColumn,
} from './sales-domain';

describe('shouldEnableSalesPreset', () => {
  test('is false for EV / charging-style columns', () => {
    const columns = [
      { name: 'sessionId' },
      { name: 'kwhTotal', role: 'numeric' },
      { name: 'dollars', role: 'numeric' },
      { name: 'created', role: 'datetime' },
    ];
    expect(shouldEnableSalesPreset(columns)).toBe(false);
    expect(scoreSalesRevenueColumn('dollars')).toBeLessThan(70);
    expect(scoreSalesRevenueColumn('amount')).toBeLessThan(70);
  });

  test('is true for strong sales/revenue column names', () => {
    const columns = [
      { name: 'year_month', role: 'datetime' },
      { name: 'total_total_value_sale_free', role: 'numeric' },
      { name: 'region' },
    ];
    expect(shouldEnableSalesPreset(columns)).toBe(true);
    expect(scoreSalesRevenueColumn('total_total_value_sale_free')).toBeGreaterThanOrEqual(70);
  });
});

describe('pickPreferredSalesDateColumn', () => {
  test('prefers created over month/dayofweek/date parts', () => {
    const columns = [
      { name: 'created', role: 'datetime', uniqueCount: 3000 },
      { name: 'year', role: 'numeric', uniqueCount: 3 },
      { name: 'month', role: 'numeric', uniqueCount: 12 },
      { name: 'dayofweek', role: 'numeric', uniqueCount: 7 },
      { name: 'date', role: 'date', uniqueCount: 400 },
    ];
    expect(isDatePartColumn('month')).toBe(true);
    expect(scoreSalesDateColumn('month')).toBe(0);
    expect(scoreSalesDateColumn('dayofweek')).toBe(0);
    expect(pickPreferredSalesDateColumn(columns)).toBe('created');
  });

  test('still prefers year_month for sales panels', () => {
    expect(
      pickPreferredSalesDateColumn([
        { name: 'year_month', role: 'datetime', uniqueCount: 36 },
        { name: 'region' },
      ]),
    ).toBe('year_month');
  });
});
