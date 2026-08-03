import { describe, expect, test } from 'vitest';
import {
  shouldEnableSalesPreset,
  scoreSalesRevenueColumn,
  pickPreferredSalesDateColumn,
  scoreSalesDateColumn,
  isDatePartColumn,
  isForecastDateColumnCandidate,
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
  const stationColumns = [
    { name: 'created', role: 'datetime' as const, uniqueCount: 3000 },
    { name: 'year', role: 'numeric' as const, uniqueCount: 3 },
    { name: 'month', role: 'numeric' as const, uniqueCount: 12 },
    { name: 'dayofweek', role: 'numeric' as const, uniqueCount: 7 },
    { name: 'date', role: 'date' as const, uniqueCount: 400 },
  ];

  test('prefers created over month/dayofweek/date parts', () => {
    expect(isDatePartColumn('month')).toBe(true);
    expect(scoreSalesDateColumn('month')).toBe(0);
    expect(scoreSalesDateColumn('dayofweek')).toBe(0);
    expect(pickPreferredSalesDateColumn(stationColumns)).toBe('created');
  });

  test('still prefers year_month for sales panels', () => {
    expect(
      pickPreferredSalesDateColumn([
        { name: 'year_month', role: 'datetime', uniqueCount: 36 },
        { name: 'region' },
      ]),
    ).toBe('year_month');
  });

  test('forecast date candidates include created and exclude month', () => {
    expect(isForecastDateColumnCandidate({ name: 'created', role: 'datetime' })).toBe(true);
    expect(isForecastDateColumnCandidate({ name: 'ended', role: 'string' })).toBe(true);
    expect(isForecastDateColumnCandidate({ name: 'month', role: 'numeric' })).toBe(false);
    expect(isForecastDateColumnCandidate({ name: 'dayofweek', role: 'numeric' })).toBe(false);
  });

  test('TS and ML tabs share the same preferred date for station-like schemas', () => {
    // Both tabs call pickPreferredSalesDateColumn — one source of truth.
    const preferred = pickPreferredSalesDateColumn(stationColumns);
    expect(preferred).toBe('created');
    expect(isDatePartColumn(preferred)).toBe(false);
    // Stale date-part selections should be treated as auto-correct candidates.
    expect(isDatePartColumn('month')).toBe(true);
  });
});
