import { describe, expect, test } from 'vitest';
import { shouldEnableSalesPreset, scoreSalesRevenueColumn } from './sales-domain';

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
