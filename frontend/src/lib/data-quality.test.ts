import { describe, expect, test } from 'vitest';
import { computeColumnMissingPercent, computeProfileCompleteness } from './data-quality';

describe('computeProfileCompleteness', () => {
  test('uses profile cell totals, not full totalRows (Superstore-style preview mismatch)', () => {
    // Preview profile: 5000 rows × 18 cols, fully populated; full file is 9800 rows.
    const columns = Array.from({ length: 18 }, (_, index) => ({
      nonNull: 5000,
      nullCount: 0,
      name: `col_${index}`,
    }));

    // Bug would divide by 9800×18 → ~51%. Correct profile rate is 100%.
    expect(computeProfileCompleteness(columns)).toBe(100);

    const badLegacy =
      Math.round((columns.reduce((sum, col) => sum + col.nonNull, 0) / (9800 * columns.length)) * 1000) / 10;
    expect(badLegacy).toBe(51);
  });

  test('reports 90% when profile has a deliberate 10% null rate', () => {
    const columns = [
      { nonNull: 4500, nullCount: 500 },
      { nonNull: 4500, nullCount: 500 },
    ];
    expect(computeProfileCompleteness(columns)).toBe(90);

    // Must not mix with a larger totalRows denominator.
    const wrong =
      Math.round((columns.reduce((sum, col) => sum + col.nonNull, 0) / (20000 * columns.length)) * 1000) / 10;
    expect(wrong).toBe(22.5);
  });

  test('returns 0 for empty column list', () => {
    expect(computeProfileCompleteness([])).toBe(0);
  });
});

describe('computeColumnMissingPercent', () => {
  test('matches per-column null share inside the profile', () => {
    expect(computeColumnMissingPercent({ nonNull: 4500, nullCount: 500 })).toBe(10);
    expect(computeColumnMissingPercent({ nonNull: 5000, nullCount: 0 })).toBe(0);
  });
});
