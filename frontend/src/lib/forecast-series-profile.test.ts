import { describe, expect, test } from 'vitest';
import {
  inferPreviewSeriesProfile,
  resolveForecastSeriesProfile,
  type ForecastSeriesProfile,
} from './forecast-series-profile';

describe('inferPreviewSeriesProfile', () => {
  test('counts preview rows, not aggregated unique calendar periods', () => {
    // Simulates capped preview: 5000 dated sales rows (many same-day orders)
    const data = Array.from({ length: 5000 }, (_, index) => ({
      order_date: `2014-01-${String((index % 28) + 1).padStart(2, '0')}`,
      sales: 10 + index,
    }));
    const profile = inferPreviewSeriesProfile(data, 'order_date', 'sales');
    expect(profile.usable_periods).toBe(5000);
  });
});

describe('resolveForecastSeriesProfile', () => {
  const localPreview: ForecastSeriesProfile = {
    detected_frequency: 'day',
    usable_periods: 5000,
    volatility: 0.4,
    zero_value_share: 0.01,
  };

  test('prefers Understanding /sales/readiness period_count over preview row count', () => {
    const understandingPeriodCount = 1458;
    const resolved = resolveForecastSeriesProfile({
      localProfile: localPreview,
      backendPeriodCount: understandingPeriodCount,
      backendFrequency: 'day',
    });
    expect(resolved.usable_periods).toBe(understandingPeriodCount);
    expect(resolved.usable_periods).not.toBe(localPreview.usable_periods);
    expect(resolved.detected_frequency).toBe('day');
  });

  test('prefers backend dataset_profile over both preview and readiness', () => {
    const resolved = resolveForecastSeriesProfile({
      localProfile: localPreview,
      backendPeriodCount: 1458,
      resultProfile: {
        detected_frequency: 'day',
        usable_periods: 1458,
        volatility: 0.55,
        zero_value_share: 0.02,
      },
    });
    expect(resolved.usable_periods).toBe(1458);
    expect(resolved.volatility).toBe(0.55);
    expect(resolved.zero_value_share).toBe(0.02);
  });

  test('falls back to local preview only when no backend period count exists', () => {
    const resolved = resolveForecastSeriesProfile({
      localProfile: localPreview,
      backendPeriodCount: null,
      resultProfile: null,
    });
    expect(resolved.usable_periods).toBe(5000);
  });

  test('TS readiness card parity fixture: same dataset/date implies equal periods', () => {
    // Shared regression contract: Understanding period_count === TS/ML usable_periods
    // once backend aggregated count is available.
    const understandingPeriodCount = 1458;
    const tsResolved = resolveForecastSeriesProfile({
      localProfile: localPreview,
      backendPeriodCount: understandingPeriodCount,
      backendFrequency: 'day',
    });
    const mlResolved = resolveForecastSeriesProfile({
      localProfile: localPreview,
      backendPeriodCount: understandingPeriodCount,
    });
    expect(tsResolved.usable_periods).toBe(understandingPeriodCount);
    expect(mlResolved.usable_periods).toBe(understandingPeriodCount);
    expect(tsResolved.usable_periods).toBe(mlResolved.usable_periods);
  });
});
