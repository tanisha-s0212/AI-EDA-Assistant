export type ForecastSeriesProfile = {
  detected_frequency: string;
  usable_periods: number;
  volatility: number;
  zero_value_share: number;
};

type DataLikeRow = Record<string, unknown>;

/**
 * Local preview-row heuristic only. Counts valid date+target rows in the browser frame
 * (often capped by DATASET_PREVIEW_ROW_LIMIT) — NOT aggregated unique periods.
 */
export function inferPreviewSeriesProfile(
  data: DataLikeRow[],
  dateColumn: string,
  targetColumn: string,
): ForecastSeriesProfile {
  const points = data
    .map((row) => ({ date: new Date(String(row[dateColumn] ?? '')), value: Number(row[targetColumn]) }))
    .filter((item) => !Number.isNaN(item.date.getTime()) && Number.isFinite(item.value))
    .sort((left, right) => left.date.getTime() - right.date.getTime());

  if (points.length < 2) {
    return { detected_frequency: 'period', usable_periods: points.length, volatility: 0, zero_value_share: 0 };
  }

  const values = points.map((item) => item.value);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const diffs = points
    .slice(1)
    .map((item, index) => (item.date.getTime() - points[index].date.getTime()) / 86400000)
    .sort((a, b) => a - b);
  const medianDays = diffs[Math.floor(diffs.length / 2)] ?? 30;
  const detected_frequency =
    medianDays <= 2 ? 'day' : medianDays <= 10 ? 'week' : medianDays <= 45 ? 'month' : medianDays <= 120 ? 'quarter' : 'year';

  return {
    detected_frequency,
    usable_periods: points.length,
    volatility: mean === 0 ? 0 : Math.sqrt(variance) / Math.abs(mean),
    zero_value_share: values.filter((value) => value === 0).length / values.length,
  };
}

/**
 * Prefer backend aggregated period counts (Understanding /sales/readiness, stationarity, run
 * dataset_profile) over preview-row local heuristics.
 */
export function resolveForecastSeriesProfile(options: {
  localProfile: ForecastSeriesProfile;
  resultProfile?: Partial<ForecastSeriesProfile> | null;
  backendPeriodCount?: number | null;
  backendFrequency?: string | null;
  backendVolatility?: number | null;
  backendZeroShare?: number | null;
}): ForecastSeriesProfile {
  const {
    localProfile,
    resultProfile,
    backendPeriodCount,
    backendFrequency,
    backendVolatility,
    backendZeroShare,
  } = options;

  const resultPeriods = resultProfile?.usable_periods;
  if (typeof resultPeriods === 'number' && resultPeriods > 0) {
    return {
      detected_frequency:
        resultProfile?.detected_frequency || backendFrequency || localProfile.detected_frequency,
      usable_periods: resultPeriods,
      volatility: resultProfile?.volatility ?? backendVolatility ?? localProfile.volatility,
      zero_value_share:
        resultProfile?.zero_value_share ?? backendZeroShare ?? localProfile.zero_value_share,
    };
  }

  if (typeof backendPeriodCount === 'number' && backendPeriodCount > 0) {
    return {
      detected_frequency: backendFrequency || localProfile.detected_frequency,
      usable_periods: backendPeriodCount,
      volatility: backendVolatility ?? localProfile.volatility,
      zero_value_share: backendZeroShare ?? localProfile.zero_value_share,
    };
  }

  return localProfile;
}
