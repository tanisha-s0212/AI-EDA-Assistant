import type { ColumnInfo, DataRow } from '@/lib/store';

const EDA_SAMPLE_LIMIT = 5000;

function sampleRows(data: DataRow[], limit = EDA_SAMPLE_LIMIT): DataRow[] {
  if (data.length <= limit) return data;

  const step = Math.max(1, Math.floor(data.length / limit));
  const sampled: DataRow[] = [];
  for (let index = 0; index < data.length && sampled.length < limit; index += step) {
    sampled.push(data[index]);
  }
  return sampled;
}

export type EdaStats = {
  numericColumns: string[];
  categoricalColumns: string[];
  stats: Record<string, Record<string, number>>;
  correlations: { pair: string; correlation: number }[];
};

export type HeatmapCell = {
  x: string;
  y: string;
  value: number;
};

export type OutlierSummary = {
  column: string;
  lowerBound: number;
  upperBound: number;
  outliers: number;
  outlierRate: number;
  sampleSize: number;
  severity: 'high' | 'moderate' | 'low' | 'none';
  note: string;
};

export function computeEdaStats(rawData: DataRow[], columns: ColumnInfo[]): EdaStats {
  if (!rawData.length || !columns.length) {
    return { numericColumns: [], categoricalColumns: [], stats: {}, correlations: [] };
  }

  const analysisRows = sampleRows(rawData);

  const numericColumns = columns
    .filter((column) => {
      const dtype = String(column.dtype).toLowerCase();
      return column.role === 'numeric' || /int|float|double|decimal|number/.test(dtype);
    })
    .map((column) => column.name);

  const categoricalColumns = columns
    .filter((column) => {
      const dtype = String(column.dtype).toLowerCase();
      return column.role === 'categorical' || column.role === 'boolean' || /string|str|utf8|object|bool/.test(dtype);
    })
    .map((column) => column.name);

  const stats: Record<string, Record<string, number>> = {};

  for (const column of numericColumns) {
    const values = analysisRows
      .map((row) => row[column])
      .filter((value): value is number => typeof value === 'number' && !Number.isNaN(value));

    if (!values.length) continue;

    const sorted = [...values].sort((a, b) => a - b);
    const count = sorted.length;
    const mean = sorted.reduce((sum, value) => sum + value, 0) / count;
    const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
    const std = Math.sqrt(variance);
    const q1 = sorted[Math.floor((count - 1) * 0.25)];
    const median =
      count % 2 === 0
        ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        : sorted[Math.floor(count / 2)];
    const q3 = sorted[Math.floor((count - 1) * 0.75)];

    stats[column] = {
      count,
      mean: Math.round(mean * 100) / 100,
      std: Math.round(std * 100) / 100,
      min: Math.round(sorted[0] * 100) / 100,
      q1: Math.round(q1 * 100) / 100,
      median: Math.round(median * 100) / 100,
      q3: Math.round(q3 * 100) / 100,
      max: Math.round(sorted[count - 1] * 100) / 100,
    };
  }

  for (const column of categoricalColumns) {
    const values = analysisRows
      .map((row) => row[column])
      .filter((value) => value !== null && value !== undefined && value !== '');
    stats[column] = {
      count: values.length,
      unique: new Set(values).size,
    };
  }

  const correlations: { pair: string; correlation: number }[] = [];
  const topNumericColumns = numericColumns.slice(0, 6);

  for (let i = 0; i < topNumericColumns.length; i++) {
    for (let j = i + 1; j < topNumericColumns.length; j++) {
      const left = topNumericColumns[i];
      const right = topNumericColumns[j];
      const pairs: [number, number][] = [];

      for (const row of analysisRows) {
        const a = row[left];
        const b = row[right];
        if (typeof a === 'number' && typeof b === 'number' && !Number.isNaN(a) && !Number.isNaN(b)) {
          pairs.push([a, b]);
        }
      }

      if (pairs.length < 2) continue;

      const count = pairs.length;
      const meanA = pairs.reduce((sum, pair) => sum + pair[0], 0) / count;
      const meanB = pairs.reduce((sum, pair) => sum + pair[1], 0) / count;

      let numerator = 0;
      let denominatorA = 0;
      let denominatorB = 0;

      for (const [a, b] of pairs) {
        const deltaA = a - meanA;
        const deltaB = b - meanB;
        numerator += deltaA * deltaB;
        denominatorA += deltaA ** 2;
        denominatorB += deltaB ** 2;
      }

      const denominator = Math.sqrt(denominatorA * denominatorB);
      if (!denominator) continue;

      correlations.push({
        pair: `${left} vs ${right}`,
        correlation: Math.round((numerator / denominator) * 100) / 100,
      });
    }
  }

  correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

  return { numericColumns, categoricalColumns, stats, correlations };
}

export function buildCorrelationHeatmap(data: DataRow[], numericColumns: string[]): HeatmapCell[] {
  const columns = numericColumns.slice(0, 5);
  const cells: HeatmapCell[] = [];

  const getCorrelation = (left: string, right: string) => {
    const pairs: [number, number][] = [];
    for (const row of data) {
      const a = row[left];
      const b = row[right];
      if (typeof a === 'number' && typeof b === 'number' && !Number.isNaN(a) && !Number.isNaN(b)) {
        pairs.push([a, b]);
      }
    }
    if (pairs.length < 3) return 0;
    const meanA = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
    const meanB = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
    let numerator = 0;
    let denomA = 0;
    let denomB = 0;
    for (const [a, b] of pairs) {
      const da = a - meanA;
      const db = b - meanB;
      numerator += da * db;
      denomA += da * da;
      denomB += db * db;
    }
    const denominator = Math.sqrt(denomA * denomB);
    return denominator === 0 ? 0 : Math.round((numerator / denominator) * 1000) / 1000;
  };

  columns.forEach((x) => {
    columns.forEach((y) => {
      cells.push({ x, y, value: x === y ? 1 : getCorrelation(x, y) });
    });
  });

  return cells;
}

function medianOf(values: number[]): number {
  if (!values.length) return 0;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle];
}

export function detectOutliers(
  data: DataRow[],
  numericColumns: string[],
  stats: EdaStats['stats']
): OutlierSummary[] {
  return numericColumns
    .slice(0, 8)
    .map<OutlierSummary>((column) => {
      const values = data
        .map((row) => row[column])
        .filter((v): v is number => typeof v === 'number' && !Number.isNaN(v));
      const q1 = stats[column]?.q1;
      const q3 = stats[column]?.q3;
      if (!values.length || q1 === undefined || q3 === undefined) {
        return {
          column,
          lowerBound: 0,
          upperBound: 0,
          outliers: 0,
          outlierRate: 0,
          sampleSize: 0,
          severity: 'none',
          note: 'Not enough numeric data.',
        };
      }

      const sorted = [...values].sort((a, b) => a - b);
      const uniqueCount = new Set(sorted.map((value) => value.toFixed(8))).size;
      const median = medianOf(sorted);
      const deviations = sorted
        .map((value) => Math.abs(value - median))
        .sort((a, b) => a - b);
      const mad = medianOf(deviations);
      const iqr = q3 - q1;

      if (sorted.length < 20) {
        return {
          column,
          lowerBound: Math.round((q1 - 1.5 * iqr) * 100) / 100,
          upperBound: Math.round((q3 + 1.5 * iqr) * 100) / 100,
          outliers: 0,
          outlierRate: 0,
          sampleSize: sorted.length,
          severity: 'none',
          note: 'Needs at least 20 numeric values for a reliable check.',
        };
      }

      if (uniqueCount <= 5 || iqr === 0 || mad === 0) {
        return {
          column,
          lowerBound: Math.round((q1 - 1.5 * iqr) * 100) / 100,
          upperBound: Math.round((q3 + 1.5 * iqr) * 100) / 100,
          outliers: 0,
          outlierRate: 0,
          sampleSize: sorted.length,
          severity: 'none',
          note: 'Column behaves like a discrete or tightly clustered field, so outliers are not highlighted.',
        };
      }

      const lowerBound = q1 - 2.2 * iqr;
      const upperBound = q3 + 2.2 * iqr;
      const outliers = sorted.filter((value) => {
        const robustZ = (0.6745 * (value - median)) / mad;
        return (value < lowerBound || value > upperBound) && Math.abs(robustZ) >= 3.5;
      }).length;
      const outlierRate = sorted.length ? Math.round((outliers / sorted.length) * 1000) / 10 : 0;
      const severity: OutlierSummary['severity'] =
        outlierRate >= 5 ? 'high' : outlierRate >= 2 ? 'moderate' : outlierRate > 0 ? 'low' : 'none';
      const note =
        outliers === 0
          ? 'No strong outlier signal after robust screening.'
          : severity === 'high'
            ? 'A meaningful share of values are far from the typical range.'
            : severity === 'moderate'
              ? 'A small but real set of unusual values deserves review.'
              : 'Only a few extreme values stand out after robust screening.';

      return {
        column,
        lowerBound: Math.round(lowerBound * 100) / 100,
        upperBound: Math.round(upperBound * 100) / 100,
        outliers,
        outlierRate,
        sampleSize: sorted.length,
        severity,
        note,
      };
    })
    .sort((a, b) => b.outlierRate - a.outlierRate);
}

export function heatColor(value: number) {
  if (value >= 0) {
    const alpha = 0.15 + Math.abs(value) * 0.55;
    return `rgba(16, 185, 129, ${alpha})`;
  }
  const alpha = 0.15 + Math.abs(value) * 0.55;
  return `rgba(239, 68, 68, ${alpha})`;
}

export function formatMetric(value: number | undefined) {
  if (value === undefined) return 'N/A';
  return Number.isInteger(value)
    ? value.toLocaleString()
    : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function severityTone(severity: OutlierSummary['severity']) {
  if (severity === 'high') return 'text-red-700 dark:text-red-300';
  if (severity === 'moderate') return 'text-amber-700 dark:text-amber-300';
  if (severity === 'low') return 'text-emerald-700 dark:text-emerald-300';
  return 'text-muted-foreground';
}
