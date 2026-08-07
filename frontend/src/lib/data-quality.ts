import type { ColumnInfo } from '@/lib/store';

/** Cell populate rate from column profile stats (preview- or full-scoped). */
export function computeProfileCompleteness(
  columns: Pick<ColumnInfo, 'nonNull' | 'nullCount'>[],
): number {
  const profileCells = columns.reduce((sum, col) => sum + col.nonNull + col.nullCount, 0);
  if (profileCells <= 0) return 0;
  const populated = columns.reduce((sum, col) => sum + col.nonNull, 0);
  return Math.round((populated / profileCells) * 1000) / 10;
}

/** Per-column missing share using the same profile denominator as completeness. */
export function computeColumnMissingPercent(
  column: Pick<ColumnInfo, 'nonNull' | 'nullCount'>,
): number {
  const profileRows = column.nonNull + column.nullCount;
  if (profileRows <= 0) return 0;
  return Math.round((column.nullCount / profileRows) * 100);
}
