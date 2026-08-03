import type { TabId } from '@/lib/store';

/** Tabs that require a successful Cleaning run (`cleaningDone`). */
export const POST_CLEANING_TABS: TabId[] = [
  'forecast_ts',
  'forecast_ml',
  'loss_forecast',
  'profit_forecast',
  'ml',
  'prediction',
  'report',
];

export function isStepTabEnabled(
  tabId: TabId,
  {
    rawData,
    cleaningDone,
    modelTrained,
  }: {
    rawData: unknown;
    cleaningDone: boolean;
    modelTrained: boolean;
  },
): boolean {
  if (tabId === 'upload') return true;
  if (!rawData) return false;
  if (POST_CLEANING_TABS.includes(tabId) && !cleaningDone) return false;
  if (tabId === 'prediction' && !modelTrained) return false;
  return true;
}
