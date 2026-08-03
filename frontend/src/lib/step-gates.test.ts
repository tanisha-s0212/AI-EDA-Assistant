import { describe, expect, test } from 'vitest';
import { isStepTabEnabled, POST_CLEANING_TABS } from './step-gates';
import type { TabId } from '@/lib/store';

describe('isStepTabEnabled', () => {
  const rawData = [{ a: 1 }];

  test('allows early tabs with rawData only', () => {
    const ctx = { rawData, cleaningDone: false, modelTrained: false };
    for (const tab of ['upload', 'understanding', 'eda', 'cleaning'] as TabId[]) {
      expect(isStepTabEnabled(tab, ctx)).toBe(true);
    }
  });

  test('blocks post-cleaning tabs until cleaningDone', () => {
    const blocked = { rawData, cleaningDone: false, modelTrained: false };
    const open = { rawData, cleaningDone: true, modelTrained: false };
    for (const tab of POST_CLEANING_TABS) {
      if (tab === 'prediction') continue;
      expect(isStepTabEnabled(tab, blocked)).toBe(false);
      expect(isStepTabEnabled(tab, open)).toBe(true);
    }
  });

  test('prediction still requires modelTrained after cleaning', () => {
    expect(
      isStepTabEnabled('prediction', { rawData, cleaningDone: true, modelTrained: false }),
    ).toBe(false);
    expect(
      isStepTabEnabled('prediction', { rawData, cleaningDone: true, modelTrained: true }),
    ).toBe(true);
  });
});
