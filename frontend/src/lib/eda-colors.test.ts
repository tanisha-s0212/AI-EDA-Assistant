import { describe, expect, it } from 'vitest';
import { heatColor } from './eda';

describe('correlation heatmap colors', () => {
  it('uses a neutral color for zero and distinct colors for each sign', () => {
    expect(heatColor(0)).toBe('rgba(148, 163, 184, 0.15)');
    expect(heatColor(1)).toContain('16, 185, 129');
    expect(heatColor(-1)).toContain('239, 68, 68');
  });
});
