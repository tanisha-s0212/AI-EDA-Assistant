import { describe, expect, it } from 'vitest';
import { formatEdaFieldLabel } from '@/lib/eda';

describe('formatEdaFieldLabel', () => {
  it('formats snake case and common abbreviations without changing the source field', () => {
    expect(formatEdaFieldLabel('avg_order_value')).toBe('Average Order Value');
    expect(formatEdaFieldLabel('fiscal_year')).toBe('Fiscal Year');
  });

  it('handles mixed case, separators, and explicit unit abbreviations', () => {
    expect(formatEdaFieldLabel('grossMargin_USD')).toBe('Gross Margin USD');
    expect(formatEdaFieldLabel('Customer-Segment Name')).toBe('Customer Segment Name');
  });

  it('uses a safe fallback for missing field names', () => {
    expect(formatEdaFieldLabel(null)).toBe('Column');
    expect(formatEdaFieldLabel('undefined')).toBe('Column');
  });
});
