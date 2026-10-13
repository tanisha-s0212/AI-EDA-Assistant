'use client';

import { useMemo } from 'react';
import {
  type CurrencyCode,
  formatAxisCurrency,
  formatCompactCurrency,
  formatCurrency,
  formatCurrencyTooltip,
  formatNumber,
  formatPercent,
  formatPercentValue,
} from '@/lib/currency-format';
import { useAppStore } from '@/lib/store';

export function useCurrencyFormat() {
  const displayCurrency = useAppStore((state) => state.displayCurrency);

  return useMemo(() => {
    const code: CurrencyCode = displayCurrency;
    return {
      currencyCode: code,
      formatCurrency: (value: number | string) => formatCurrency(value, code),
      formatCompactCurrency: (value: number | string) => formatCompactCurrency(value, code),
      formatAxisCurrency: (value: number | string) => formatAxisCurrency(value, code),
      formatCurrencyTooltip: (value: number | string) => formatCurrencyTooltip(value, code),
      formatNumber: (value: number | string | null | undefined, digits = 2) => formatNumber(value, code, digits),
      formatPercent: (value: number | string, digits = 1) => formatPercent(value, code, digits),
      formatPercentValue: (value: number | string, digits = 1) => formatPercentValue(value, code, digits),
    };
  }, [displayCurrency]);
}
