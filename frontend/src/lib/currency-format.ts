export type CurrencyCode = 'USD' | 'INR';

type CurrencyConfig = {
  locale: string;
  currency: CurrencyCode;
  symbol: string;
};

const CURRENCY_CONFIG: Record<CurrencyCode, CurrencyConfig> = {
  USD: { locale: 'en-US', currency: 'USD', symbol: '$' },
  INR: { locale: 'en-IN', currency: 'INR', symbol: '₹' },
};

function resolveCurrency(code: CurrencyCode): CurrencyCode {
  return CURRENCY_CONFIG[code] ? code : 'USD';
}

function toNumeric(value: number | string): number {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : 0;
}

export function inferCurrencyFromColumns(columns: Array<{ name: string }>): CurrencyCode | null {
  const hasUsdHint = columns.some((column) => /(?:^|[_-])usd(?:$|[_-])|usd/i.test(column.name));
  return hasUsdHint ? 'USD' : null;
}

export function formatCurrency(value: number | string, code: CurrencyCode = 'USD'): string {
  const currencyCode = resolveCurrency(code);
  const config = CURRENCY_CONFIG[currencyCode];
  return new Intl.NumberFormat(config.locale, {
    style: 'currency',
    currency: config.currency,
    maximumFractionDigits: 0,
  }).format(toNumeric(value));
}

export function formatNumber(value: number | string | null | undefined, code: CurrencyCode = 'USD', digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  const currencyCode = resolveCurrency(code);
  return new Intl.NumberFormat(CURRENCY_CONFIG[currencyCode].locale, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(toNumeric(value));
}

export function formatPercent(value: number | string, code: CurrencyCode = 'USD', digits = 1): string {
  const currencyCode = resolveCurrency(code);
  return new Intl.NumberFormat(CURRENCY_CONFIG[currencyCode].locale, {
    style: 'percent',
    maximumFractionDigits: digits,
  }).format(toNumeric(value));
}

export function formatPercentValue(value: number | string, code: CurrencyCode = 'USD', digits = 1): string {
  const currencyCode = resolveCurrency(code);
  return `${new Intl.NumberFormat(CURRENCY_CONFIG[currencyCode].locale, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(toNumeric(value))}%`;
}

export function formatCompactCurrency(value: number | string, code: CurrencyCode = 'USD'): string {
  const currencyCode = resolveCurrency(code);
  const config = CURRENCY_CONFIG[currencyCode];
  const numericValue = toNumeric(value);
  if (!Number.isFinite(numericValue)) return `${config.symbol}0`;

  const sign = numericValue < 0 ? '-' : '';
  const absoluteValue = Math.abs(numericValue);
  const compactNumber = new Intl.NumberFormat(config.locale, { maximumFractionDigits: 1 });

  if (currencyCode === 'INR') {
    if (absoluteValue >= 1_00_00_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_00_00_000)} Cr`;
    if (absoluteValue >= 1_00_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_00_000)} L`;
    if (absoluteValue >= 1_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_000)} K`;
    return `${sign}${formatCurrency(absoluteValue, currencyCode)}`;
  }

  if (absoluteValue >= 1_000_000_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_000_000_000)}B`;
  if (absoluteValue >= 1_000_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_000_000)}M`;
  if (absoluteValue >= 1_000) return `${sign}${config.symbol}${compactNumber.format(absoluteValue / 1_000)}K`;
  return `${sign}${formatCurrency(absoluteValue, currencyCode)}`;
}

export function formatAxisCurrency(value: number | string, code: CurrencyCode = 'USD'): string {
  const currencyCode = resolveCurrency(code);
  const config = CURRENCY_CONFIG[currencyCode];
  const numericValue = toNumeric(value);
  if (!Number.isFinite(numericValue)) return `${config.symbol}0`;

  const sign = numericValue < 0 ? '-' : '';
  const absoluteValue = Math.abs(numericValue);

  if (currencyCode === 'INR') {
    if (absoluteValue >= 10_000_000) return `${sign}${config.symbol}${(absoluteValue / 10_000_000).toFixed(1)}Cr`;
    if (absoluteValue >= 100_000) return `${sign}${config.symbol}${(absoluteValue / 100_000).toFixed(1)}L`;
    if (absoluteValue >= 1_000) return `${sign}${config.symbol}${(absoluteValue / 1_000).toFixed(1)}K`;
    return `${sign}${config.symbol}${absoluteValue}`;
  }

  if (absoluteValue >= 1_000_000_000) return `${sign}${config.symbol}${(absoluteValue / 1_000_000_000).toFixed(1)}B`;
  if (absoluteValue >= 1_000_000) return `${sign}${config.symbol}${(absoluteValue / 1_000_000).toFixed(1)}M`;
  if (absoluteValue >= 1_000) return `${sign}${config.symbol}${(absoluteValue / 1_000).toFixed(1)}K`;
  return `${sign}${config.symbol}${absoluteValue}`;
}

export function formatCurrencyTooltip(value: number | string, code: CurrencyCode = 'USD'): string {
  return formatCurrency(value, code);
}
