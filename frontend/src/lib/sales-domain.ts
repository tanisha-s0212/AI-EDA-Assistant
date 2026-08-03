/** Sales-domain column heuristics shared by forecast tabs (no LLM). */

const TARGET_EXCLUSION = /(^id$|_id$|uuid|index|code|^sku$)/i;
const REVENUE_EXCLUDE = /loss|lost|missed|return|refund|cost|cogs|expense|tax|qty|quantity|unit_price|^price$/i;

const DATE_TOKEN_SCORES: Array<[RegExp | string, number]> = [
  ['invoice_date', 100],
  ['order_date', 95],
  ['bill_date', 90],
  ['doc_date', 88],
  ['transaction_date', 85],
  ['sale_date', 85],
  ['year_month', 80],
  ['period', 70],
  ['month', 60],
  ['week', 55],
  ['date', 50],
  ['time', 20],
  ['start', 15],
];

const REVENUE_TOKEN_SCORES: Array<[string, number]> = [
  ['sale_free', 100],
  ['net_sales', 98],
  ['sale_value', 95],
  ['total_value_sale', 92],
  ['gmv', 90],
  ['turnover', 88],
  ['revenue', 85],
  ['sales', 80],
  ['total_value', 70],
  ['amount', 40],
  ['total', 25],
];

function tokenScore(name: string, scores: Array<[string, number]>) {
  const lowered = name.toLowerCase();
  let best = 0;
  for (const [token, score] of scores) {
    if (lowered.includes(token)) best = Math.max(best, score);
  }
  return best;
}

export function scoreSalesDateColumn(name: string, role?: string) {
  const score = tokenScore(name, DATE_TOKEN_SCORES as Array<[string, number]>);
  if (score > 0) return score;
  if (role === 'datetime' || role === 'date') return 45;
  return 0;
}

export function scoreSalesRevenueColumn(name: string) {
  const lowered = name.toLowerCase();
  if (TARGET_EXCLUSION.test(lowered)) return 0;
  if (REVENUE_EXCLUDE.test(lowered) && !/(sale_free|sale_value|net_sales|revenue|sales)/i.test(lowered)) return 0;
  return tokenScore(name, REVENUE_TOKEN_SCORES);
}

export function pickPreferredSalesDateColumn<T extends { name: string; role?: string }>(columns: T[]) {
  const scored = columns
    .map((column) => ({ name: column.name, score: scoreSalesDateColumn(column.name, column.role) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
  return scored[0]?.name ?? '';
}

export function pickSmartSalesTargetColumn<T extends { name: string; role?: string }>(
  columns: T[],
  data: Array<Record<string, unknown>>,
) {
  const scored = columns
    .filter((column) => column.role === 'numeric' && !TARGET_EXCLUSION.test(column.name))
    .map((column) => {
      const values = data.map((row) => Number(row[column.name])).filter(Number.isFinite);
      const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
      const variance = values.length ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length : 0;
      const nameScore = scoreSalesRevenueColumn(column.name);
      return { name: column.name, nameScore, variance };
    })
    .filter((item) => item.nameScore > 0 || item.variance > 0)
    .sort((a, b) => b.nameScore - a.nameScore || b.variance - a.variance || a.name.localeCompare(b.name));
  return scored[0]?.name ?? '';
}

/** Keep in sync with backend sales_domain.SALES_PRESET_REVENUE_SCORE_THRESHOLD */
export const SALES_PRESET_REVENUE_SCORE_THRESHOLD = 70;

/** True when a column name strongly indicates sales/revenue (score >= 70). */
export function shouldEnableSalesPreset(
  columns: Array<{ name: string; role?: string }>,
): boolean {
  return columns.some(
    (column) => scoreSalesRevenueColumn(column.name) >= SALES_PRESET_REVENUE_SCORE_THRESHOLD,
  );
}
