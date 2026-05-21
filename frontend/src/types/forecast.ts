export type RiskLabel = 'Low' | 'Medium' | 'High';
export type ProfitScenario = 'optimistic' | 'baseline' | 'pessimistic';

export interface LossForecastResult {
  id: string;
  session_id: string;
  period: string;
  revenue_loss: number;
  operational_loss: number;
  inventory_loss: number;
  discount_loss: number;
  total_loss: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  loss_risk_score: number;
  risk_label: RiskLabel;
  segment?: string | null;
  created_at: string;
}

export interface ProfitForecastResult {
  id: string;
  session_id: string;
  period: string;
  forecasted_revenue: number;
  forecasted_cogs: number;
  gross_profit: number;
  operating_expenses: number;
  total_losses: number;
  net_profit: number;
  gross_margin_pct: number;
  net_margin_pct: number;
  scenario: ProfitScenario;
  created_at: string;
}

export interface SegmentBreakdown {
  segment: string;
  segment_type: 'category' | 'region' | 'portfolio' | string;
  total_loss: number;
  risk_score: number;
  risk_label: RiskLabel;
}

export interface LossForecastSummary {
  total_loss: number;
  highest_risk_period: string | null;
  average_risk_score: number;
  top_loss_driver: string;
  driver_weights?: Record<string, number>;
}

export interface ScenarioComparison {
  optimistic: ProfitForecastResult[];
  baseline: ProfitForecastResult[];
  pessimistic: ProfitForecastResult[];
}

export interface ForecastAssumptionAudit {
  assumptions_audit?: string[];
}

export interface ReportConfig {
  includeLoss: boolean;
  includeProfit: boolean;
  scenario: ProfitScenario;
}
