#!/usr/bin/env python3
"""Generate PrecisionForge Industries manufacturing demo datasets under samples/sales/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

# PrecisionForge Industries — fictional precision manufacturing OEM (demo only)
COMPANY = 'PrecisionForge Industries'

PLANTS: list[dict[str, str | float]] = [
    {'plant_code': 'PF-MW-01', 'plant_region': 'Midwest Plant', 'country': 'USA', 'scale': 1.12},
    {'plant_code': 'PF-SE-02', 'plant_region': 'Southeast Plant', 'country': 'USA', 'scale': 1.05},
    {'plant_code': 'PF-WC-03', 'plant_region': 'West Coast Plant', 'country': 'USA', 'scale': 1.18},
    {'plant_code': 'PF-MX-04', 'plant_region': 'Monterrey Plant', 'country': 'Mexico', 'scale': 0.94},
]

PRODUCT_LINES: list[dict[str, str | float]] = [
    {
        'product_line': 'Industrial Valves',
        'product_family': 'Fluid Power',
        'base_revenue_usd': 42_000,
        'cogs_ratio': 0.58,
        'asp_usd': 285.0,
        'units_per_order': 18,
    },
    {
        'product_line': 'CNC Components',
        'product_family': 'Precision Machining',
        'base_revenue_usd': 38_500,
        'cogs_ratio': 0.61,
        'asp_usd': 124.0,
        'units_per_order': 42,
    },
    {
        'product_line': 'Hydraulic Systems',
        'product_family': 'Fluid Power',
        'base_revenue_usd': 51_000,
        'cogs_ratio': 0.59,
        'asp_usd': 412.0,
        'units_per_order': 11,
    },
    {
        'product_line': 'Precision Bearings',
        'product_family': 'Precision Components',
        'base_revenue_usd': 29_500,
        'cogs_ratio': 0.64,
        'asp_usd': 68.0,
        'units_per_order': 55,
    },
    {
        'product_line': 'Automation Controls',
        'product_family': 'Industrial Automation',
        'base_revenue_usd': 47_500,
        'cogs_ratio': 0.55,
        'asp_usd': 198.0,
        'units_per_order': 24,
    },
]

CUSTOMER_SEGMENTS = ['OEM', 'Industrial Distributor', 'MRO Services', 'Export Partner']

# Stable segment mix per plant-product pair (realistic B2B manufacturing)
SEGMENT_BY_COMBO: dict[tuple[str, str], str] = {
    ('PF-MW-01', 'Industrial Valves'): 'OEM',
    ('PF-MW-01', 'CNC Components'): 'Industrial Distributor',
    ('PF-MW-01', 'Hydraulic Systems'): 'OEM',
    ('PF-MW-01', 'Precision Bearings'): 'MRO Services',
    ('PF-MW-01', 'Automation Controls'): 'OEM',
    ('PF-SE-02', 'Industrial Valves'): 'Industrial Distributor',
    ('PF-SE-02', 'CNC Components'): 'OEM',
    ('PF-SE-02', 'Hydraulic Systems'): 'Export Partner',
    ('PF-SE-02', 'Precision Bearings'): 'Industrial Distributor',
    ('PF-SE-02', 'Automation Controls'): 'MRO Services',
    ('PF-WC-03', 'Industrial Valves'): 'Export Partner',
    ('PF-WC-03', 'CNC Components'): 'OEM',
    ('PF-WC-03', 'Hydraulic Systems'): 'OEM',
    ('PF-WC-03', 'Precision Bearings'): 'OEM',
    ('PF-WC-03', 'Automation Controls'): 'Industrial Distributor',
    ('PF-MX-04', 'Industrial Valves'): 'Export Partner',
    ('PF-MX-04', 'CNC Components'): 'Export Partner',
    ('PF-MX-04', 'Hydraulic Systems'): 'Industrial Distributor',
    ('PF-MX-04', 'Precision Bearings'): 'MRO Services',
    ('PF-MX-04', 'Automation Controls'): 'Export Partner',
}


def _seasonality(month: int) -> float:
    """Manufacturing calendar: Q4 capital spend, summer maintenance dip."""
    seasonal = {
        1: 0.96, 2: 0.98, 3: 1.02, 4: 1.04, 5: 1.03, 6: 0.97,
        7: 0.95, 8: 0.96, 9: 1.05, 10: 1.08, 11: 1.10, 12: 1.07,
    }
    return seasonal[month]


def _fiscal_quarter(month: int) -> str:
    return f'FY-Q{(month - 1) // 3 + 1}'


def build_monthly() -> pd.DataFrame:
    """Plant × product_line monthly operational sales (72 months × 20 SKUs = 1,440 rows)."""
    months = pd.date_range('2019-01-01', periods=72, freq='MS')
    rows: list[dict] = []
    rng = np.random.default_rng(42)

    for month_idx, month in enumerate(months):
        fiscal_year = month.year
        growth = 1.0 + (0.028 * month_idx / 12)  # ~2.8% CAGR
        season = _seasonality(month.month)
        covid_dip = 0.88 if month.year == 2020 and month.month in (4, 5) else 1.0
        recovery = 1.04 if month.year == 2021 and month.month <= 3 else 1.0

        for plant in PLANTS:
            for product in PRODUCT_LINES:
                combo_key = (str(plant['plant_code']), str(product['product_line']))
                customer_segment = SEGMENT_BY_COMBO.get(combo_key, CUSTOMER_SEGMENTS[month_idx % 4])

                base = float(product['base_revenue_usd']) * float(plant['scale'])
                noise = float(rng.uniform(0.94, 1.06))
                net_sales = round(base * growth * season * covid_dip * recovery * noise, 2)

                cogs_ratio = float(product['cogs_ratio']) + float(rng.uniform(-0.015, 0.015))
                cogs = round(net_sales * cogs_ratio, 2)
                gross_profit = round(net_sales - cogs, 2)
                gross_margin_pct = round(100.0 * gross_profit / net_sales, 2) if net_sales else 0.0

                asp = round(float(product['asp_usd']) * float(rng.uniform(0.98, 1.02)), 2)
                units_shipped = max(1, int(round(net_sales / asp)))
                order_count = max(1, int(round(units_shipped / float(product['units_per_order']))))
                backlog_usd = round(net_sales * float(rng.uniform(0.18, 0.32)), 2)
                on_time_delivery_pct = round(float(rng.uniform(91.5, 98.8)), 1)
                scrap_rate_pct = round(float(rng.uniform(0.35, 1.85)), 2)
                capacity_utilization_pct = round(float(rng.uniform(72.0, 94.0)), 1)

                rows.append({
                    'year_month': month.strftime('%Y-%m-%d'),
                    'fiscal_year': fiscal_year,
                    'fiscal_quarter': _fiscal_quarter(month.month),
                    'plant_code': plant['plant_code'],
                    'plant_region': plant['plant_region'],
                    'plant_country': plant['country'],
                    'product_line': product['product_line'],
                    'product_family': product['product_family'],
                    'customer_segment': customer_segment,
                    'net_sales': net_sales,
                    'cogs': cogs,
                    'gross_profit': gross_profit,
                    'gross_margin_pct': gross_margin_pct,
                    'units_shipped': units_shipped,
                    'avg_selling_price_usd': asp,
                    'order_count': order_count,
                    'backlog_usd': backlog_usd,
                    'on_time_delivery_pct': on_time_delivery_pct,
                    'scrap_rate_pct': scrap_rate_pct,
                    'capacity_utilization_pct': capacity_utilization_pct,
                })

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(['year_month', 'plant_code', 'product_line']).reset_index(drop=True)
    return frame


def build_invoices() -> pd.DataFrame:
    """Daily shipment ledger aligned with monthly manufacturing domain."""
    dates = pd.date_range('2023-01-01', periods=540, freq='D')
    rng = np.random.default_rng(7)
    rows: list[dict] = []

    for idx, day in enumerate(dates):
        plant = PLANTS[int(rng.integers(0, len(PLANTS)))]
        product = PRODUCT_LINES[int(rng.integers(0, len(PRODUCT_LINES)))]
        combo_key = (str(plant['plant_code']), str(product['product_line']))
        customer_segment = SEGMENT_BY_COMBO.get(combo_key, 'OEM')

        weekend_factor = 0.68 if day.dayofweek >= 5 else 1.0
        base = float(product['base_revenue_usd']) * float(plant['scale']) / 22.0
        net_sales = round(base * weekend_factor * float(rng.uniform(0.85, 1.15)), 2)
        cogs = round(net_sales * float(product['cogs_ratio']) * float(rng.uniform(0.98, 1.02)), 2)
        asp = round(float(product['asp_usd']) * float(rng.uniform(0.99, 1.01)), 2)
        units_shipped = max(1, int(round(net_sales / asp)))

        rows.append({
            'invoice_date': day.strftime('%Y-%m-%d'),
            'shipment_id': f'SHP-{idx:06d}',
            'sales_order_id': f'SO-2023-{idx:05d}',
            'plant_code': plant['plant_code'],
            'plant_region': plant['plant_region'],
            'product_line': product['product_line'],
            'product_family': product['product_family'],
            'customer_segment': customer_segment,
            'net_sales': net_sales,
            'cogs': cogs,
            'units_shipped': units_shipped,
            'freight_cost_usd': round(float(rng.uniform(120, 480)), 2),
            'incoterms': rng.choice(['FOB', 'DDP', 'EXW']),
        })

    return pd.DataFrame(rows)


def main() -> None:
    monthly = build_monthly()
    invoices = build_invoices()
    monthly.to_csv(OUT / 'sales_monthly.csv', index=False)
    monthly.to_csv(OUT / 'sales_monthly.tsv', sep='\t', index=False)
    monthly.to_parquet(OUT / 'sales_monthly.parquet', index=False)
    monthly.to_excel(OUT / 'sales_monthly.xlsx', index=False)
    invoices.to_csv(OUT / 'sales_invoices.csv', index=False)

    months = monthly['year_month'].nunique()
    print(f'Company: {COMPANY}')
    print(f'Monthly: {len(monthly)} rows ({months} months × {len(PLANTS)} plants × {len(PRODUCT_LINES)} product lines)')
    print(f'Columns: {len(monthly.columns)}')
    print(f'Invoices: {len(invoices)} daily shipment rows')
    print(f'Output: {OUT}')


if __name__ == '__main__':
    main()
