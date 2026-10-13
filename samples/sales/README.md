# Manufacturing sales golden samples (PrecisionForge Industries)

Professional demo datasets for the **Intelligent Data Assistant** — a fictional precision manufacturing OEM with multi-plant operations.

## Files

| File | Grain | Rows (approx.) | Use case |
|------|-------|----------------|----------|
| `sales_monthly.csv` / `.tsv` / `.xlsx` / `.parquet` | Monthly plant × product line | **1,440** (72 months) | **Primary demo** — TS Forecast, ML Forecast, Loss/Profit, Report |
| `sales_invoices.csv` | Daily shipment ledger | **540** | EDA, Cleaning, ML Assistant (daily grain) |

## Domain: PrecisionForge Industries

| Attribute | Values |
|-----------|--------|
| Plants | PF-MW-01 Midwest, PF-SE-02 Southeast, PF-WC-03 West Coast, PF-MX-04 Monterrey |
| Product families | Fluid Power, Precision Machining, Precision Components, Industrial Automation |
| Product lines | Industrial Valves, CNC Components, Hydraulic Systems, Precision Bearings, Automation Controls |
| Customer segments | OEM, Industrial Distributor, MRO Services, Export Partner |
| History span | Jan 2019 – Dec 2024 (72 monthly periods) |

## Key columns (`sales_monthly.csv`)

| Column | Description |
|--------|-------------|
| `year_month` | Reporting period (use as date column) |
| `net_sales` | Recognized revenue USD (use as forecast target) |
| `cogs` | Cost of goods sold |
| `gross_profit`, `gross_margin_pct` | Margin metrics |
| `units_shipped`, `avg_selling_price_usd`, `order_count` | Volume & pricing |
| `backlog_usd` | Open order backlog |
| `on_time_delivery_pct`, `scrap_rate_pct`, `capacity_utilization_pct` | Operations KPIs |
| `plant_code`, `plant_region`, `plant_country` | Manufacturing site |
| `product_line`, `product_family`, `customer_segment` | Commercial dimensions |
| `fiscal_year`, `fiscal_quarter` | Finance calendar |

## Currency display

All monetary columns in this demo (`net_sales`, `cogs`, `backlog_usd`, etc.) are **USD magnitudes**. In the app, use the **Currency: USD** toggle on forecast tabs (top-right of the step navigator). The backend never converts currency — only the display label and number grouping change.

## Recommended demo flow

1. Upload **`sales_monthly.csv`**
2. Auto-map: `year_month` → date, `net_sales` → target
3. Run **Cleaning** (sales preset)
4. **Time Series Forecast** — aggregates to 72 monthly periods (passes ≥24 gate)
5. **ML Assistant** — predict `net_sales` using `plant_region`, `product_line`, `customer_segment`, `units_shipped`, etc.

## Regenerate

```bash
python samples/sales/generate_samples.py
python backend/scripts/train_sales_demo_model.py
```

Customer production path remains train-on-upload; these files support demos and regression tests only.
