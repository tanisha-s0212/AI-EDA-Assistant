# Sales golden samples

| File | Grain | Key columns |
|------|-------|-------------|
| sales_monthly.csv/.tsv/.xlsx/.parquet | Monthly | year_month, total_total_value_sale_free, cogs, region, category |
| sales_invoices.csv | Daily invoice | invoice_date, net_sales, cogs, region, category |

Use monthly files for Time Series / ML Forecast (needs >=24 periods after aggregation).
Customer production path remains train-on-upload; these files support demos and regression tests.
