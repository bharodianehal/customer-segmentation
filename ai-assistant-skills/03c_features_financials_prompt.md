# Databricks Assistant Prompt: 03c Feature Derivation -- Financials

## Prompt

Create a Databricks notebook called "03c -- Feature Derivation: Financials" that aggregates balance and revenue data into customer-level financial features. Use `CATALOG = "customer_segmentation"`, `SCHEMA = "ai_banking"`, and `LOOKBACK_MONTHS = 12`.

Start with a markdown header containing a features table:

| Feature | Logic | Type | Purpose |
|---------|-------|------|---------|
| `total_balance` | SUM(balance) for latest snapshot | Aggregation | Customer's total holdings |
| `total_revenue` | SUM(revenue) over 12-month window | Aggregation | Revenue contribution |
| `total_balance_log` | `log1p(abs(total_balance))` | Log Transform | Reduce right-skew |
| `total_revenue_log` | `log1p(abs(total_revenue))` | Log Transform | Reduce right-skew |
| `{product_type}_balance_avg/last` | Per-product-type balance breakdown | Aggregation | Granularity for cross-sell |
| `{product_type}_revenue_avg/sum` | Per-product-type revenue breakdown | Aggregation | Revenue by product line |

Output table: `ai_03_financials_features` -- one row per customer.

**Step 1: Setup**
- Set CATALOG, SCHEMA, LOOKBACK_MONTHS. Import `timedelta` from datetime and `pyspark.sql.functions as F`.

**Step 2: Load Raw Data**
- Load `ai_01_raw_balances` (cast date to date type).
- Load `ai_01_raw_revenues` (cast date to date type).
- Load `ai_01_raw_product_holdings` (cast start_date to date type).
- Get `reference_date` from max `start_date` of product holdings.
- Print row counts and reference date.

**Step 3: Build Product Reference Lookup**
- From `ai_01_raw_product_holdings`, group by `product_id` and get last `product` and `product_type` values.

**Step 4: Aggregate Balances and Revenue**
- Compute `window_start = reference_date - timedelta(days=LOOKBACK_MONTHS * 30)`.
- Filter both balances and revenues to the window `[window_start, reference_date]`.
- Join both with the product reference on `product_id` to get `product_type`. Use `F.regexp_replace(product_type, " ", "_")` to normalize product type names.
- **Total balance**: Get `latest_balance_date` from max date in windowed balances. Filter to that date, group by `customer_id`, `SUM(balance)` as `total_balance`.
- **Total revenue**: Group windowed revenues by `customer_id`, `SUM(revenue)` as `total_revenue`.

**Step 5: Per-Product-Type Breakdown (Pivot)**
- Balance by type: group by `(customer_id, product_type)`, compute `avg(balance)` and `last(balance)`. Then pivot on `product_type` with `first("balance_avg")` and `first("balance_last")`. Fill nulls with 0.
- Revenue by type: group by `(customer_id, product_type)`, compute `avg(revenue)` and `sum(revenue)`. Then pivot on `product_type` with `first("revenue_avg")` and `first("revenue_sum")`. Fill nulls with 0.

**Step 6: Assemble Financial Features**
- Join `total_balance`, `total_revenue`, `balance_pivot`, and `revenue_pivot` on `customer_id` using outer/left joins. Fill nulls with 0.

**Step 7: Log Transforms**
- `total_balance_log = F.log1p(F.abs(F.col("total_balance")))`
- `total_revenue_log = F.log1p(F.abs(F.col("total_revenue")))`

**Step 8: Summary and Save**
- Print row count and column count. Display first 5 rows.
- Write to `{CATALOG}.{SCHEMA}.ai_03_financials_features` with overwrite mode.
- Read back and print saved row count.

## Expected Output

A notebook that produces `ai_03_financials_features` -- one row per customer with: customer_id, total_balance, total_revenue, total_balance_log, total_revenue_log, plus per-product-type pivot columns for balance_avg, balance_last, revenue_avg, revenue_sum.
