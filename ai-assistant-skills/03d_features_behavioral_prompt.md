# Databricks Assistant Prompt: 03d Feature Derivation -- Behavioral

## Prompt

Create a Databricks notebook called "03d -- Feature Derivation: Behavioral" that aggregates behavioral interaction data into customer-level features. Use `CATALOG = "customer_segmentation"`, `SCHEMA = "ai_banking"`, and `LOOKBACK_MONTHS = 12`.

Start with a markdown header listing the 7 raw behavioral metrics and their _avg/_last aggregations:

| Raw Column | `_avg` | `_last` |
|------------|--------|---------|
| `in_person_interaction` | Avg in-person visits | Most recent count |
| `atm_withdrawals` | Avg ATM frequency | Most recent frequency |
| `atm_withdrawals_sum` | Avg ATM amount | Most recent amount |
| `physical_services` | Avg physical usage | Most recent count |
| `physical_services_sum` | Avg physical amount | Most recent amount |
| `digital_services` | Avg digital usage | Most recent count |
| `fx_services` | Avg FX usage | Most recent count |

Plus derived features:

| Feature | Logic | Type | Purpose |
|---------|-------|------|---------|
| `digital_ratio` | digital / (digital + in_person) | Derived Ratio | Channel preference (0=branch, 1=digital) |
| `atm_intensity` | Passthrough of `atm_withdrawals_sum_avg` | Composite | Cash dependency indicator |

Output table: `ai_03_behavioral_features` -- one row per customer.

**Step 1: Setup**
- Set CATALOG, SCHEMA, LOOKBACK_MONTHS. Import `timedelta` from datetime and `pyspark.sql.functions as F`.

**Step 2: Load Raw Behavioral Data**
- Load `ai_01_raw_additional_info` and cast `date` to date type.
- Get `reference_date` from max `start_date` of `ai_01_raw_product_holdings` (cast to date).
- Print row count and reference date.

**Step 3: Aggregate Per Customer**
- Compute `window_start = reference_date - timedelta(days=LOOKBACK_MONTHS * 30)`.
- Filter to the lookback window.
- Define `behavior_cols` list: `["in_person_interaction", "atm_withdrawals", "atm_withdrawals_sum", "physical_services", "physical_services_sum", "digital_services", "fx_services"]`.
- Build aggregation expressions: for each column, compute `F.avg(c).alias(f"{c}_avg")` and `F.last(c).alias(f"{c}_last")`.
- Group by `customer_id` and apply all aggregation expressions.

**Step 4: Derived Ratio -- Digital Ratio**
- `digital_ratio = digital_services_avg / (digital_services_avg + in_person_interaction_avg)` when denominator > 0, else 0.0.
- Use `F.coalesce()` to handle nulls.

**Step 5: Composite -- ATM Intensity**
- `atm_intensity = F.coalesce(F.col("atm_withdrawals_sum_avg"), F.lit(0.0))`

**Step 6: Summary and Save**
- Print row count and column count. Display first 5 rows.
- Write to `{CATALOG}.{SCHEMA}.ai_03_behavioral_features` with overwrite mode.
- Read back and print saved row count.

## Expected Output

A notebook that produces `ai_03_behavioral_features` -- one row per customer with: customer_id, 14 aggregated columns (7 metrics x avg/last), digital_ratio, and atm_intensity.
