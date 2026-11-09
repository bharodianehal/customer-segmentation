# Databricks Assistant Prompt: 04 Feature Join and Feature Store

## Prompt

Create a Databricks notebook called "04 -- Feature Join & Feature Store Registration" that joins the four domain feature tables and registers the result in Unity Catalog Feature Store. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

Start with a markdown header showing the pipeline:
```
ai_03_customers_features  (demographics + income_log, income_quintile, gender encoding) --+
ai_03_products_features   (portfolio + product_churn_rate, net_product_growth) -----------+
ai_03_financials_features (aggregates + total_balance_log, total_revenue_log) ------------+-> [04] Join
ai_03_behavioral_features (metrics + digital_ratio, atm_intensity) -----------------------+
```

**Step 1: Install and Setup**
- First cell: `%pip install databricks-feature-engineering --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Re-set CATALOG and SCHEMA (because of restartPython). Import `FeatureEngineeringClient` from `databricks.feature_engineering`, `pyspark.sql.functions as F`, and `Window`. Initialize `fe = FeatureEngineeringClient()`.

**Step 2: Join All Domain Feature Tables (4.1)**
- Load all four tables: `ai_03_customers_features`, `ai_03_products_features`, `ai_03_financials_features`, `ai_03_behavioral_features`.
- Print row and column counts for each.
- Join all on `customer_id` using left joins: `customers.join(products, "customer_id", "left").join(financials, "customer_id", "left").join(behavioral, "customer_id", "left").fillna(0)`.
- Print joined table dimensions. Display first 5 rows.

**Step 3: Cross-Domain Feature (4.2)**
- Add `revenue_per_product = total_revenue / nb_subscribed_products` (use 0 when subscribed is 0).

**Step 4: Feature Summary and Quality Check (4.3)**
- Define a list of engineered features to check: `["income_log", "total_balance_log", "total_revenue_log", "digital_ratio", "revenue_per_product", "product_churn_rate", "net_product_growth", "atm_intensity", "income_quintile"]`.
- Dynamically add any `gender_*` columns (excluding raw `gender`).
- For each feature, compute min, max, mean, and null count.
- Print a formatted stats table.

**Step 5: Register Feature Tables in Feature Store (4.5)**
- Register `ai_04_customer_features_reference` with primary key `["customer_id"]` using `fe.create_table()`. Include a description about customer segmentation features. If the table already exists, use `fe.write_table()` with `mode="overwrite"`.

**Step 6: Add Column Descriptions (4.6)**
- Define a `COLUMN_DESCRIPTIONS` dictionary mapping column names to descriptions:
  - `customer_id` -> "Unique customer identifier (primary key)"
  - `reference_date` -> "Monthly snapshot date"
  - `gender` -> "Customer gender (raw)"
  - `region` -> "Geographic region code"
  - `income` -> "Annual income (raw)"
  - `tier` -> "Customer tier (Gold, Silver, Bronze, etc.)"
  - `customer_age` -> "Customer age in years"
  - `tenure_months` -> "Account tenure in months"
  - `age_bracket` -> "Age group: Young (<30), Middle (30-49), Senior (50-64), Elderly (65+)"
  - `income_log` -> "Log-transformed income: log1p(|income|)"
  - `total_balance_log` -> "Log-transformed total balance: log1p(|total_balance|)"
  - `total_revenue_log` -> "Log-transformed total revenue: log1p(|total_revenue|)"
  - `gender_f` -> "One-hot: gender is Female (1/0)"
  - `gender_m` -> "One-hot: gender is Male (1/0)"
  - `total_balance` -> "Sum of all balance amounts (latest snapshot)"
  - `total_revenue` -> "Sum of all revenue amounts (12-month window)"
  - `nb_subscribed_products` -> "Total products ever subscribed"
  - `nb_terminated_products` -> "Total products terminated"
  - `nb_active_products` -> "Currently active products"
  - `product_diversity` -> "COUNT(DISTINCT product_type) where active"
  - `digital_ratio` -> "Digital / (digital + in-person) -- channel preference"
  - `revenue_per_product` -> "Total revenue / subscribed products"
  - `product_churn_rate` -> "Terminated / subscribed products -- attrition risk"
  - `net_product_growth` -> "Active - terminated products"
  - `atm_intensity` -> "ATM withdrawal volume (from atm_withdrawals_sum_avg)"
  - `income_quintile` -> "Income quintile bucket (1=lowest, 5=highest)"
- Apply using `ALTER TABLE ... ALTER COLUMN ... COMMENT` for each column that exists.

**Step 7: Summary**
- Print a final summary showing the reference table name, row count, and column count.

## Expected Output

A notebook that joins the 4 domain feature tables on customer_id, adds revenue_per_product, runs a feature quality check, registers `ai_04_customer_features_reference` in Feature Store with primary key customer_id, and adds column descriptions via ALTER TABLE.
