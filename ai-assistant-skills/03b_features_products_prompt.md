# Databricks Assistant Prompt: 03b Feature Derivation -- Products

## Prompt

Create a Databricks notebook called "03b -- Feature Derivation: Products" that derives product portfolio features from raw product holdings. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

Start with a markdown header containing a features table:

| Feature | Logic | Type | Purpose |
|---------|-------|------|---------|
| `nb_subscribed_products` | COUNT(*) where start_date <= ref_date | Aggregation | Total products ever subscribed |
| `nb_terminated_products` | COUNT(*) where end_date IS NOT NULL and end_date <= ref_date | Aggregation | Total products terminated |
| `nb_active_products` | subscribed - terminated | Aggregation | Currently active products |
| `product_diversity` | COUNT(DISTINCT product_type) where active | Aggregation | Distinct product categories held |
| `product_churn_rate` | terminated / subscribed | Derived Ratio | Attrition risk indicator |
| `net_product_growth` | active - terminated | Composite | Engagement trajectory |

Output table: `ai_03_products_features` -- one row per customer.

**Step 1: Setup**
- Set CATALOG and SCHEMA. Import `pyspark.sql.functions as F`.

**Step 2: Load Raw Product Holdings**
- Load `ai_01_raw_product_holdings`. Cast `start_date` and `end_date` to date type.
- Get `reference_date` from `raw_holdings.agg(F.max("start_date")).collect()[0][0]`.
- Print row count and reference date.

**Step 3: Derive Aggregation Features**
- Filter to products where `start_date <= reference_date` (`holdings_current`).
- Add `is_active` flag: 1 when `end_date` is null OR `end_date > reference_date`, else 0.
- Group by `customer_id` and aggregate:
  - `nb_subscribed_products`: `F.count("*")`
  - `nb_terminated_products`: `F.sum(F.when(F.col("is_active") == 0, 1).otherwise(0))`
  - `nb_active_products`: `F.sum(F.col("is_active"))`
  - `product_diversity`: `F.countDistinct(F.when(F.col("is_active") == 1, F.col("product_type")))`

**Step 4: Derived Ratio -- Product Churn Rate**
- `product_churn_rate = nb_terminated_products / nb_subscribed_products` (use 0 when subscribed is 0).

**Step 5: Composite -- Net Product Growth**
- `net_product_growth = nb_active_products - nb_terminated_products`

**Step 6: Summary and Save**
- Print row count and column count. Display first 5 rows.
- Write to `{CATALOG}.{SCHEMA}.ai_03_products_features` with overwrite mode.
- Read back and print saved row count.
- Add markdown documenting output columns.

## Expected Output

A notebook that produces `ai_03_products_features` -- one row per customer with: customer_id, nb_subscribed_products, nb_terminated_products, nb_active_products, product_diversity, product_churn_rate, net_product_growth.
