# Databricks notebook source
# MAGIC %md
# MAGIC # 03c — Feature Derivation: Financials
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook aggregates balance and revenue data into customer-level financial features
# MAGIC and applies log transforms to skewed variables.
# MAGIC
# MAGIC ### Features Derived
# MAGIC | Feature | Logic | Type | Purpose |
# MAGIC |---------|-------|------|---------|
# MAGIC | `total_balance` | SUM(balance) for latest snapshot | Aggregation | Customer's total holdings |
# MAGIC | `total_revenue` | SUM(revenue) over 12-month window | Aggregation | Revenue contribution |
# MAGIC | `total_balance_log` | `log1p(abs(total_balance))` | Log Transform | Reduce right-skew |
# MAGIC | `total_revenue_log` | `log1p(abs(total_revenue))` | Log Transform | Reduce right-skew |
# MAGIC | `{product_type}_balance_avg/last` | Per-product-type balance breakdown | Aggregation | Granularity for cross-sell |
# MAGIC | `{product_type}_revenue_avg/sum` | Per-product-type revenue breakdown | Aggregation | Revenue by product line |
# MAGIC
# MAGIC ### Output Table
# MAGIC `03_financials_features` — one row per customer with aggregated balance/revenue columns

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "banking"
LOOKBACK_MONTHS = 12

# COMMAND ----------

from datetime import timedelta
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Data

# COMMAND ----------

raw_balances = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_balances")
    .withColumn("date", F.to_date(F.col("date")))
)

raw_revenues = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_revenues")
    .withColumn("date", F.to_date(F.col("date")))
)

raw_holdings = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
)

reference_date = raw_holdings.agg(F.max("start_date")).collect()[0][0]

print(f"Balances:  {raw_balances.count():,} rows")
print(f"Revenues:  {raw_revenues.count():,} rows")
print(f"Reference date: {reference_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build Product Reference Lookup

# COMMAND ----------

product_ref = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_product_holdings")
    .groupBy("product_id")
    .agg(
        F.last("product").alias("product"),
        F.last("product_type").alias("product_type"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Aggregate Balances & Revenue
# MAGIC
# MAGIC **Per the data transformation document:**
# MAGIC - **Total Balance**: `SUM(balance_amount)` grouped by `customer_id` for the most recent snapshot
# MAGIC - **Total Revenue**: `SUM(revenue_amount)` grouped by `customer_id` over a rolling 12-month window

# COMMAND ----------

window_start = reference_date - timedelta(days=LOOKBACK_MONTHS * 30)

balances_windowed = raw_balances.filter(
    (F.col("date") >= F.lit(window_start)) & (F.col("date") <= F.lit(reference_date))
)

revenues_windowed = raw_revenues.filter(
    (F.col("date") >= F.lit(window_start)) & (F.col("date") <= F.lit(reference_date))
)

# Join with product type
balances_with_type = (
    balances_windowed.alias("b")
    .join(product_ref.alias("p"), "product_id", "left")
    .select(
        F.col("b.customer_id"), F.col("b.date"), F.col("b.product_id"),
        F.col("b.balance"),
        F.regexp_replace(F.col("p.product_type"), " ", "_").alias("product_type"),
    )
)

revenues_with_type = (
    revenues_windowed.alias("r")
    .join(product_ref.alias("p"), "product_id", "left")
    .select(
        F.col("r.customer_id"), F.col("r.date"), F.col("r.product_id"),
        F.col("r.revenue"),
        F.regexp_replace(F.col("p.product_type"), " ", "_").alias("product_type"),
    )
)

print(f"Balances in window: {balances_with_type.count():,}")
print(f"Revenues in window: {revenues_with_type.count():,}")

# COMMAND ----------

# Total aggregated balance (latest snapshot)
latest_balance_date = balances_windowed.agg(F.max("date")).collect()[0][0]

total_balance = (
    balances_windowed
    .filter(F.col("date") == F.lit(latest_balance_date))
    .groupBy("customer_id")
    .agg(F.sum("balance").alias("total_balance"))
)

# Total aggregated revenue (12-month rolling sum)
total_revenue = (
    revenues_windowed
    .groupBy("customer_id")
    .agg(F.sum("revenue").alias("total_revenue"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-Product-Type Breakdown (Pivot)

# COMMAND ----------

balance_by_type = (
    balances_with_type.groupBy("customer_id", "product_type")
    .agg(F.avg("balance").alias("balance_avg"), F.last("balance").alias("balance_last"))
)

balance_pivot = (
    balance_by_type.groupBy("customer_id").pivot("product_type")
    .agg(F.first("balance_avg").alias("balance_avg"), F.first("balance_last").alias("balance_last"))
    .fillna(0)
)

revenue_by_type = (
    revenues_with_type.groupBy("customer_id", "product_type")
    .agg(F.avg("revenue").alias("revenue_avg"), F.sum("revenue").alias("revenue_sum"))
)

revenue_pivot = (
    revenue_by_type.groupBy("customer_id").pivot("product_type")
    .agg(F.first("revenue_avg").alias("revenue_avg"), F.first("revenue_sum").alias("revenue_sum"))
    .fillna(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Assemble Financial Features

# COMMAND ----------

financials_features = (
    total_balance
    .join(total_revenue, "customer_id", "outer")
    .join(balance_pivot, "customer_id", "left")
    .join(revenue_pivot, "customer_id", "left")
    .fillna(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Transforms
# MAGIC
# MAGIC **Per the data transformation document:**
# MAGIC > *"log(variable + 1) — the +1 prevents mathematical errors with zero balances.
# MAGIC > Compresses the tail of heavily right-skewed financial data."*

# COMMAND ----------

financials_features = (
    financials_features
    .withColumn("total_balance_log", F.log1p(F.abs(F.col("total_balance"))))
    .withColumn("total_revenue_log", F.log1p(F.abs(F.col("total_revenue"))))
)

print("Applied log1p transforms:")
print("  total_balance -> total_balance_log")
print("  total_revenue -> total_revenue_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary & Save

# COMMAND ----------

fin_count = financials_features.count()
fin_cols = len(financials_features.columns)
print(f"Financials features: {fin_count:,} customers x {fin_cols} columns")
display(financials_features.limit(5))

# COMMAND ----------

financials_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.03_financials_features"
)

saved = spark.table(f"{CATALOG}.{SCHEMA}.03_financials_features").count()
print(f"Saved 03_financials_features: {saved:,} rows x {fin_cols} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC **Output:** `03_financials_features` — `customer_id, total_balance, total_revenue, total_balance_log, total_revenue_log, {product_type}_balance_avg/last, {product_type}_revenue_avg/sum`
