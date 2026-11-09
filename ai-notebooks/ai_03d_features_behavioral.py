# Databricks notebook source
# MAGIC %md
# MAGIC # 03d — Feature Derivation: Behavioral
# MAGIC **Catalog:** `customer_segmentation.ai_banking`
# MAGIC
# MAGIC This notebook aggregates behavioral interaction data into customer-level features
# MAGIC and derives channel preference and cash dependency indicators.
# MAGIC
# MAGIC ### Features Derived
# MAGIC For each of 7 behavioral metrics, compute the **average** and **latest value** over the lookback window:
# MAGIC
# MAGIC | Raw Column | `_avg` | `_last` |
# MAGIC |------------|--------|---------|
# MAGIC | `in_person_interaction` | Avg in-person visits | Most recent count |
# MAGIC | `atm_withdrawals` | Avg ATM frequency | Most recent frequency |
# MAGIC | `atm_withdrawals_sum` | Avg ATM amount | Most recent amount |
# MAGIC | `physical_services` | Avg physical usage | Most recent count |
# MAGIC | `physical_services_sum` | Avg physical amount | Most recent amount |
# MAGIC | `digital_services` | Avg digital usage | Most recent count |
# MAGIC | `fx_services` | Avg FX usage | Most recent count |
# MAGIC
# MAGIC **Plus derived features:**
# MAGIC | Feature | Logic | Type | Purpose |
# MAGIC |---------|-------|------|---------|
# MAGIC | `digital_ratio` | digital / (digital + in_person) | Derived Ratio | Channel preference (0=branch, 1=digital) |
# MAGIC | `atm_intensity` | Passthrough of `atm_withdrawals_sum_avg` | Composite | Cash dependency indicator |
# MAGIC
# MAGIC ### Output Table
# MAGIC `ai_03_behavioral_features` — one row per customer with aggregated behavioral metrics and derived features

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"
LOOKBACK_MONTHS = 12

# COMMAND ----------

from datetime import timedelta
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Behavioral Data

# COMMAND ----------

raw_behavior = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_additional_info")
    .withColumn("date", F.to_date(F.col("date")))
)

reference_date = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
    .agg(F.max("start_date")).collect()[0][0]
)

print(f"Raw behavioral data: {raw_behavior.count():,} rows")
print(f"Reference date: {reference_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate Per Customer
# MAGIC
# MAGIC For each customer, compute the **average** and **latest value** over the lookback window.

# COMMAND ----------

window_start = reference_date - timedelta(days=LOOKBACK_MONTHS * 30)

behavior_windowed = raw_behavior.filter(
    (F.col("date") >= F.lit(window_start)) & (F.col("date") <= F.lit(reference_date))
)

print(f"Records in {LOOKBACK_MONTHS}-month window: {behavior_windowed.count():,}")

# COMMAND ----------

behavior_cols = [
    "in_person_interaction",
    "atm_withdrawals",
    "atm_withdrawals_sum",
    "physical_services",
    "physical_services_sum",
    "digital_services",
    "fx_services",
]

agg_exprs = []
for c in behavior_cols:
    agg_exprs.append(F.avg(c).alias(f"{c}_avg"))
    agg_exprs.append(F.last(c).alias(f"{c}_last"))

behavioral_features = behavior_windowed.groupBy("customer_id").agg(*agg_exprs)

print(f"Base behavioral features: {behavioral_features.count():,} customers x {len(behavioral_features.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Derived Ratio: Digital Ratio
# MAGIC
# MAGIC `digital_ratio = digital_services_avg / (digital_services_avg + in_person_interaction_avg)`
# MAGIC — Channel preference (0 = branch-first, 1 = digital-first)

# COMMAND ----------

behavioral_features = behavioral_features.withColumn(
    "digital_ratio",
    F.when(
        (F.coalesce(F.col("digital_services_avg"), F.lit(0))
         + F.coalesce(F.col("in_person_interaction_avg"), F.lit(0))) > 0,
        F.col("digital_services_avg")
        / (F.col("digital_services_avg") + F.col("in_person_interaction_avg"))
    ).otherwise(0.0)
)

print("Applied: digital_ratio = digital / (digital + in_person)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Composite: ATM Intensity
# MAGIC
# MAGIC `atm_intensity = atm_withdrawals_sum_avg` — Cash dependency indicator

# COMMAND ----------

behavioral_features = behavioral_features.withColumn(
    "atm_intensity",
    F.coalesce(F.col("atm_withdrawals_sum_avg"), F.lit(0.0))
)

print("Applied: atm_intensity = atm_withdrawals_sum_avg")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary & Save

# COMMAND ----------

beh_count = behavioral_features.count()
beh_cols = len(behavioral_features.columns)
print(f"Behavioral features: {beh_count:,} customers x {beh_cols} columns")
display(behavioral_features.limit(5))

# COMMAND ----------

behavioral_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_03_behavioral_features"
)

saved = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_behavioral_features").count()
print(f"Saved ai_03_behavioral_features: {saved:,} rows x {beh_cols} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC **Output:** `ai_03_behavioral_features` — `customer_id, {metric}_avg, {metric}_last, digital_ratio, atm_intensity`
