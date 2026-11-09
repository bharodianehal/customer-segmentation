# Databricks notebook source
# MAGIC %md
# MAGIC # 03a — Feature Derivation: Customers
# MAGIC **Catalog:** `customer_segmentation.ai_banking`
# MAGIC
# MAGIC This notebook derives customer-level features from the raw demographics table.
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC ```
# MAGIC [02a Explore] ──> [03a Features] ──> ai_03_customers_features ──┐
# MAGIC [02b Explore] ──> [03b Features] ──> ai_03_products_features ───┤
# MAGIC [02c Explore] ──> [03c Features] ──> ai_03_financials_features ─┼──> [04] Feature Join
# MAGIC [02d Explore] ──> [03d Features] ──> ai_03_behavioral_features ─┘
# MAGIC ```
# MAGIC
# MAGIC ### Features Derived
# MAGIC | Feature | Logic | Type | Purpose |
# MAGIC |---------|-------|------|---------|
# MAGIC | `customer_age` | `floor((reference_date - birth_date) / 365.25)` | Aggregation | Continuous age variable |
# MAGIC | `tenure_months` | `months_between(reference_date, creation_date)` | Aggregation | Customer lifecycle stage |
# MAGIC | `age_bracket` | Case-when on customer_age | Categorical | Age group for interpretability |
# MAGIC | `income_log` | `log1p(abs(income))` | Log Transform | Reduce right-skew of income |
# MAGIC | `income_quintile` | `ntile(5) over income` | Composite | Ordinal income bucket |
# MAGIC | `gender_f` / `gender_m` | One-hot encode gender | Encoding | Binary columns for clustering |
# MAGIC
# MAGIC ### Output Table
# MAGIC `ai_03_customers_features` — one row per customer with demographics and all derived features

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Customer Data

# COMMAND ----------

raw_customers = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_customers")
    .withColumn("creation_date", F.to_date(F.col("creation_date")))
    .withColumn("birth_date", F.to_date(F.col("birth_date")))
)

row_count = raw_customers.count()
print(f"Raw customers: {row_count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Derive Aggregation Features
# MAGIC
# MAGIC **Per the data transformation document:**
# MAGIC - `customer_age`: `floor((reference_date - birth_date) / 365.25)` — age in years
# MAGIC - `tenure_months`: `months_between(reference_date, creation_date)` — tenure in months (not years)
# MAGIC - `age_bracket`: Categorical binning for interpretability

# COMMAND ----------

# Get reference date from product holdings (latest snapshot)
reference_date = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
    .agg(F.max("start_date"))
    .collect()[0][0]
)
print(f"Reference date: {reference_date}")

# COMMAND ----------

customers_features = (
    raw_customers
    .withColumn("reference_date", F.lit(reference_date))
    .withColumn("customer_age",
        F.floor(F.datediff(F.lit(reference_date), F.col("birth_date")) / 365.25).cast("int"))
    .withColumn("tenure_months",
        F.round(F.months_between(F.lit(reference_date), F.col("creation_date")), 0).cast("int"))
    .withColumn("age_bracket",
        F.when(F.col("customer_age") < 30, "Young")
         .when(F.col("customer_age") < 50, "Middle")
         .when(F.col("customer_age") < 65, "Senior")
         .otherwise("Elderly"))
    .drop("birth_date", "creation_date")
)

print(f"Base features derived: customer_age, tenure_months, age_bracket")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log Transform: Income
# MAGIC
# MAGIC **Per the data transformation document:**
# MAGIC > *"log(variable + 1) — the +1 prevents mathematical errors with zero values.
# MAGIC > Compresses the tail of heavily right-skewed data."*

# COMMAND ----------

customers_features = customers_features.withColumn(
    "income_log", F.log1p(F.abs(F.col("income")))
)

print("Applied: income_log = log1p(|income|)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Income Quintile

# COMMAND ----------

customers_features = customers_features.withColumn(
    "income_quintile", F.ntile(5).over(Window.orderBy("income"))
)

print("Applied: income_quintile = ntile(5) over income")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. One-Hot Encoding: Gender
# MAGIC
# MAGIC **Per the data transformation document:**
# MAGIC > *"Converts categorical text into binary columns. Machine learning algorithms
# MAGIC > require numerical inputs and cannot interpret raw text strings."*

# COMMAND ----------

distinct_genders = [row[0] for row in customers_features.select("gender").distinct().collect() if row[0] is not None]
distinct_genders.sort()

for g in distinct_genders:
    col_name = f"gender_{g.lower().replace(' ', '_')}"
    customers_features = customers_features.withColumn(
        col_name, F.when(F.col("gender") == g, 1).otherwise(0)
    )
    print(f"  Created: {col_name}")

print(f"\nOne-hot encoded gender into {len(distinct_genders)} binary columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary & Save

# COMMAND ----------

print(f"Customers features: {customers_features.count():,} rows x {len(customers_features.columns)} cols")
display(customers_features.limit(5))

# COMMAND ----------

customers_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_03_customers_features"
)

saved = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_customers_features").count()
print(f"Saved ai_03_customers_features: {saved:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC **Output:** `ai_03_customers_features` — `customer_id, gender, region, income, tier, first_name, last_name, reference_date, customer_age, tenure_months, age_bracket, income_log, income_quintile, gender_f, gender_m`
