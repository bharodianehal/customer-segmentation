# Databricks notebook source
# MAGIC %md
# MAGIC # 02b — Data Exploration: Products
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook performs exploratory data analysis on the **product holdings** dataset
# MAGIC using automated profiling.
# MAGIC
# MAGIC ### What This Notebook Does
# MAGIC 1. Load the raw product holdings table (`01_raw_product_holdings`)
# MAGIC 2. Run automated profiling with `ydata-profiling`
# MAGIC 3. Display an interactive profiling report
# MAGIC
# MAGIC ### Key Questions This Answers
# MAGIC - What product types exist and how common is each?
# MAGIC - What is the ratio of active to terminated products?
# MAGIC - How many products do customers typically hold?
# MAGIC - Are there data quality issues (nulls, duplicates)?

# COMMAND ----------

# MAGIC %pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "banking"

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Product Holdings Data

# COMMAND ----------

raw_holdings = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
    .withColumn("end_date", F.to_date(F.col("end_date")))
)

row_count = raw_holdings.count()
col_count = len(raw_holdings.columns)
print(f"Raw product holdings: {row_count:,} rows x {col_count} columns")
print(f"Columns: {raw_holdings.columns}")

# COMMAND ----------

raw_holdings.printSchema()
display(raw_holdings.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Enrich for Exploration
# MAGIC
# MAGIC Add derived flags (is_active, holding_duration) to make the profiling report
# MAGIC more informative.

# COMMAND ----------

reference_date = raw_holdings.agg(F.max("start_date")).collect()[0][0]
print(f"Reference date: {reference_date}")

df_explore = (
    raw_holdings
    .withColumn("is_active",
        F.when(F.col("end_date").isNull(), "Active")
         .when(F.col("end_date") > F.lit(reference_date), "Active")
         .otherwise("Terminated"))
    .withColumn("holding_months",
        F.round(F.months_between(
            F.coalesce(F.col("end_date"), F.lit(reference_date)),
            F.col("start_date")
        ), 0).cast("int"))
    # Convert dates to strings for ydata-profiling compatibility
    .withColumn("start_date", F.col("start_date").cast("string"))
    .withColumn("end_date", F.col("end_date").cast("string"))
)

display(df_explore.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Profiling Report

# COMMAND ----------

df = df_explore.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Truncate Rows
# MAGIC Only the first 10,000 rows will be considered for profiling to avoid out-of-memory issues.

# COMMAND ----------

df = df.iloc[:10000, :]

# COMMAND ----------

from ydata_profiling import ProfileReport

df_profile = ProfileReport(
    df,
    title="Product Holdings — Profiling Report",
    minimal=True,
    progress_bar=False,
    infer_dtypes=False,
)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Key Observations
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|-------------|
# MAGIC | Product diversity ranges from 1-4 types | Good discriminating variable for segments |
# MAGIC | Some product types have higher churn | `product_churn_rate` feature will capture this |
# MAGIC | Multi-product customers are the majority | Cross-sell analysis will be meaningful |
# MAGIC
# MAGIC **Next:** NB03b (Feature Derivation: Products)
