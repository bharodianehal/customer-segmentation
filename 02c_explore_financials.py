# Databricks notebook source
# MAGIC %md
# MAGIC # 02c — Data Exploration: Financials (Balances & Revenue)
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook performs exploratory data analysis on the **balance** and **revenue** datasets
# MAGIC using automated profiling.
# MAGIC
# MAGIC ### What This Notebook Does
# MAGIC 1. Load the raw balances (`01_raw_balances`) and revenues (`01_raw_revenues`)
# MAGIC 2. Run automated profiling with `ydata-profiling`
# MAGIC 3. Display interactive profiling reports for both datasets
# MAGIC
# MAGIC ### Key Questions This Answers
# MAGIC - How are balances and revenues distributed? Are they heavily skewed?
# MAGIC - What is the relationship between balance and revenue?
# MAGIC - Are there seasonal patterns or outliers?
# MAGIC - How do financials vary by product type?

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
# MAGIC ## 1. Load Raw Financial Data

# COMMAND ----------

raw_balances = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_balances")
    .withColumn("date", F.to_date(F.col("date")))
)

raw_revenues = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_revenues")
    .withColumn("date", F.to_date(F.col("date")))
)

print(f"Raw balances: {raw_balances.count():,} rows x {len(raw_balances.columns)} columns")
print(f"Raw revenues: {raw_revenues.count():,} rows x {len(raw_revenues.columns)} columns")

# COMMAND ----------

raw_balances.printSchema()
display(raw_balances.limit(5))

# COMMAND ----------

raw_revenues.printSchema()
display(raw_revenues.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Profiling Report: Balances

# COMMAND ----------

# Convert date to string for ydata-profiling compatibility
df_bal = raw_balances.withColumn("date", F.col("date").cast("string")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Truncate Rows
# MAGIC Only the first 10,000 rows will be considered for profiling to avoid out-of-memory issues.

# COMMAND ----------

df_bal = df_bal.iloc[:10000, :]

# COMMAND ----------

from ydata_profiling import ProfileReport

bal_profile = ProfileReport(
    df_bal,
    title="Balances — Profiling Report",
    minimal=True,
    progress_bar=False,
    infer_dtypes=False,
)
displayHTML(bal_profile.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Profiling Report: Revenues

# COMMAND ----------

# Convert date to string for ydata-profiling compatibility
df_rev = raw_revenues.withColumn("date", F.col("date").cast("string")).toPandas()
df_rev = df_rev.iloc[:10000, :]

# COMMAND ----------

rev_profile = ProfileReport(
    df_rev,
    title="Revenues — Profiling Report",
    minimal=True,
    progress_bar=False,
    infer_dtypes=False,
)
displayHTML(rev_profile.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Key Observations
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|-------------|
# MAGIC | Balance distribution is right-skewed | Apply `log1p` transform in feature derivation |
# MAGIC | Revenue distribution is right-skewed | Apply `log1p` transform in feature derivation |
# MAGIC | Balance and revenue are correlated | May have redundancy — monitor VIF |
# MAGIC | Product-level breakdowns add granularity | Use for cross-sell analysis downstream |
# MAGIC
# MAGIC **Next:** NB03c (Feature Derivation: Financials)
