# Databricks notebook source
# MAGIC %md
# MAGIC # 02d — Data Exploration: Behavioral
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook performs exploratory data analysis on the **behavioral / interaction**
# MAGIC dataset using automated profiling.
# MAGIC
# MAGIC ### What This Notebook Does
# MAGIC 1. Load the raw behavioral data (`01_raw_additional_info`)
# MAGIC 2. Run automated profiling with `ydata-profiling`
# MAGIC 3. Display an interactive profiling report
# MAGIC
# MAGIC ### Key Questions This Answers
# MAGIC - How do customers interact with the bank (digital vs in-person)?
# MAGIC - What are ATM withdrawal patterns?
# MAGIC - Are behavioral features correlated with each other?
# MAGIC - Are there missing values or quality issues?

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
# MAGIC ## 1. Load Raw Behavioral Data

# COMMAND ----------

raw_behavior = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_additional_info")
    .withColumn("date", F.to_date(F.col("date")))
)

row_count = raw_behavior.count()
col_count = len(raw_behavior.columns)
print(f"Raw behavioral data: {row_count:,} rows x {col_count} columns")
print(f"Columns: {raw_behavior.columns}")

# COMMAND ----------

raw_behavior.printSchema()
display(raw_behavior.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Profiling Report

# COMMAND ----------

# Convert date to string for ydata-profiling compatibility
df = raw_behavior.withColumn("date", F.col("date").cast("string")).toPandas()

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
    title="Behavioral Data — Profiling Report",
    minimal=True,
    progress_bar=False,
    infer_dtypes=False,
)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Key Observations
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|-------------|
# MAGIC | Digital vs in-person shows channel preference | `digital_ratio` feature in feature derivation |
# MAGIC | ATM amount is skewed | May need log transform |
# MAGIC | Physical and ATM services are correlated | Watch for multicollinearity |
# MAGIC
# MAGIC **Next:** NB03d (Feature Derivation: Behavioral)
