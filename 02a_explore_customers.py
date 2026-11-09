# Databricks notebook source
# MAGIC %md
# MAGIC # 02a — Data Exploration: Customers
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook performs exploratory data analysis on the **customers** dataset
# MAGIC using automated profiling, following the same pattern as Databricks AutoML
# MAGIC Data Exploration notebooks.
# MAGIC
# MAGIC ### What This Notebook Does
# MAGIC 1. Load the raw customers table (`01_raw_customers`)
# MAGIC 2. Run automated profiling with `ydata-profiling` to understand distributions,
# MAGIC    correlations, missing values, and data quality
# MAGIC 3. Display an interactive profiling report
# MAGIC
# MAGIC ### Key Questions This Answers
# MAGIC - What does the customer demographic profile look like?
# MAGIC - Are there any missing values or data quality issues?
# MAGIC - How are income, age, and tenure distributed? Are they skewed?
# MAGIC - What is the gender and tier breakdown?
# MAGIC - Are there correlations between demographic features?

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
# MAGIC ## 1. Load Raw Customers Data

# COMMAND ----------

raw_customers = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_customers")
    .withColumn("creation_date", F.to_date(F.col("creation_date")))
    .withColumn("birth_date", F.to_date(F.col("birth_date")))
)

row_count = raw_customers.count()
col_count = len(raw_customers.columns)
print(f"Raw customers: {row_count:,} rows x {col_count} columns")
print(f"Columns: {raw_customers.columns}")

# COMMAND ----------

raw_customers.printSchema()
display(raw_customers.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Derive Exploration Features
# MAGIC
# MAGIC Before profiling, we compute a few derived fields so the profiling report
# MAGIC captures meaningful distributions (age in years, tenure in months).

# COMMAND ----------

reference_date = (
    spark.table(f"{CATALOG}.{SCHEMA}.01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
    .agg(F.max("start_date")).collect()[0][0]
)
print(f"Reference date: {reference_date}")

# COMMAND ----------

df_explore = (
    raw_customers
    .withColumn("customer_age",
        F.floor(F.datediff(F.lit(reference_date), F.col("birth_date")) / 365.25).cast("int"))
    .withColumn("tenure_months",
        F.round(F.months_between(F.lit(reference_date), F.col("creation_date")), 0).cast("int"))
    .drop("birth_date", "creation_date", "first_name", "last_name")
)

display(df_explore.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Profiling Report
# MAGIC
# MAGIC The profiling report provides:
# MAGIC - **Overview**: dataset statistics, variable types, missing values
# MAGIC - **Variables**: distribution plots, descriptive stats, frequency tables
# MAGIC - **Correlations**: Pearson, Spearman, Kendall, Cramér's V
# MAGIC - **Missing Values**: matrix and bar chart
# MAGIC - **Sample**: first and last rows

# COMMAND ----------

df = df_explore.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Truncate Rows
# MAGIC Only the first 10,000 rows will be considered for profiling to avoid out-of-memory issues.
# MAGIC Comment out the next cell and rerun the notebook to profile the full dataset.

# COMMAND ----------

df = df.iloc[:10000, :]

# COMMAND ----------

from ydata_profiling import ProfileReport

df_profile = ProfileReport(
    df,
    title="Customer Demographics — Profiling Report",
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
# MAGIC After reviewing the profiling report, note any findings here:
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|-------------|
# MAGIC | Income distribution is right-skewed | Apply `log1p` transform in feature derivation |
# MAGIC | Gender has 2 categories | One-hot encode for clustering |
# MAGIC | Tier shows income stratification | Validates segmentation approach |
# MAGIC | Age and tenure are roughly symmetric | No transform needed |
# MAGIC
# MAGIC **Next:** NB03a (Feature Derivation: Customers)
