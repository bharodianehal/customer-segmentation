# Databricks notebook source
# MAGIC %md
# MAGIC # 01 -- Data Ingestion
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC This notebook ingests 7 raw TSV files from the Unity Catalog Volume into Delta tables.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Load each TSV file and display sample rows and schema
# MAGIC 2. Run basic data quality checks (null counts, duplicate keys)
# MAGIC 3. Write raw Delta tables with `01_raw_` prefix
# MAGIC 4. Print a final summary

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "banking"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw TSV Files

# COMMAND ----------

def read_tsv(filename: str) -> DataFrame:
    """Read a tab-separated file from the raw_data volume."""
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("sep", "\t")
        .load(f"{VOL}/{filename}.csv")
    )

# Dataset name -> (source filename, primary key columns)
DATASET_CONFIG = {
    "customers":        ("customers",              ["customer_id"]),
    "balances":         ("balances",                ["customer_id"]),
    "revenues":         ("revenues",                ["customer_id"]),
    "products":         ("products",                ["product_id"]),
    "product_holdings": ("product_holdings",        ["customer_id", "product_id"]),
    "additional_info":  ("additional_information",  ["customer_id"]),
    "dates":            ("dates",                   ["date"]),
}

datasets = {}
for name, (filename, _) in DATASET_CONFIG.items():
    datasets[name] = read_tsv(filename)

print(f"Loaded {len(datasets)} datasets from {VOL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sample Rows and Schemas
# MAGIC Display the first 5 rows and the schema for each dataset.

# COMMAND ----------

for name, df in datasets.items():
    row_count = df.count()
    col_count = len(df.columns)
    print("=" * 80)
    print(f"  DATASET: {name}  |  {row_count:,} rows  |  {col_count} columns")
    print("=" * 80)

    print("\n--- Schema ---")
    df.printSchema()

    print("--- Sample (first 5 rows) ---")
    df.show(5, truncate=False)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Checks
# MAGIC For each dataset:
# MAGIC - Count of nulls per column
# MAGIC - Duplicate check on primary key columns

# COMMAND ----------

dq_results = []

for name, df in datasets.items():
    _, pk_cols = DATASET_CONFIG[name]
    total_rows = df.count()

    # --- Null counts per column ---
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_counts = df.select(null_exprs).collect()[0]

    cols_with_nulls = {
        c: null_counts[c] for c in df.columns if null_counts[c] and null_counts[c] > 0
    }

    # --- Duplicate key check ---
    if all(c in df.columns for c in pk_cols):
        distinct_keys = df.select(pk_cols).distinct().count()
        duplicate_keys = total_rows - distinct_keys
    else:
        distinct_keys = None
        duplicate_keys = None

    dq_results.append({
        "dataset": name,
        "total_rows": total_rows,
        "total_cols": len(df.columns),
        "pk_cols": pk_cols,
        "distinct_keys": distinct_keys,
        "duplicate_keys": duplicate_keys,
        "cols_with_nulls": cols_with_nulls,
    })

    # Print per-dataset DQ report
    print("=" * 80)
    print(f"  DQ CHECK: {name}  |  Primary Key: {pk_cols}")
    print("=" * 80)

    if cols_with_nulls:
        print(f"\n  Null counts:")
        for col_name, cnt in cols_with_nulls.items():
            pct = 100.0 * cnt / total_rows if total_rows > 0 else 0
            print(f"    {col_name:40s}  {cnt:>8,} nulls  ({pct:5.1f}%)")
    else:
        print("\n  No null values found in any column.")

    if duplicate_keys is not None:
        if duplicate_keys > 0:
            print(f"\n  WARNING: {duplicate_keys:,} duplicate rows on key {pk_cols}")
        else:
            print(f"\n  Primary key {pk_cols} is unique. No duplicates.")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write Raw Delta Tables

# COMMAND ----------

for name, df in datasets.items():
    table = f"{CATALOG}.{SCHEMA}.01_raw_{name}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    print(f"  Saved: {table}")

print("\nAll raw tables written to Delta.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Final Summary

# COMMAND ----------

# Build summary table
print("=" * 110)
print(f"  {'Dataset':<22s} {'Rows':>10s} {'Cols':>6s} {'Primary Key':<30s} {'Dup Keys':>10s} {'Null Cols':>10s} {'Status':<10s}")
print("-" * 110)

all_ok = True
for r in dq_results:
    null_col_count = len(r["cols_with_nulls"])
    dup_keys = r["duplicate_keys"] if r["duplicate_keys"] is not None else 0
    status = "OK" if dup_keys == 0 else "WARNING"
    if dup_keys > 0:
        all_ok = False

    pk_str = ", ".join(r["pk_cols"])
    print(
        f"  {r['dataset']:<22s} "
        f"{r['total_rows']:>10,} "
        f"{r['total_cols']:>6} "
        f"{pk_str:<30s} "
        f"{dup_keys:>10,} "
        f"{null_col_count:>10} "
        f"{status:<10s}"
    )

print("-" * 110)
total_rows_all = sum(r["total_rows"] for r in dq_results)
print(f"  {'TOTAL':<22s} {total_rows_all:>10,}")
print("=" * 110)

if all_ok:
    print("\n  All primary key checks passed. Data ingestion complete.")
else:
    print("\n  WARNING: Some datasets have duplicate keys. Review before proceeding.")
