# Databricks Assistant Prompt: 01 Data Ingestion

## Prompt

Create a Databricks notebook called "01 -- Data Ingestion" that ingests 7 raw TSV files from a Unity Catalog Volume into Delta tables. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`. The volume path is `/Volumes/customer_segmentation/ai_banking/raw_data/`.

**Step 1: Setup**
- Set CATALOG, SCHEMA, and VOL variables.
- Run `spark.sql(f"USE CATALOG {CATALOG}")` and `spark.sql(f"USE SCHEMA {SCHEMA}")`.
- Import `pyspark.sql.functions as F`.

**Step 2: Load TSV Files**
- Create a `read_tsv(filename)` helper function that reads CSV format with `header=true`, `inferSchema=true`, `sep="\t"`, loading from `{VOL}/{filename}.csv`.
- Define a `DATASET_CONFIG` dictionary mapping dataset names to (source_filename, primary_key_columns):
  - `"customers"` -> `("customers", ["customer_id"])`
  - `"balances"` -> `("balances", ["customer_id"])`
  - `"revenues"` -> `("revenues", ["customer_id"])`
  - `"products"` -> `("products", ["product_id"])`
  - `"product_holdings"` -> `("product_holdings", ["customer_id", "product_id"])`
  - `"additional_info"` -> `("additional_information", ["customer_id"])`
  - `"dates"` -> `("dates", ["date"])`
- Load all 7 into a `datasets` dict using a loop. Print count of loaded datasets.

**Step 3: Display Schemas and Samples**
- For each dataset, print a formatted header with dataset name, row count, and column count.
- Print the schema using `printSchema()` and show the first 5 rows with `show(5, truncate=False)`.

**Step 4: Data Quality Checks**
- For each dataset:
  - Compute null counts per column using `F.sum(F.when(F.col(c).isNull(), 1).otherwise(0))` for each column.
  - Track columns with nulls in a dict `{column_name: null_count}`.
  - Check for duplicate keys: compare `df.select(pk_cols).distinct().count()` vs `total_rows`.
  - Store results in a `dq_results` list of dicts with keys: dataset, total_rows, total_cols, pk_cols, distinct_keys, duplicate_keys, cols_with_nulls.
  - Print a per-dataset DQ report showing null counts (with percentages) and duplicate key warnings.

**Step 5: Write Delta Tables**
- For each dataset, write with `mode("overwrite").option("overwriteSchema", "true").saveAsTable()` using the naming convention `{CATALOG}.{SCHEMA}.ai_01_raw_{name}` (e.g., `ai_01_raw_customers`, `ai_01_raw_balances`, etc.).
- Print each saved table name.

**Step 6: Final Summary**
- Print a formatted summary table with columns: Dataset, Rows, Cols, Primary Key, Dup Keys, Null Cols, Status.
- Status is "OK" if no duplicate keys, "WARNING" otherwise.
- Print total row count across all datasets.
- Print whether all primary key checks passed.

Use `%md` markdown cells for section headers: "1. Load Raw TSV Files", "2. Sample Rows and Schemas", "3. Data Quality Checks", "4. Write Raw Delta Tables", "5. Final Summary".

## Expected Output

A notebook that reads 7 TSV files from the Volume, displays schemas and sample rows, runs null and duplicate-key DQ checks, writes 7 Delta tables (`ai_01_raw_customers`, `ai_01_raw_balances`, `ai_01_raw_revenues`, `ai_01_raw_products`, `ai_01_raw_product_holdings`, `ai_01_raw_additional_info`, `ai_01_raw_dates`), and prints a summary report.
