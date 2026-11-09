# Databricks Assistant Prompt: 02b Data Exploration -- Products

## Prompt

Create a Databricks notebook called "02b -- Data Exploration: Products" that performs exploratory data analysis on the product holdings dataset using ydata-profiling. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install Dependencies**
- First cell: `%pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Set CATALOG and SCHEMA variables.
- Import `pandas as pd` and `pyspark.sql.functions as F`.

**Step 2: Load Raw Product Holdings Data**
- Load `spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_product_holdings")`.
- Cast `start_date` and `end_date` to date type using `F.to_date()`.
- Print row count, column count, and column names.
- Print schema and display first 5 rows.

**Step 3: Enrich for Exploration**
- Get `reference_date` from `raw_holdings.agg(F.max("start_date")).collect()[0][0]`.
- Add derived columns:
  - `is_active`: "Active" when `end_date` is null or `end_date > reference_date`, otherwise "Terminated".
  - `holding_months`: `F.round(F.months_between(F.coalesce(F.col("end_date"), F.lit(reference_date)), F.col("start_date")), 0).cast("int")`
- Cast `start_date` and `end_date` to string for ydata-profiling compatibility.
- Display first 5 rows of the enriched DataFrame.

**Step 4: Profiling Report**
- Convert to pandas with `.toPandas()`.
- Truncate to first 10,000 rows: `df = df.iloc[:10000, :]`.
- Import `ProfileReport` from `ydata_profiling`.
- Create the profile: `ProfileReport(df, title="Product Holdings -- Profiling Report", minimal=True, progress_bar=False, infer_dtypes=False)`.
- Display with `displayHTML(df_profile.to_html())`.

**Step 5: Key Observations**
- Add a markdown cell with a table:

| Observation | Implication |
|-------------|-------------|
| Product diversity ranges from 1-4 types | Good discriminating variable for segments |
| Some product types have higher churn | `product_churn_rate` feature will capture this |
| Multi-product customers are the majority | Cross-sell analysis will be meaningful |

- End with: "**Next:** NB03b (Feature Derivation: Products)"

## Expected Output

A notebook that loads the raw product holdings table, adds is_active flag and holding_months, generates an interactive ydata-profiling report, and documents key observations about product diversity and churn patterns.
