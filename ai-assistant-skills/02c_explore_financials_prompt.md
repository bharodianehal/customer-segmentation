# Databricks Assistant Prompt: 02c Data Exploration -- Financials

## Prompt

Create a Databricks notebook called "02c -- Data Exploration: Financials (Balances & Revenue)" that performs exploratory data analysis on the balance and revenue datasets using ydata-profiling. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install Dependencies**
- First cell: `%pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Set CATALOG and SCHEMA variables.
- Import `pandas as pd` and `pyspark.sql.functions as F`.

**Step 2: Load Raw Financial Data**
- Load `ai_01_raw_balances` and `ai_01_raw_revenues` from the catalog.
- Cast the `date` column to date type in both DataFrames using `F.to_date()`.
- Print row counts and column counts for both datasets.
- Print schema and display first 5 rows for each.

**Step 3: Profiling Report -- Balances**
- Convert date column to string for ydata-profiling compatibility: `.withColumn("date", F.col("date").cast("string"))`.
- Convert to pandas.
- Truncate to first 10,000 rows.
- Import `ProfileReport` from `ydata_profiling`.
- Create profile: `ProfileReport(df_bal, title="Balances -- Profiling Report", minimal=True, progress_bar=False, infer_dtypes=False)`.
- Display with `displayHTML(bal_profile.to_html())`.

**Step 4: Profiling Report -- Revenues**
- Same pattern: cast date to string, convert to pandas, truncate to 10,000 rows.
- Create profile: `ProfileReport(df_rev, title="Revenues -- Profiling Report", minimal=True, progress_bar=False, infer_dtypes=False)`.
- Display with `displayHTML(rev_profile.to_html())`.

**Step 5: Key Observations**
- Add a markdown cell with a table:

| Observation | Implication |
|-------------|-------------|
| Balance distribution is right-skewed | Apply `log1p` transform in feature derivation |
| Revenue distribution is right-skewed | Apply `log1p` transform in feature derivation |
| Balance and revenue are correlated | May have redundancy -- monitor VIF |
| Product-level breakdowns add granularity | Use for cross-sell analysis downstream |

- End with: "**Next:** NB03c (Feature Derivation: Financials)"

## Expected Output

A notebook that loads the raw balances and revenues tables, generates two separate interactive ydata-profiling reports (one for balances, one for revenues), and documents key observations about skewness and correlations in financial data.
