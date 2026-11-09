# Databricks Assistant Prompt: 02d Data Exploration -- Behavioral

## Prompt

Create a Databricks notebook called "02d -- Data Exploration: Behavioral" that performs exploratory data analysis on the behavioral/interaction dataset using ydata-profiling. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install Dependencies**
- First cell: `%pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Set CATALOG and SCHEMA variables.
- Import `pandas as pd` and `pyspark.sql.functions as F`.

**Step 2: Load Raw Behavioral Data**
- Load `spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_additional_info")`.
- Cast the `date` column to date type using `F.to_date()`.
- Print row count, column count, and column names.
- Print schema and display first 5 rows.

**Step 3: Profiling Report**
- Convert date to string for ydata-profiling compatibility: `.withColumn("date", F.col("date").cast("string"))`.
- Convert to pandas.
- Truncate to first 10,000 rows: `df = df.iloc[:10000, :]`.
- Import `ProfileReport` from `ydata_profiling`.
- Create profile: `ProfileReport(df, title="Behavioral Data -- Profiling Report", minimal=True, progress_bar=False, infer_dtypes=False)`.
- Display with `displayHTML(df_profile.to_html())`.

**Step 4: Key Observations**
- Add a markdown cell with a table:

| Observation | Implication |
|-------------|-------------|
| Digital vs in-person shows channel preference | `digital_ratio` feature in feature derivation |
| ATM amount is skewed | May need log transform |
| Physical and ATM services are correlated | Watch for multicollinearity |

- End with: "**Next:** NB03d (Feature Derivation: Behavioral)"

## Expected Output

A notebook that loads the raw additional_info (behavioral) table, generates an interactive ydata-profiling report, and documents key observations about channel preferences and ATM patterns.
