# Databricks Assistant Prompt: 02a Data Exploration -- Customers

## Prompt

Create a Databricks notebook called "02a -- Data Exploration: Customers" that performs exploratory data analysis on the customers dataset using ydata-profiling. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install Dependencies**
- First cell: `%pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Set CATALOG and SCHEMA variables.
- Import `pandas as pd` and `pyspark.sql.functions as F`.

**Step 2: Load Raw Customers Data**
- Load `spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_customers")`.
- Cast `creation_date` and `birth_date` to date type using `F.to_date()`.
- Print row count, column count, and column names.
- Print schema and display first 5 rows.

**Step 3: Derive Exploration Features**
- Get `reference_date` from the max `start_date` in `ai_01_raw_product_holdings` (cast start_date to date first).
- From the raw customers DataFrame, derive:
  - `customer_age`: `F.floor(F.datediff(F.lit(reference_date), F.col("birth_date")) / 365.25).cast("int")`
  - `tenure_months`: `F.round(F.months_between(F.lit(reference_date), F.col("creation_date")), 0).cast("int")`
- Drop columns: `birth_date`, `creation_date`, `first_name`, `last_name`.
- Display first 5 rows of the enriched DataFrame.

**Step 4: Profiling Report**
- Add a markdown cell explaining what the profiling report provides (Overview, Variables, Correlations, Missing Values, Sample).
- Convert to pandas with `.toPandas()`.
- Truncate to first 10,000 rows: `df = df.iloc[:10000, :]` (add a markdown note about this).
- Import `ProfileReport` from `ydata_profiling`.
- Create the profile: `ProfileReport(df, title="Customer Demographics -- Profiling Report", minimal=True, progress_bar=False, infer_dtypes=False)`.
- Convert to HTML and display with `displayHTML(df_profile.to_html())`.

**Step 5: Key Observations**
- Add a markdown cell with a table of key observations:

| Observation | Implication |
|-------------|-------------|
| Income distribution is right-skewed | Apply `log1p` transform in feature derivation |
| Gender has 2 categories | One-hot encode for clustering |
| Tier shows income stratification | Validates segmentation approach |
| Age and tenure are roughly symmetric | No transform needed |

- End with: "**Next:** NB03a (Feature Derivation: Customers)"

## Expected Output

A notebook that loads the raw customers table, derives customer_age and tenure_months for richer profiling, generates an interactive ydata-profiling report displayed inline, and documents key observations for downstream feature engineering.
