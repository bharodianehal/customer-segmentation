# Databricks Assistant Prompt: 03a Feature Derivation -- Customers

## Prompt

Create a Databricks notebook called "03a -- Feature Derivation: Customers" that derives customer-level features from the raw demographics table. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

Start with a markdown header that includes a pipeline context diagram showing how 03a/b/c/d features feed into 04 Feature Join, and a features table:

| Feature | Logic | Type | Purpose |
|---------|-------|------|---------|
| `customer_age` | `floor((reference_date - birth_date) / 365.25)` | Aggregation | Continuous age variable |
| `tenure_months` | `months_between(reference_date, creation_date)` | Aggregation | Customer lifecycle stage |
| `age_bracket` | Case-when on customer_age | Categorical | Age group for interpretability |
| `income_log` | `log1p(abs(income))` | Log Transform | Reduce right-skew of income |
| `income_quintile` | `ntile(5) over income` | Composite | Ordinal income bucket |
| `gender_f` / `gender_m` | One-hot encode gender | Encoding | Binary columns for clustering |

**Step 1: Setup**
- Set CATALOG and SCHEMA. Import `pyspark.sql.functions as F` and `Window` from `pyspark.sql.window`.

**Step 2: Load Raw Customer Data**
- Load `ai_01_raw_customers`. Cast `creation_date` and `birth_date` to date type. Print row count.

**Step 3: Derive Aggregation Features**
- Get `reference_date` from max `start_date` in `ai_01_raw_product_holdings` (cast to date first). Print it.
- Create `customers_features` DataFrame from `raw_customers`:
  - Add `reference_date` column as literal.
  - `customer_age`: `F.floor(F.datediff(F.lit(reference_date), F.col("birth_date")) / 365.25).cast("int")`
  - `tenure_months`: `F.round(F.months_between(F.lit(reference_date), F.col("creation_date")), 0).cast("int")`
  - `age_bracket`: `F.when(col < 30, "Young").when(col < 50, "Middle").when(col < 65, "Senior").otherwise("Elderly")`
  - Drop `birth_date` and `creation_date`.

**Step 4: Log Transform -- Income**
- Add markdown explaining log1p transform rationale.
- `income_log = F.log1p(F.abs(F.col("income")))`

**Step 5: Income Quintile**
- `income_quintile = F.ntile(5).over(Window.orderBy("income"))`

**Step 6: One-Hot Encoding -- Gender**
- Add markdown explaining why ML algorithms need numerical inputs.
- Get distinct gender values, sort them.
- For each gender value `g`, create column `gender_{g.lower()}` with value 1 when gender matches, 0 otherwise.
- Print each created column name.

**Step 7: Summary and Save**
- Print final row count and column count. Display first 5 rows.
- Write to `{CATALOG}.{SCHEMA}.ai_03_customers_features` with `mode("overwrite").option("overwriteSchema", "true")`.
- Read back and print saved row count.
- Add a markdown cell documenting the output columns.

## Expected Output

A notebook that produces `ai_03_customers_features` -- one row per customer with: customer_id, gender, region, income, tier, first_name, last_name, reference_date, customer_age, tenure_months, age_bracket, income_log, income_quintile, gender_f, gender_m.
