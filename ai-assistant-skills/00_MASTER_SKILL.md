# Databricks Assistant Skill: Bank Customer Segmentation Pipeline

## Skill Name
`Bank Customer Segmentation`

## Skill Description
Build an end-to-end customer segmentation pipeline for a retail bank using K-Means clustering. The pipeline ingests raw banking data, explores it with automated profiling, derives domain-specific features, joins them into a unified feature table registered in Unity Catalog Feature Store, trains a K-Means model, and validates segment stability over time.

## Skill Instructions

You are an expert Data Scientist building a customer segmentation pipeline in Databricks. When asked to build this pipeline, generate the following notebooks in order. Each notebook should be a standalone Databricks notebook with `# COMMAND ----------` cell separators and `# MAGIC %md` markdown cells for documentation.

### Environment
- **Catalog:** `customer_segmentation`
- **Schema:** `ai_banking`
- **Volume:** `/Volumes/customer_segmentation/ai_banking/raw_data`
- **Compute:** Serverless
- **Notebook prefix:** `ai_`
- **Job prefix:** `ai_`

### Source Data (7 TSV files in the Volume)
| File | Key Columns | Description |
|------|-------------|-------------|
| `customers.csv` | `customer_id` | Demographics: gender, region, income, tier, birth_date, creation_date |
| `products.csv` | `product_id` | Product catalog: product name, product_type |
| `product_holdings.csv` | `customer_id, product_id` | Which products each customer holds, with start/end dates |
| `balances.csv` | `customer_id, product_id, date` | Monthly balance snapshots per product |
| `revenues.csv` | `customer_id, product_id, date` | Monthly revenue per product |
| `additional_information.csv` | `customer_id, date` | Behavioral metrics: ATM, digital, physical, FX services |
| `dates.csv` | `date` | Reference date calendar |

### Pipeline Architecture (DAG)
```
[01 Ingestion] (run once)

[02a Explore Customers]   ──> [03a Features Customers]   ──┐
[02b Explore Products]    ──> [03b Features Products]    ───┤
[02c Explore Financials]  ──> [03c Features Financials]  ───┼──> [04 Feature Join] ──> [05 Model Calibration] ──> [06 Model Validation]
[02d Explore Behavioral]  ──> [03d Features Behavioral]  ───┘
```

- 02a-02d run in **parallel**
- 03a-03d run in **parallel** (each depends on its corresponding 02x)
- 04 depends on ALL of 03a-03d
- 05 depends on 04
- 06 depends on 05

### Notebook Specifications

When generating each notebook, follow these rules:
1. Start with a `%md` header cell showing notebook title, catalog reference, purpose, features derived (as a table), and output table name
2. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"` as the first code cell
3. Include `# COMMAND ----------` between every cell
4. Use PySpark for all data transformations
5. Use `display()` for showing sample data
6. Save tables with `.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable()`
7. Print row counts after each major operation
8. End with a `%md` cell summarizing the output table and its columns

### Feature Engineering Rules
ALL feature engineering happens in 03a-03d. Notebook 04 is ONLY a join + Feature Store registration.

| Feature | Notebook | Logic | Type |
|---------|----------|-------|------|
| `customer_age` | 03a | `floor(datediff(ref_date, birth_date) / 365.25)` | Aggregation |
| `tenure_months` | 03a | `months_between(ref_date, creation_date)` | Aggregation |
| `age_bracket` | 03a | Young(<30), Middle(30-49), Senior(50-64), Elderly(65+) | Categorical |
| `income_log` | 03a | `log1p(abs(income))` | Log Transform |
| `income_quintile` | 03a | `ntile(5) over income` | Composite |
| `gender_f`, `gender_m` | 03a | One-hot encode gender | Encoding |
| `nb_subscribed_products` | 03b | COUNT(*) all holdings | Aggregation |
| `nb_terminated_products` | 03b | COUNT where end_date not null | Aggregation |
| `nb_active_products` | 03b | subscribed - terminated | Aggregation |
| `product_diversity` | 03b | COUNT(DISTINCT product_type) where active | Aggregation |
| `product_churn_rate` | 03b | terminated / subscribed | Derived Ratio |
| `net_product_growth` | 03b | active - terminated | Composite |
| `total_balance` | 03c | SUM(balance) latest snapshot | Aggregation |
| `total_revenue` | 03c | SUM(revenue) 12-month window | Aggregation |
| `total_balance_log` | 03c | `log1p(abs(total_balance))` | Log Transform |
| `total_revenue_log` | 03c | `log1p(abs(total_revenue))` | Log Transform |
| Per-product-type pivots | 03c | balance_avg/last, revenue_avg/sum by product_type | Aggregation |
| 7 behavioral _avg/_last | 03d | avg and last per customer over 12-month window | Aggregation |
| `digital_ratio` | 03d | digital / (digital + in_person) | Derived Ratio |
| `atm_intensity` | 03d | atm_withdrawals_sum_avg passthrough | Composite |
| `revenue_per_product` | 04 | total_revenue / nb_subscribed (cross-domain) | Derived Ratio |

### Model Specifications (NB05)
- Algorithm: K-Means clustering
- K sweep: 3 to 8, with elbow/silhouette/CH/DB metrics
- Final K: 4
- Preprocessing: log-transform skewed columns, StandardScaler
- Visualizations: 2x2 metric grid, segment bar chart, heatmap, radar chart
- Auto-labeling: rule-based segment names (Premium High-Value, Digital-First Engaged, etc.)
- Output tables: cluster_metrics, cluster_profiles, customer_reference_scored, customer_segments, age_demographics, customers_nbo

### Feature Store Registration (NB04)
Use `from databricks.feature_engineering import FeatureEngineeringClient`:
```python
fe = FeatureEngineeringClient()
fe.create_table(name="...", primary_keys=["customer_id"], df=df, description="...")
```
Tables appear under AI/ML > Features with special feature table icon.

### Data Exploration (NB02a-02d)
Use `ydata-profiling` for automated EDA:
```python
%pip install ydata-profiling==4.8.3 pandas==2.2.3 --quiet
```
Then `ProfileReport(df, title="...", minimal=True, progress_bar=False, infer_dtypes=False)` and `displayHTML(profile.to_html())`.
Cast date columns to strings before `.toPandas()` to avoid datetime.date errors.
Truncate to 10,000 rows for profiling.
