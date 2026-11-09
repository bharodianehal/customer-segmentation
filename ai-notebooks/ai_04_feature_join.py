# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Feature Join & Feature Store Registration
# MAGIC **Catalog:** `customer_segmentation.ai_banking`
# MAGIC
# MAGIC This notebook **joins** the four domain feature tables (each already containing
# MAGIC all their derived features from NB03a-03d) and registers the result in
# MAGIC Unity Catalog Feature Store.
# MAGIC
# MAGIC ### Input Tables (from NB03a-03d)
# MAGIC ```
# MAGIC ai_03_customers_features  (demographics + income_log, income_quintile, gender encoding) ──┐
# MAGIC ai_03_products_features   (portfolio + product_churn_rate, net_product_growth) ────────────┤
# MAGIC ai_03_financials_features (aggregates + total_balance_log, total_revenue_log) ────────────┼──> [04] Join
# MAGIC ai_03_behavioral_features (metrics + digital_ratio, atm_intensity) ───────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Steps
# MAGIC | Step | What |
# MAGIC |------|------|
# MAGIC | 4.1 | **Join** all 4 domain tables on `customer_id` |
# MAGIC | 4.2 | **Cross-domain feature**: `revenue_per_product` (needs data from financials + products) |
# MAGIC | 4.3 | **Feature summary** — stats check |
# MAGIC | 4.4 | **Panel build** — historical data for temporal validation |
# MAGIC | 4.5 | **Feature Store** — register tables |
# MAGIC | 4.6 | **Column descriptions** |
# MAGIC
# MAGIC ### Output Tables
# MAGIC | Table | PK | Use |
# MAGIC |-------|-----|-----|
# MAGIC | `ai_ai_04_customer_features_reference` | `customer_id` | Latest month — clustering/training |
# MAGIC | `ai_04_customer_features` | `customer_id, reference_date` | Panel — temporal validation |

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
from pyspark.sql.window import Window

fe = FeatureEngineeringClient()
print("FeatureEngineeringClient initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.1 Join All Domain Feature Tables

# COMMAND ----------

customers = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_customers_features")
products = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_products_features")
financials = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_financials_features")
behavioral = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_behavioral_features")

print(f"Customers:  {customers.count():,} rows x {len(customers.columns)} cols")
print(f"Products:   {products.count():,} rows x {len(products.columns)} cols")
print(f"Financials: {financials.count():,} rows x {len(financials.columns)} cols")
print(f"Behavioral: {behavioral.count():,} rows x {len(behavioral.columns)} cols")

# COMMAND ----------

joined = (
    customers
    .join(products, "customer_id", "left")
    .join(financials, "customer_id", "left")
    .join(behavioral, "customer_id", "left")
    .fillna(0)
)

joined_count = joined.count()
joined_cols = len(joined.columns)
print(f"Joined table: {joined_count:,} rows x {joined_cols} columns")
display(joined.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.2 Cross-Domain Feature: Revenue Per Product
# MAGIC
# MAGIC This feature requires data from both financials (`total_revenue`) and products
# MAGIC (`nb_subscribed_products`), so it can only be computed after the join.

# COMMAND ----------

df = joined.withColumn(
    "revenue_per_product",
    F.when(F.col("nb_subscribed_products") > 0,
           F.col("total_revenue") / F.col("nb_subscribed_products"))
    .otherwise(0.0)
)

print("Applied: revenue_per_product = total_revenue / nb_subscribed_products")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.3 Feature Summary & Quality Check

# COMMAND ----------

ENGINEERED_FEATURES = [
    "income_log", "total_balance_log", "total_revenue_log",
    "digital_ratio", "revenue_per_product", "product_churn_rate",
    "net_product_growth", "atm_intensity", "income_quintile",
]

# Add any gender one-hot columns dynamically
gender_cols = [c for c in df.columns if c.startswith("gender_") and c != "gender"]
ENGINEERED_FEATURES.extend(gender_cols)

numeric_eng = [c for c in ENGINEERED_FEATURES if c in df.columns]
agg_exprs = []
for c in numeric_eng:
    agg_exprs.append(F.min(F.col(c)).alias(f"{c}__min"))
    agg_exprs.append(F.max(F.col(c)).alias(f"{c}__max"))
    agg_exprs.append(F.mean(F.col(c).cast("double")).alias(f"{c}__mean"))
    agg_exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__nulls"))

stats_row = df.agg(*agg_exprs).collect()[0]

print(f"{'Feature':<30s} {'Min':>12s} {'Max':>12s} {'Mean':>12s} {'Nulls':>8s}")
print("-" * 70)
for feat in numeric_eng:
    mn = stats_row[f"{feat}__min"]
    mx = stats_row[f"{feat}__max"]
    avg = stats_row[f"{feat}__mean"]
    nulls = int(stats_row[f"{feat}__nulls"])
    print(f"{feat:<30s} {mn:>12.2f} {mx:>12.2f} {avg:>12.4f} {nulls:>8d}")

total_rows = df.count()
total_cols = len(df.columns)
print(f"\nFinal table: {total_rows:,} rows x {total_cols} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.4 Build Panel Data for Temporal Validation

# COMMAND ----------

customer_ref = df

try:
    panel_exists = spark.table(f"{CATALOG}.{SCHEMA}.ai_02_customer_full_data").count() > 0
except:
    panel_exists = False

if panel_exists:
    print("Found existing panel data (ai_02_customer_full_data) — building panel features...")

    customer_full_raw = spark.table(f"{CATALOG}.{SCHEMA}.ai_02_customer_full_data")
    panel = customer_full_raw

    panel = panel.withColumn("income_log", F.log1p(F.abs(F.col("income"))))

    balance_cols_panel = [c for c in panel.columns if "balance_avg" in c.lower()]
    revenue_cols_panel = [c for c in panel.columns if "revenue_avg" in c.lower()]

    if balance_cols_panel:
        panel = panel.withColumn("total_balance",
            sum([F.coalesce(F.col(c), F.lit(0.0)) for c in balance_cols_panel]))
        panel = panel.withColumn("total_balance_log", F.log1p(F.abs(F.col("total_balance"))))

    if revenue_cols_panel:
        panel = panel.withColumn("total_revenue",
            sum([F.coalesce(F.col(c), F.lit(0.0)) for c in revenue_cols_panel]))
        panel = panel.withColumn("total_revenue_log", F.log1p(F.abs(F.col("total_revenue"))))

    if "digital_services_avg" in panel.columns and "in_person_interaction_avg" in panel.columns:
        panel = panel.withColumn("digital_ratio",
            F.when(
                (F.coalesce(F.col("digital_services_avg"), F.lit(0))
                 + F.coalesce(F.col("in_person_interaction_avg"), F.lit(0))) > 0,
                F.col("digital_services_avg")
                / (F.col("digital_services_avg") + F.col("in_person_interaction_avg"))
            ).otherwise(0.0))

    if "nb_subscribed_products" in panel.columns:
        panel = panel.withColumn("revenue_per_product",
            F.when(F.col("nb_subscribed_products") > 0,
                   F.col("total_revenue") / F.col("nb_subscribed_products"))
            .otherwise(0.0))
        panel = panel.withColumn("product_churn_rate",
            F.when(F.col("nb_subscribed_products") > 0,
                   F.col("nb_terminated_products") / F.col("nb_subscribed_products"))
            .otherwise(0.0))
        panel = panel.withColumn("net_product_growth",
            F.col("nb_subscribed_product_lag") - F.col("nb_terminated_products_lag"))

    panel = panel.withColumn("atm_intensity",
        F.coalesce(F.col("atm_withdrawals_sum_avg"), F.lit(0.0)))
    panel = panel.withColumn("income_quintile", F.ntile(5).over(Window.orderBy("income")))

    panel = panel.withColumn("tenure_months",
        F.round(F.col("account_age") * 12, 0).cast("int"))
    panel = panel.withColumn("age_bracket",
        F.when(F.col("customer_age") < 30, "Young")
         .when(F.col("customer_age") < 50, "Middle")
         .when(F.col("customer_age") < 65, "Senior")
         .otherwise("Elderly"))

    product_count_cols = [c for c in panel.columns if "product_count" in c.lower()]
    if product_count_cols:
        panel = panel.withColumn("product_diversity",
            sum([F.when(F.coalesce(F.col(c), F.lit(0)) > 0, F.lit(1)).otherwise(F.lit(0))
                 for c in product_count_cols]))

    # One-hot encode gender in panel
    distinct_genders = [row[0] for row in panel.select("gender").distinct().collect() if row[0] is not None]
    distinct_genders.sort()
    for g in distinct_genders:
        col_name = f"gender_{g.lower().replace(' ', '_')}"
        if "gender" in panel.columns:
            panel = panel.withColumn(col_name, F.when(F.col("gender") == g, 1).otherwise(0))

    customer_full = panel
    print(f"Panel features: {customer_full.count():,} rows x {len(customer_full.columns)} cols")
else:
    customer_full = None
    print("No panel data found — only reference table will be registered")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.5 Register Feature Tables in Feature Store

# COMMAND ----------

REF_TABLE = f"{CATALOG}.{SCHEMA}.ai_ai_04_customer_features_reference"
PANEL_TABLE = f"{CATALOG}.{SCHEMA}.ai_04_customer_features"

# Register reference feature table
try:
    fe.create_table(
        name=REF_TABLE,
        primary_keys=["customer_id"],
        df=customer_ref,
        description=(
            "Customer segmentation features for the latest reference period. "
            "Includes demographics, product portfolio, financial aggregates, behavioral metrics, "
            "plus engineered features: log transforms, one-hot encoding, derived ratios, composites. "
            "Used as the training dataset for K-Means clustering."
        ),
    )
    print(f"Created feature table: {REF_TABLE}")
except Exception as e:
    if "already exists" in str(e).lower():
        fe.write_table(name=REF_TABLE, df=customer_ref, mode="overwrite")
        print(f"Updated existing feature table: {REF_TABLE}")
    else:
        raise e

print(f"  Rows: {spark.table(REF_TABLE).count():,}")

# Register panel feature table (if historical data exists)
if customer_full is not None:
    try:
        fe.create_table(
            name=PANEL_TABLE,
            primary_keys=["customer_id", "reference_date"],
            df=customer_full,
            description=(
                "Customer segmentation feature table (panel: customer x month). "
                "Same features as reference table but across all historical months. "
                "Used for temporal validation and segment stability analysis."
            ),
        )
        print(f"Created feature table: {PANEL_TABLE}")
    except Exception as e:
        if "already exists" in str(e).lower():
            fe.write_table(name=PANEL_TABLE, df=customer_full, mode="overwrite")
            print(f"Updated existing feature table: {PANEL_TABLE}")
        else:
            raise e

    print(f"  Rows: {spark.table(PANEL_TABLE).count():,}")
else:
    print("Panel table skipped (no historical data available)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4.6 Add Column Descriptions

# COMMAND ----------

COLUMN_DESCRIPTIONS = {
    "customer_id": "Unique customer identifier (primary key)",
    "reference_date": "Monthly snapshot date",
    "gender": "Customer gender (raw)",
    "region": "Geographic region code",
    "income": "Annual income (raw)",
    "tier": "Customer tier (Gold, Silver, Bronze, etc.)",
    "customer_age": "Customer age in years",
    "tenure_months": "Account tenure in months (per doc: months_between)",
    "age_bracket": "Age group: Young (<30), Middle (30-49), Senior (50-64), Elderly (65+)",
    "income_log": "Log-transformed income: log1p(|income|)",
    "total_balance_log": "Log-transformed total balance: log1p(|total_balance|)",
    "total_revenue_log": "Log-transformed total revenue: log1p(|total_revenue|)",
    "gender_f": "One-hot: gender is Female (1/0)",
    "gender_m": "One-hot: gender is Male (1/0)",
    "total_balance": "Sum of all balance amounts (latest snapshot)",
    "total_revenue": "Sum of all revenue amounts (12-month window)",
    "nb_subscribed_products": "Total products ever subscribed",
    "nb_terminated_products": "Total products terminated",
    "nb_active_products": "Currently active products",
    "product_diversity": "COUNT(DISTINCT product_type) where active",
    "digital_ratio": "Digital / (digital + in-person) — channel preference",
    "revenue_per_product": "Total revenue / subscribed products",
    "product_churn_rate": "Terminated / subscribed products — attrition risk",
    "net_product_growth": "Active - terminated products",
    "atm_intensity": "ATM withdrawal volume (from atm_withdrawals_sum_avg)",
    "income_quintile": "Income quintile bucket (1=lowest, 5=highest)",
}

for table in [REF_TABLE, PANEL_TABLE]:
    try:
        existing_cols = {c.name for c in spark.table(table).schema}
        applied = 0
        for col, desc in COLUMN_DESCRIPTIONS.items():
            if col in existing_cols:
                try:
                    safe_desc = desc.replace("'", "''")
                    spark.sql(f"ALTER TABLE {table} ALTER COLUMN `{col}` COMMENT '{safe_desc}'")
                    applied += 1
                except:
                    pass
        print(f"Applied {applied} column descriptions to {table}")
    except:
        print(f"Skipped column descriptions for {table} (table not found)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC All feature engineering was performed in NB03a-03d. This notebook only:
# MAGIC 1. Joined the 4 domain tables
# MAGIC 2. Added one cross-domain feature (`revenue_per_product`)
# MAGIC 3. Registered in Feature Store
# MAGIC
# MAGIC **Next:** NB05 (Model Calibration) — K-Means clustering with Z-score normalization.

# COMMAND ----------

print("=" * 70)
print("  FEATURE JOIN COMPLETE")
print("=" * 70)
print(f"  Reference table: {REF_TABLE}")
print(f"    Rows: {spark.table(REF_TABLE).count():,}")
print(f"    Cols: {len(spark.table(REF_TABLE).columns)}")
if customer_full is not None:
    print(f"  Panel table: {PANEL_TABLE}")
    print(f"    Rows: {spark.table(PANEL_TABLE).count():,}")
    print(f"    Cols: {len(spark.table(PANEL_TABLE).columns)}")
print(f"\n  Browse: AI/ML > Features > catalog: {CATALOG}")
print("=" * 70)
