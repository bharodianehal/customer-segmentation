---
name: banking-c360-feature-engineering
description: "Customer 360 feature engineering standards for banking segmentation. Use when profiling banking customer data, engineering features for clustering or propensity models, building RFM metrics, registering features in Unity Catalog Feature Store, or performing customer value and behavioral segmentation for campaign targeting."
---

# Banking C360 Feature Engineering

Standards and patterns for engineering production-quality features from banking Customer 360 data for segmentation, campaign targeting, and propensity models.

## When to Use This Skill

Use this skill when:
- Profiling banking customer data (demographics, accounts, transactions, loans, digital engagement)
- Engineering features for **customer value segmentation** (RFM, financial profile, relationship depth)
- Engineering features for **behavioral/channel segmentation** (digital engagement, channel mix, product usage)
- Selecting and reducing features (correlation, VIF, variable clustering)
- Registering features in **Unity Catalog Feature Store**
- Preparing feature matrices for K-Means, hierarchical, or DBSCAN clustering
- Building features for campaign targeting (cross-sell, retention, migration)

## Reference Files

| Topic | File | Description |
|-------|------|-------------|
| Feature Catalog | [1-feature-catalog.md](1-feature-catalog.md) | Complete list of 60+ banking features organized by domain with SQL/PySpark code |
| Feature Selection | [2-feature-selection.md](2-feature-selection.md) | Correlation, VIF, VARCLUS, PCA patterns for reducing feature dimensionality |
| Feature Store Patterns | [3-feature-store-patterns.md](3-feature-store-patterns.md) | Unity Catalog Feature Store registration, serving, and lifecycle management |

## Quick Start: Two-Model Segmentation

Banking segmentation requires **two complementary clustering models**:

| Model | Purpose | Key Features |
|-------|---------|-------------|
| **Customer Value** | WHO is valuable, how much | RFM, balance, products, tenure, revenue |
| **Behavioral/Channel** | HOW they interact | Digital engagement, channel mix, transaction patterns |

Together they create a targeting matrix: *"Emerging Affluent + Mobile-First"* = in-app investment education campaign.

### Step 1: Profile the Data

Always start with data profiling before feature engineering:

```python
from pyspark.sql import functions as F

CATALOG = "your_catalog"
SCHEMA = "your_schema"

# Load source tables
customers = spark.table(f"{CATALOG}.{SCHEMA}.customers")
accounts = spark.table(f"{CATALOG}.{SCHEMA}.accounts")
transactions = spark.table(f"{CATALOG}.{SCHEMA}.transactions")
loans = spark.table(f"{CATALOG}.{SCHEMA}.loans")
digital = spark.table(f"{CATALOG}.{SCHEMA}.digital_engagement")

# Profile each table: row counts, null rates, distributions
for name, df in [("customers", customers), ("accounts", accounts),
                  ("transactions", transactions), ("loans", loans),
                  ("digital", digital)]:
    print(f"\n{'='*60}")
    print(f"TABLE: {name} | Rows: {df.count():,} | Columns: {len(df.columns)}")
    # Null analysis
    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show(vertical=True)
    # Numeric distributions
    df.describe().show()
```

### Step 2: Engineer Value Segmentation Features

```python
from pyspark.sql import functions as F, Window
from datetime import date

REFERENCE_DATE = date.today()

# --- RFM Features (12-month primary window with 30d/90d sub-windows for trends) ---
rfm = (
    transactions
    .filter(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 365))
    .groupBy("customer_id")
    .agg(
        F.datediff(F.lit(REFERENCE_DATE), F.max("transaction_date")).alias("recency_days"),
        # 12-month primary RFM metrics
        F.count("transaction_id").alias("frequency_12m"),
        F.sum("amount").alias("monetary_12m"),
        F.avg("amount").alias("avg_txn_amount"),
        F.countDistinct(F.date_trunc("week", "transaction_date")).alias("active_weeks_12m"),
        # 30-day and 90-day sub-windows for trend calculation
        F.sum(F.when(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 30),
                     F.col("amount"))).alias("monetary_30d"),
        F.count(F.when(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 30),
                       F.col("transaction_id"))).alias("frequency_30d"),
        F.sum(F.when(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 90),
                     F.col("amount"))).alias("monetary_90d"),
        F.count(F.when(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 90),
                       F.col("transaction_id"))).alias("frequency_90d"),
    )
    # Trend ratios: recent activity as fraction of 12-month baseline
    .withColumn("frequency_acceleration_30d",
                F.when(F.col("frequency_12m") > 0,
                       F.col("frequency_30d") / (F.col("frequency_12m") / 12))
                .otherwise(0))
    .withColumn("monetary_trend_90d",
                F.when(F.col("monetary_12m") > 0,
                       F.col("monetary_90d") / (F.col("monetary_12m") / 4))
                .otherwise(0))
)

# --- Financial Profile (from accounts) ---
financial = (
    accounts
    .filter(F.col("status") == "active")
    .groupBy("customer_id")
    .agg(
        F.sum("current_balance").alias("total_deposit_balance"),
        F.sum(F.when(F.col("account_type") == "checking", F.col("current_balance"))).alias("checking_balance"),
        F.sum(F.when(F.col("account_type") == "savings", F.col("current_balance"))).alias("savings_balance"),
        F.count("account_id").alias("num_deposit_accounts"),
        F.avg("avg_balance_30d").alias("avg_balance_30d"),
        F.avg("avg_balance_90d").alias("avg_balance_90d"),
    )
)

# --- Lending Profile (from loans) ---
lending = (
    loans
    .filter(F.col("loan_status").isin("current", "delinquent"))
    .groupBy("customer_id")
    .agg(
        F.sum("current_balance").alias("total_loan_balance"),
        F.count("loan_id").alias("num_active_loans"),
        F.avg("interest_rate").alias("avg_loan_rate"),
        F.max(F.when(F.col("loan_type") == "mortgage", 1).otherwise(0)).alias("has_mortgage"),
        F.max(F.when(F.col("loan_type") == "credit_card", 1).otherwise(0)).alias("has_credit_card"),
        F.max(F.when(F.col("delinquency_days") > 0, 1).otherwise(0)).alias("has_delinquency"),
    )
)

# --- Assemble Customer Value Features ---
value_features = (
    customers.select("customer_id", "annual_income", "credit_score",
                     "tenure_months", "customer_segment")
    .join(rfm, "customer_id", "left")
    .join(financial, "customer_id", "left")
    .join(lending, "customer_id", "left")
    .fillna(0)
    .withColumn("total_relationship_value",
                F.col("total_deposit_balance") + F.coalesce(F.col("total_loan_balance"), F.lit(0)))
    .withColumn("deposit_to_loan_ratio",
                F.when(F.col("total_loan_balance") > 0,
                       F.col("total_deposit_balance") / F.col("total_loan_balance"))
                .otherwise(F.lit(999)))
    .withColumn("balance_trend",
                F.col("avg_balance_30d") - F.col("avg_balance_90d"))
    .withColumn("product_count",
                F.col("num_deposit_accounts") + F.coalesce(F.col("num_active_loans"), F.lit(0)))
    .withColumn("frequency_acceleration_30d",
                F.when(F.col("frequency_12m") > 0,
                       F.col("frequency_30d") / (F.col("frequency_12m") / 12))
                .otherwise(0))
)
```

### Step 3: Engineer Behavioral Segmentation Features

```python
# --- Channel Mix (from transactions) ---
channel_mix = (
    transactions
    .filter(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 90))
    .groupBy("customer_id")
    .agg(
        F.sum(F.when(F.col("channel") == "mobile", 1).otherwise(0)).alias("mobile_txns_90d"),
        F.sum(F.when(F.col("channel") == "online", 1).otherwise(0)).alias("online_txns_90d"),
        F.sum(F.when(F.col("channel") == "branch", 1).otherwise(0)).alias("branch_txns_90d"),
        F.sum(F.when(F.col("channel") == "atm", 1).otherwise(0)).alias("atm_txns_90d"),
        F.countDistinct("channel").alias("channel_diversity"),
        F.count("transaction_id").alias("total_txns_90d"),
    )
    .withColumn("digital_txn_ratio",
                (F.col("mobile_txns_90d") + F.col("online_txns_90d")) / F.col("total_txns_90d"))
    .withColumn("branch_dependency_ratio",
                F.col("branch_txns_90d") / F.col("total_txns_90d"))
)

# --- Digital Engagement (from digital_engagement table) ---
# Use directly: mobile_logins_30d, online_logins_30d, digital_engagement_score,
#               preferred_channel, bill_pay_enrolled, direct_deposit_enrolled

# --- Assemble Behavioral Features ---
behavior_features = (
    customers.select("customer_id", "age")
    .join(channel_mix, "customer_id", "left")
    .join(digital.select(
        "customer_id", "mobile_logins_30d", "online_logins_30d",
        "mobile_transactions_30d", "branch_visits_30d", "atm_transactions_30d",
        "digital_engagement_score", "preferred_channel",
        "bill_pay_enrolled", "direct_deposit_enrolled",
        "mobile_app_enrolled", "online_banking_enrolled"
    ), "customer_id", "left")
    .fillna(0)
    .withColumn("digital_adoption_score",
                F.col("mobile_app_enrolled").cast("int") +
                F.col("online_banking_enrolled").cast("int") +
                F.col("bill_pay_enrolled").cast("int") +
                F.col("direct_deposit_enrolled").cast("int"))
    .withColumn("engagement_intensity",
                F.col("mobile_logins_30d") + F.col("online_logins_30d") +
                F.col("branch_visits_30d"))
)
```

### Step 4: Register in Feature Store

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Register Value Features
fe.create_table(
    name=f"{CATALOG}.{SCHEMA}.customer_value_features",
    primary_keys=["customer_id"],
    df=value_features,
    description="Customer value segmentation features: RFM, financial profile, lending, relationship depth"
)

# Register Behavioral Features
fe.create_table(
    name=f"{CATALOG}.{SCHEMA}.customer_behavior_features",
    primary_keys=["customer_id"],
    df=behavior_features,
    description="Customer behavioral segmentation features: channel mix, digital engagement, product usage"
)
```

## Feature Engineering Rules

### Mandatory Standards

1. **Always scale features** before clustering. Use `StandardScaler` (mean=0, std=1) for K-Means. Financial data is heavily right-skewed -- apply `log1p` transform before scaling for balance and monetary features.

2. **Handle nulls explicitly.** Never drop customers with nulls -- fill with 0 for counts/flags, fill with median for continuous variables. Document every fill decision.

3. **Cap outliers.** Winsorize at 1st and 99th percentiles for financial variables. Wealth management customers create extreme right tails that distort K-Means centroids.

4. **Use 12-month lookback** as the primary window for transactional features (captures full seasonal cycles and true customer behavior). Compute 30-day and 90-day sub-windows for recency and trend signals. The 12-month window feeds RFM totals; 30d/90d sub-windows enable trend calculation (e.g., `frequency_30d / frequency_12m` for acceleration).

5. **Derive ratios, not just absolutes.** `digital_txn_ratio` is more informative than `mobile_txns_90d` alone because it normalizes for activity level.

6. **Primary keys must be NOT NULL.** Required for Unity Catalog Feature Store. Always set `customer_id` as the primary key.

### Regulatory Constraints (US Banking)

- **Never use** race, ethnicity, religion, national origin, sex, or marital status as clustering features. These are prohibited under ECOA and Fair Lending.
- **Age** can be used for behavioral segmentation but not for credit decisions.
- **Geographic features** (state, zip) are acceptable for segmentation but require fair lending review if used in credit models. ZIP codes can serve as proxies for race/ethnicity -- always run disparate impact analysis.
- **Income** is acceptable for value segmentation.
- **Proxy discrimination testing**: Even when protected attributes are excluded, features correlated with protected classes (geography, income, surname patterns) can produce disparate impact. Run adverse impact ratio analysis on segment assignments: if any protected group is assigned to low-value segments at a rate >80% higher than the majority group, investigate and document remediation.
- **SR 11-7 (Model Risk Management)**: Segmentation models used for credit marketing, pricing, or allocation decisions must be documented under the SR 11-7 framework. Maintain: (1) model development documentation, (2) independent validation results, (3) ongoing monitoring plan with PSI thresholds, (4) fair lending test results. The segmentation model inventory should be registered with Model Risk Management (MRM).
- **Disparate impact analysis pattern**: After assigning segments, join segment labels to HMDA/CRA data or BISG-estimated demographics and compute selection rates per protected class per segment. Flag any segment where the ratio of minority-to-majority assignment exceeds the 80% adverse impact threshold.

### Feature Naming Convention

Use `{domain}_{metric}_{window}` format:
- `txn_frequency_12m` (transaction domain, frequency metric, 12-month window)
- `txn_frequency_90d` (transaction domain, frequency metric, 90-day sub-window)
- `deposit_balance_avg_30d` (deposit domain, average balance, 30-day window)
- `digital_login_ratio_30d` (digital domain, login ratio, 30-day window)
- `product_count` (no window for point-in-time features)

## Using AI Functions for Feature Enhancement

Use `ai_query()` to create features from unstructured data or to auto-label segments:

```sql
-- Auto-label segments from centroid profiles
SELECT segment_id,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'You are a banking customer intelligence analyst. ',
      'Given this customer cluster centroid: ',
      'avg_balance=$', ROUND(avg_balance), ', ',
      'monthly_txns=', ROUND(avg_txns), ', ',
      'products=', ROUND(avg_products, 1), ', ',
      'tenure_years=', ROUND(avg_tenure, 1), ', ',
      'digital_score=', ROUND(avg_digital), ', ',
      'has_mortgage_pct=', ROUND(mortgage_pct * 100), '%. ',
      'Provide a 3-4 word business segment name and a one-sentence profile description. ',
      'Format: Name: [name] | Description: [description]'
    )
  ) AS segment_label
FROM cluster_centroids
```

```sql
-- Use LLM to suggest additional features from schema
SELECT ai_query(
  'databricks-meta-llama-3-3-70b-instruct',
  CONCAT(
    'You are a senior banking data scientist. Given these C360 source tables: ',
    'customers(customer_id, age, income, credit_score, tenure, segment), ',
    'transactions(customer_id, date, amount, channel, merchant_category), ',
    'accounts(customer_id, type, balance, avg_balance_30d), ',
    'loans(customer_id, type, balance, rate, delinquency_days), ',
    'digital_engagement(customer_id, mobile_logins, online_logins, engagement_score). ',
    'Suggest 10 high-value derived features for customer segmentation ',
    'that are NOT simple aggregations. Focus on behavioral signals, ',
    'trend indicators, and cross-domain ratios. ',
    'For each feature, provide: name, SQL expression, and business rationale.'
  )
) AS feature_suggestions
```

## Common Patterns

### Pattern 1: Elbow + Silhouette Analysis for Optimal K

```python
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

# Log-transform skewed financial features, then scale
import numpy as np
skewed_cols = ["total_deposit_balance", "monetary_12m", "monetary_90d",
               "total_loan_balance", "total_relationship_value", "annual_income"]
for col in skewed_cols:
    feature_pdf[col] = np.log1p(feature_pdf[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_pdf[feature_columns])

# Sweep K=3 to K=8 with MLflow tracking (target range: 4-6)
mlflow.set_experiment(f"/Users/{username}/customer_segmentation")

results = []
for k in range(3, 9):
    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        model = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
        labels = model.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)

        mlflow.log_params({"k": k, "n_init": 20, "features": len(feature_columns)})
        mlflow.log_metrics({
            "silhouette_score": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "inertia": model.inertia_
        })
        mlflow.sklearn.log_model(model, "kmeans_model")
        results.append({"k": k, "silhouette": sil, "ch": ch, "db": db})

# Business constraint: prefer K=4-6. More than 6 segments are hard for marketing
# teams to operationalize (unique campaigns, staffing, tracking) and often lead to
# segments that are statistically distinct but not actionably different.
```

### Pattern 2: Stability Validation (Split-Half + PSI)

```python
from sklearn.metrics import adjusted_rand_score

# Split-half stability test
np.random.seed(42)
idx = np.random.permutation(len(X_scaled))
half = len(idx) // 2

model_a = KMeans(n_clusters=optimal_k, n_init=20, random_state=42).fit(X_scaled[idx[:half]])
model_b = KMeans(n_clusters=optimal_k, n_init=20, random_state=42).fit(X_scaled[idx[half:]])

labels_a = model_a.predict(X_scaled)
labels_b = model_b.predict(X_scaled)

ari = adjusted_rand_score(labels_a, labels_b)
print(f"Adjusted Rand Index: {ari:.4f}")
# ARI > 0.80 = stable model; 0.60-0.80 = moderate; < 0.60 = unstable

# Population Stability Index per feature
def calculate_psi(baseline, current, bins=10):
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct = np.histogram(current, bins=breakpoints)[0] / len(current)
    baseline_pct = np.clip(baseline_pct, 0.001, None)
    current_pct = np.clip(current_pct, 0.001, None)
    return np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

# PSI thresholds: < 0.10 = stable, 0.10-0.25 = investigate, > 0.25 = recalibrate
```

### Pattern 3: Customer Lifetime Value (CLV) Features

Estimate the present value of future customer cash flows for value-based segmentation and prioritization.

```python
from pyspark.sql import functions as F

# CLV estimation features
clv_features = (
    customers.select("customer_id", "tenure_months")
    .join(
        transactions
        .filter(F.col("transaction_date") >= F.date_sub(F.lit(REFERENCE_DATE), 365))
        .groupBy("customer_id")
        .agg(
            F.sum("amount").alias("total_revenue_12m"),
            F.avg("amount").alias("avg_txn_amount_12m"),
            F.countDistinct(F.date_trunc("month", "transaction_date")).alias("active_months_12m"),
        ),
        "customer_id", "left"
    )
    .join(
        accounts
        .filter(F.col("status") == "active")
        .groupBy("customer_id")
        .agg(F.avg("current_balance").alias("avg_balance")),
        "customer_id", "left"
    )
    .fillna(0)
    # Revenue per month (annualized run rate)
    .withColumn("revenue_per_month",
                F.when(F.col("active_months_12m") > 0,
                       F.col("total_revenue_12m") / F.col("active_months_12m"))
                .otherwise(0))
    # Cost-to-serve estimate (simplified: branch-heavy customers cost more)
    # In production, join actual cost data from finance tables
    .withColumn("cost_to_serve_estimate",
                F.lit(15.0)  # base monthly cost per customer
                + F.when(F.col("avg_balance") < 1000, F.lit(5.0)).otherwise(F.lit(0))
                )
    # Net revenue over 12 months
    .withColumn("net_revenue_12m",
                F.col("total_revenue_12m") - (F.col("cost_to_serve_estimate") * 12))
    # Estimated CLV: present value of projected 5-year net revenue
    # Uses simple DCF with 8% annual discount rate, assumes stable revenue
    .withColumn("estimated_clv",
                F.when(F.col("revenue_per_month") > 0,
                       (F.col("revenue_per_month") - F.col("cost_to_serve_estimate"))
                       * 12  # annual net
                       * ((1 - F.pow(F.lit(1.08), F.lit(-5))) / F.lit(0.08))  # 5-year annuity factor
                       ).otherwise(0))
)
```

Key CLV features:
- **estimated_clv**: Present value of future cash flows (5-year horizon, 8% discount rate)
- **revenue_per_month**: Average monthly revenue from transactions
- **cost_to_serve_estimate**: Estimated monthly servicing cost (refine with actual cost data)
- **net_revenue_12m**: Revenue minus estimated cost over trailing 12 months

### Pattern 4: Attrition and Risk Signal Features

Early warning signals for customer attrition, dormancy, and dissatisfaction.

```python
from pyspark.sql import functions as F, Window

# --- Balance decline rate (90-day trend) ---
balance_window = Window.partitionBy("customer_id").orderBy("snapshot_date")
balance_trend = (
    account_snapshots  # daily/weekly balance snapshots table
    .filter(F.col("snapshot_date") >= F.date_sub(F.lit(REFERENCE_DATE), 90))
    .withColumn("prev_balance", F.lag("balance", 30).over(balance_window))
    .groupBy("customer_id")
    .agg(
        F.avg(
            F.when(F.col("prev_balance") > 0,
                   (F.col("balance") - F.col("prev_balance")) / F.col("prev_balance"))
        ).alias("balance_decline_rate_90d"),  # negative = declining
    )
)

# --- Days since last deposit ---
last_deposit = (
    transactions
    .filter(F.col("transaction_type") == "deposit")
    .groupBy("customer_id")
    .agg(
        F.datediff(F.lit(REFERENCE_DATE), F.max("transaction_date")).alias("days_since_last_deposit"),
    )
)

# --- Dormancy flag ---
# Dormant = no debit transactions in 90 days
dormancy = (
    transactions
    .filter(F.col("transaction_type").isin("debit", "withdrawal", "transfer_out"))
    .groupBy("customer_id")
    .agg(F.max("transaction_date").alias("last_active_date"))
    .withColumn("dormancy_flag",
                F.when(F.datediff(F.lit(REFERENCE_DATE), F.col("last_active_date")) > 90, 1)
                .otherwise(0))
)

# --- Complaint count (12 months) ---
complaints = (
    customer_complaints  # CRM complaints table
    .filter(F.col("complaint_date") >= F.date_sub(F.lit(REFERENCE_DATE), 365))
    .groupBy("customer_id")
    .agg(F.count("complaint_id").alias("complaint_count_12m"))
)

# --- Product downgrade flag ---
product_changes = (
    product_change_log  # product upgrade/downgrade events
    .filter(
        (F.col("change_date") >= F.date_sub(F.lit(REFERENCE_DATE), 365)) &
        (F.col("change_type") == "downgrade")
    )
    .groupBy("customer_id")
    .agg(F.lit(1).alias("product_downgrade_flag"))
)

# Assemble attrition features
attrition_features = (
    customers.select("customer_id")
    .join(balance_trend, "customer_id", "left")
    .join(last_deposit, "customer_id", "left")
    .join(dormancy, "customer_id", "left")
    .join(complaints, "customer_id", "left")
    .join(product_changes, "customer_id", "left")
    .fillna(0)
)
```

Key attrition features:
- **balance_decline_rate_90d**: Average rate of balance change over 90 days (negative = risk)
- **days_since_last_deposit**: Days since customer's last incoming deposit
- **dormancy_flag**: 1 if no outgoing transactions in 90 days
- **complaint_count_12m**: Number of complaints filed in trailing 12 months
- **product_downgrade_flag**: 1 if customer downgraded any product in trailing 12 months

### Pattern 5: Life-Stage Features

Infer customer life stage for lifecycle-targeted marketing.

```python
from pyspark.sql import functions as F

life_stage_features = (
    customers.select("customer_id", "age", "annual_income", "tenure_months")
    .join(
        accounts.groupBy("customer_id").agg(
            F.max(F.when(F.col("account_type") == "joint", 1).otherwise(0)).alias("has_joint_account"),
        ),
        "customer_id", "left"
    )
    # --- Age bracket ---
    .withColumn("age_bracket",
                F.when(F.col("age") < 30, "Young Professional")
                .when(F.col("age") < 40, "Growing Family")
                .when(F.col("age") < 55, "Established")
                .when(F.col("age") < 65, "Pre-Retirement")
                .otherwise("Retired"))
    # --- Estimated career stage ---
    .withColumn("estimated_career_stage",
                F.when(F.col("age") < 25, "Entry Level")
                .when((F.col("age") >= 25) & (F.col("age") < 35), "Early Career")
                .when((F.col("age") >= 35) & (F.col("age") < 50), "Mid Career")
                .when((F.col("age") >= 50) & (F.col("age") < 62), "Late Career")
                .otherwise("Post-Career"))
    # --- Household ID for household-level aggregation ---
    # Typically derived from address matching, SSN linkage, or joint account relationships.
    # If not available in source data, use address-based heuristic:
    .join(
        customer_addresses.select("customer_id", "household_id"),
        "customer_id", "left"
    )
    .fillna({"household_id": "UNKNOWN"})
)
```

Key life-stage features:
- **age_bracket**: Young Professional / Growing Family / Established / Pre-Retirement / Retired
- **estimated_career_stage**: Entry Level / Early Career / Mid Career / Late Career / Post-Career
- **household_id**: Household identifier for household-level aggregation (see Pattern 8)

> **Note on age**: Age-based features are permissible for marketing segmentation and behavioral analysis. They must NOT be used as inputs to credit decisioning models (ECOA).

### Pattern 6: Value x Behavioral Cross-Reference Matrix

Create actionable micro-segments by crossing value segments with behavioral segments.

```python
import pandas as pd

# Assume value_segment and behavior_segment are already assigned (from two K-Means models)
cross_matrix = (
    customer_segments  # table with customer_id, value_segment, behavior_segment
    .groupBy("value_segment", "behavior_segment")
    .agg(
        F.count("customer_id").alias("customer_count"),
        F.avg("estimated_clv").alias("avg_clv"),
        F.avg("total_deposit_balance").alias("avg_balance"),
        F.avg("digital_engagement_score").alias("avg_digital_score"),
    )
    .orderBy(F.desc("avg_clv"))
)

# Example cross-reference output:
# +-------------------+---------------------+-------+---------+
# | value_segment     | behavior_segment    | count | avg_clv |
# +-------------------+---------------------+-------+---------+
# | High Value        | Digitally Engaged   | 4,200 | $12,400 |  <-- Retain & upsell via app
# | High Value        | Digitally Disengaged| 1,800 | $11,200 |  <-- Digital migration campaign
# | Emerging Affluent | Mobile-First        | 3,100 | $6,800  |  <-- Investment education
# | Low Value         | Digitally Disengaged| 5,400 | $800    |  <-- Cost-to-serve reduction
# +-------------------+---------------------+-------+---------+

# Assign actionable micro-segment labels
cross_segments = (
    customer_segments
    .withColumn("micro_segment",
                F.concat_ws(" + ", F.col("value_segment"), F.col("behavior_segment")))
    .withColumn("recommended_action",
                F.when((F.col("value_segment") == "High Value") &
                       (F.col("behavior_segment") == "Digitally Disengaged"),
                       "Digital migration: RM outreach to onboard to mobile app")
                .when((F.col("value_segment") == "High Value") &
                      (F.col("behavior_segment") == "Digitally Engaged"),
                      "Retain & deepen: premium in-app features, investment upsell")
                .when((F.col("value_segment") == "Emerging Affluent") &
                      (F.col("behavior_segment").contains("Mobile")),
                      "Grow: in-app investment education, credit card cross-sell")
                .when((F.col("value_segment") == "Low Value") &
                      (F.col("behavior_segment") == "Digitally Disengaged"),
                      "Cost reduction: migrate to digital self-service, reduce branch touches")
                .otherwise("Standard engagement"))
)
```

The cross-reference matrix is the primary deliverable for marketing stakeholders. Each cell in the matrix maps to a distinct campaign strategy.

### Pattern 7: Campaign Activation via Salesforce Marketing Cloud

Push segments to **Salesforce Marketing Cloud (SFMC)** for campaign execution. Use **Hightouch** as the Reverse ETL layer to sync segment assignments from Databricks to SFMC.

```python
# --- Option A: Hightouch Reverse ETL (recommended) ---
# 1. Create a model in Hightouch pointing to the segment table:
#    SELECT customer_id, micro_segment, recommended_action, email, phone
#    FROM {CATALOG}.{SCHEMA}.customer_micro_segments
#    WHERE micro_segment IS NOT NULL
#
# 2. Configure Hightouch sync to Salesforce Marketing Cloud:
#    - Destination: SFMC Data Extension
#    - Match key: customer_id -> ContactKey
#    - Sync mode: Upsert
#    - Schedule: Daily after segmentation pipeline completes
#
# 3. In SFMC, create Journey Builder entries triggered by Data Extension updates.

# --- Option B: Direct SFMC API push (for smaller volumes) ---
# Export segment assignments for SFMC ingestion
sfmc_export = (
    customer_micro_segments
    .select(
        F.col("customer_id").alias("ContactKey"),
        "micro_segment",
        "value_segment",
        "behavior_segment",
        "recommended_action",
        "estimated_clv",
        "email_address",
    )
    .filter(F.col("micro_segment").isNotNull())
)

# Write to a Unity Catalog table that Hightouch or SFMC connector reads
sfmc_export.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.sfmc_segment_sync"
)
```

**Integration architecture**: Databricks (feature engineering + clustering) -> Unity Catalog table -> **Hightouch** (Reverse ETL) -> **Salesforce Marketing Cloud** (campaign execution + journey orchestration). Hightouch provides schedule-based or event-triggered sync with built-in data diff (only changed records are pushed).

### Pattern 8: Household-Level Aggregation

Aggregate customer-level features up to `household_id` for household-level segmentation and marketing (e.g., household-level direct mail, combined wallet share analysis).

```python
from pyspark.sql import functions as F

# Aggregate customer features to household level
household_features = (
    customer_features  # customer-level feature table with household_id
    .filter(F.col("household_id") != "UNKNOWN")
    .groupBy("household_id")
    .agg(
        # Household composition
        F.count("customer_id").alias("household_member_count"),
        F.collect_set("age_bracket").alias("age_brackets_in_household"),
        # Household financial profile (sum across members)
        F.sum("total_deposit_balance").alias("hh_total_deposit_balance"),
        F.sum("total_loan_balance").alias("hh_total_loan_balance"),
        F.sum("total_relationship_value").alias("hh_total_relationship_value"),
        F.sum("estimated_clv").alias("hh_estimated_clv"),
        F.sum("net_revenue_12m").alias("hh_net_revenue_12m"),
        # Household product breadth (union across members)
        F.max("has_mortgage").alias("hh_has_mortgage"),
        F.max("has_credit_card").alias("hh_has_credit_card"),
        F.sum("product_count").alias("hh_product_count"),
        # Household digital engagement (max engagement across members)
        F.max("digital_engagement_score").alias("hh_max_digital_score"),
        F.avg("digital_engagement_score").alias("hh_avg_digital_score"),
        # Household risk signals
        F.max("dormancy_flag").alias("hh_any_dormant_member"),
        F.sum("complaint_count_12m").alias("hh_complaint_count_12m"),
        F.min("balance_decline_rate_90d").alias("hh_worst_balance_trend"),
    )
    .withColumn("hh_wallet_share_estimate",
                F.when(F.col("hh_total_relationship_value") > 0,
                       F.col("hh_total_deposit_balance") / F.col("hh_total_relationship_value"))
                .otherwise(0))
)

# Register household features in Feature Store
fe.create_table(
    name=f"{CATALOG}.{SCHEMA}.household_features",
    primary_keys=["household_id"],
    df=household_features,
    description="Household-level aggregated features for household segmentation and wallet share analysis"
)
```

Household aggregation enables:
- **Wallet share analysis**: Total household relationship vs. estimated total addressable wallet
- **Household lifecycle marketing**: Direct mail campaigns addressed to households, not individuals
- **Cross-member cross-sell**: If one member has a mortgage but no one has HELOC, target the household

## Common Issues

| Issue | Solution |
|-------|----------|
| **K-Means dominated by high-balance outliers** | Apply `log1p` transform to balance/monetary features before scaling. Winsorize at 99th percentile. |
| **Too many correlated features** | Run VIF analysis (remove VIF > 5) or use VARCLUS-style variable clustering. See [2-feature-selection.md](2-feature-selection.md). |
| **Segments not business-interpretable** | Reduce to fewer, broader features. Use PCA on correlated groups. Business rule: if you can't name a segment in 3 words, there are too many features. |
| **Cluster sizes extremely uneven** | Check for outlier micro-clusters. Consider removing extreme outliers or using DBSCAN first to isolate them, then K-Means on the core population. |
| **Feature Store write fails on NULLs** | Primary key columns must be NOT NULL. Use `.fillna(0)` or `.na.drop(subset=["customer_id"])` before writing. |
| **Silhouette score is low (< 0.25)** | Customer data often has overlapping segments -- silhouette > 0.20 is acceptable for banking. Focus on business interpretability over statistical purity. |
| **PSI > 0.25 after one month** | Likely seasonal effect (payroll cycles, tax season). Extend baseline window to 3+ months before triggering recalibration. |
