# Feature Store Patterns for Banking Segmentation

How to register, manage, and serve banking C360 features using Databricks Unity Catalog Feature Store.

## Why Feature Store for Segmentation

1. **Consistency:** Same features used for training and scoring -- eliminates train/serve skew
2. **Reusability:** RFM features built for segmentation can be reused for churn prediction, cross-sell models
3. **Governance:** Unity Catalog tracks lineage (which models use which features), access control, and audit
4. **Discoverability:** Other data scientists can search for and reuse your banking features

## Pattern 1: Create Feature Tables

### Value Segmentation Features

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Register the value features table
fe.create_table(
    name="catalog.schema.customer_value_features",
    primary_keys=["customer_id"],
    df=value_features_df,
    description=(
        "Customer value segmentation features for regional bank C360. "
        "Includes RFM metrics (90d), financial profile (deposits, loans, balances), "
        "relationship depth, and trend indicators. "
        "Refresh cadence: monthly. Source tables: customers, accounts, transactions, loans."
    ),
    tags={"domain": "customer_segmentation", "model": "value", "refresh": "monthly"}
)

# Register the behavioral features table
fe.create_table(
    name="catalog.schema.customer_behavior_features",
    primary_keys=["customer_id"],
    df=behavior_features_df,
    description=(
        "Customer behavioral segmentation features for regional bank C360. "
        "Includes digital engagement, channel mix, product usage patterns, "
        "and digital adoption scores. "
        "Refresh cadence: monthly. Source tables: transactions, digital_engagement, products."
    ),
    tags={"domain": "customer_segmentation", "model": "behavioral", "refresh": "monthly"}
)
```

### Updating Features (Monthly Refresh)

```python
# Upsert with mode='merge' for incremental updates
fe.write_table(
    name="catalog.schema.customer_value_features",
    df=updated_value_features_df,
    mode="merge"  # Insert new customers, update existing
)
```

## Pattern 2: Create Training Sets with Feature Lookups

When training the clustering model, use Feature Lookups to ensure consistency:

```python
from databricks.feature_engineering import FeatureLookup

# Define features to look up for value segmentation model
value_lookups = [
    FeatureLookup(
        table_name="catalog.schema.customer_value_features",
        feature_names=[
            "recency_days", "frequency_90d", "monetary_90d",
            "total_deposit_balance", "savings_balance",
            "total_loan_balance", "product_count", "tenure_months",
            "credit_score", "annual_income", "balance_trend_30d",
            "deposit_to_loan_ratio", "has_mortgage", "has_credit_card"
        ],
        lookup_key="customer_id"
    )
]

# Create training set
# base_df just needs customer_id (the lookup key)
base_df = spark.table("catalog.schema.customers").select("customer_id")

training_set = fe.create_training_set(
    df=base_df,
    feature_lookups=value_lookups,
    label=None,  # Unsupervised -- no label for clustering
    exclude_columns=[]
)

training_pdf = training_set.load_df().toPandas()
```

## Pattern 3: Log Model with Feature Store Lineage

```python
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(training_pdf[feature_columns])
model = KMeans(n_clusters=5, n_init=20, random_state=42)
model.fit(X_scaled)

# Log model with Feature Store lineage
# This links the model to its feature tables in Unity Catalog
with mlflow.start_run(run_name="customer_value_segmentation_v1"):
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="catalog.schema.customer_value_kmeans"
    )
    mlflow.log_params({"n_clusters": 5, "features": feature_columns})
    mlflow.log_metric("silhouette_score", sil_score)
```

## Pattern 4: Batch Scoring with Automatic Feature Lookup

When scoring new or updated customers, the Feature Store automatically retrieves the latest features:

```python
# Score all customers -- Feature Store auto-fetches features
scored_df = fe.score_batch(
    model_uri="models:/catalog.schema.customer_value_kmeans/1",
    df=spark.table("catalog.schema.customers").select("customer_id")
)

# scored_df now has customer_id + prediction (segment assignment)
scored_df.write.mode("overwrite").saveAsTable("catalog.schema.customer_value_segments")
```

## Pattern 5: Feature Serving for Real-Time Scoring

For scoring a single customer in real-time (e.g., when they call the contact center):

```python
from databricks.feature_engineering import FeatureFunction

# Create a FeatureSpec combining lookups
feature_spec = fe.create_feature_spec(
    name="catalog.schema.customer_value_feature_spec",
    features=value_lookups
)

# Deploy as a serving endpoint (via UI or SDK)
# The endpoint auto-retrieves features from online tables by customer_id
# Client sends: {"customer_id": "C0000123"}
# Endpoint returns: segment assignment + feature values
```

## Pattern 6: Monitor Feature Drift

Use Lakehouse Monitoring to detect when features drift beyond PSI thresholds:

```sql
-- Create a monitor on the feature table
CREATE OR REPLACE MONITOR catalog.schema.customer_value_features
WITH (
  PROFILE_TYPE = TimeSeries,
  TIMESTAMP_COL = "feature_computed_date",
  GRANULARITIES = ("1 month"),
  BASELINE_TABLE = "catalog.schema.customer_value_features_baseline"
);

-- Query drift metrics
SELECT column_name, psi, js_divergence, chi_squared_test
FROM catalog.schema.customer_value_features_profile_metrics
WHERE window_start = current_date() - INTERVAL 30 DAYS
  AND psi > 0.10
ORDER BY psi DESC;
```

## Feature Table Schema Reference

### customer_value_features

| Column | Type | Description |
|--------|------|-------------|
| customer_id | STRING NOT NULL (PK) | Unique customer identifier |
| recency_days | INT | Days since last transaction |
| frequency_90d | INT | Transaction count in 90 days |
| monetary_90d | DOUBLE | Total spend in 90 days |
| avg_txn_amount | DOUBLE | Average transaction amount |
| total_deposit_balance | DOUBLE | Sum of all deposit balances |
| checking_balance | DOUBLE | Checking account balance |
| savings_balance | DOUBLE | Savings account balance |
| total_loan_balance | DOUBLE | Sum of all loan balances |
| num_active_loans | INT | Count of active loans |
| has_mortgage | INT | 1 if has active mortgage |
| has_credit_card | INT | 1 if has active credit card |
| credit_score | INT | Current FICO score |
| annual_income | INT | Annual income |
| tenure_months | INT | Months as customer |
| product_count | INT | Total active products |
| balance_trend_30d | DOUBLE | 30d avg - 90d avg balance |
| deposit_to_loan_ratio | DOUBLE | Deposits / Loans (capped at 999) |
| total_relationship_value | DOUBLE | Deposits + Loans |

### customer_behavior_features

| Column | Type | Description |
|--------|------|-------------|
| customer_id | STRING NOT NULL (PK) | Unique customer identifier |
| digital_engagement_score | INT | Composite digital score (0-100) |
| digital_txn_ratio | DOUBLE | % of transactions via digital channels |
| branch_dependency_ratio | DOUBLE | % of transactions via branch |
| mobile_logins_30d | INT | Mobile app logins in 30 days |
| online_logins_30d | INT | Online banking logins in 30 days |
| channel_diversity | INT | Count of distinct channels used |
| digital_adoption_score | INT | Count of digital features adopted (0-4) |
| mobile_to_total_login_ratio | DOUBLE | Mobile / (Mobile + Online) logins |
| bill_pay_enrolled | INT | 1 if enrolled in bill pay |
| direct_deposit_enrolled | INT | 1 if has direct deposit |
| engagement_intensity | INT | Total logins + branch visits (30d) |
| product_count | INT | Total active products |
| frequency_90d | INT | Transaction count in 90 days |
