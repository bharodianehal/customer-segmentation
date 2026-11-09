---
name: banking-segmentation
description: "Customer segmentation for banking: EDA, feature engineering, and K-Means modeling skills. Use when exploring banking customer data, engineering features from balances/revenues/products/behavioral data, building clustering models, or validating segment stability."
---

# Banking Customer Segmentation Skills

Practical skills for the banking customer segmentation pipeline. Covers three domains: exploratory data analysis, feature engineering, and model development with validation.

## Data Model

The pipeline operates on 7 source tables in the `customer_segmentation.banking` schema:

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `customers` | customer_id, gender, income, tier, birth_date, creation_date, region | Static customer demographics |
| `balances` | customer_id, product_id, date, balance | Monthly product balances |
| `revenues` | customer_id, product_id, date, revenue | Monthly product revenues |
| `products` | product_id, product, product_type | Product reference (4 types: Deposit Account, Secured Lending, Securities, Unsecured Lending) |
| `product_holdings` | customer_id, product_id, product, product_type, start_date, end_date | Product subscription/termination history |
| `additional_information` | customer_id, date, in_person_interaction, atm_withdrawals, digital_services, fx_services, ... | Monthly behavioral activity |
| `dates` | date | Available reference dates |

## Skill 1: Exploratory Data Analysis (EDA)

### When to Use
- First look at customer data before feature engineering
- Understanding distributions, outliers, and relationships
- Identifying data quality issues
- Presenting findings to stakeholders

### Pattern: Comprehensive EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

pdf = spark.table(f"{CATALOG}.{SCHEMA}.03_customer_features_reference").toPandas()

# 1. Shape and types
print(f"Shape: {pdf.shape[0]:,} rows x {pdf.shape[1]} columns")
print(f"Memory: {pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# 2. Null audit
null_pct = (pdf.isnull().sum() / len(pdf) * 100).round(2)
print(null_pct[null_pct > 0].sort_values(ascending=False).to_string())

# 3. Descriptive stats with skewness
desc = pdf.describe(include=[np.number]).T
desc["skew"] = pdf.select_dtypes(include=[np.number]).skew()
print(desc.round(2).to_string())
```

### Pattern: Distribution Visualization

```python
def plot_distribution(pdf, col, title, log_transform=False):
    """Plot histogram with mean/median lines, optionally with log transform."""
    fig, axes = plt.subplots(1, 2 if log_transform else 1, figsize=(14, 5))
    ax = axes[0] if log_transform else axes

    data = pdf[col].dropna()
    ax.hist(data, bins=50, edgecolor="white", alpha=0.8)
    ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():,.0f}")
    ax.axvline(data.median(), color="blue", linestyle="--", label=f"Median: {data.median():,.0f}")
    ax.set_title(f"{title} — Raw", fontsize=14, fontweight="bold")
    ax.legend()

    if log_transform:
        log_data = np.log1p(np.abs(data))
        axes[1].hist(log_data, bins=50, edgecolor="white", alpha=0.8, color="orange")
        axes[1].set_title(f"{title} — Log Transformed", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()
    print(f"  Skewness: {data.skew():.2f} | Kurtosis: {data.kurtosis():.2f}")

# Use for monetary features
plot_distribution(pdf, "income", "Income Distribution", log_transform=True)
```

### Pattern: Correlation Analysis

```python
def plot_correlation_matrix(pdf, feature_cols, title="Feature Correlations"):
    """Plot masked upper-triangle correlation heatmap."""
    corr = pdf[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print top correlated pairs
    pairs = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            pairs.append((corr.columns[i], corr.columns[j], abs(corr.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("Top 10 Correlated Pairs:")
    for c1, c2, r in pairs[:10]:
        print(f"  {c1:40s} <-> {c2:40s}  r={r:.3f}")
```

### Pattern: Segment Profile Comparison

```python
def plot_segment_comparison(pdf, group_col, metric_cols, title):
    """Grouped bar chart comparing metrics across segments."""
    agg = pdf.groupby(group_col)[metric_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    agg.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Average Value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
```

## Skill 2: Feature Engineering

### When to Use
- Computing derived features from raw banking data
- Preparing features for clustering (log transforms, scaling)
- Registering features in Unity Catalog Feature Store

### Core Principle: Functions, Not Scripts

Every feature transformation should be a **named Python function** that takes a DataFrame and returns a DataFrame with new columns appended. This makes features testable, composable, and self-documenting.

### Pattern: Demographic Features

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def compute_demographic_features(df):
    """Derive demographic features from raw customer attributes.

    New columns: income_log, income_quintile, age_bracket, tenure_years
    """
    return (df
        .withColumn("income_log", F.log1p(F.abs(F.col("income"))))
        .withColumn("income_quintile", F.ntile(5).over(Window.orderBy("income")))
        .withColumn("age_bracket",
            F.when(F.col("customer_age") < 30, "Young")
             .when(F.col("customer_age") < 50, "Middle")
             .when(F.col("customer_age") < 65, "Senior")
             .otherwise("Elderly"))
        .withColumn("tenure_years", F.round(F.col("account_age"), 0).cast("int"))
    )
```

### Pattern: Financial Features

```python
def compute_financial_features(df):
    """Derive financial features from balance and revenue columns.

    Dynamically finds balance/revenue columns by naming convention.
    New columns: total_balance, total_revenue, total_balance_log, total_revenue_log, revenue_per_product
    """
    balance_cols = [c for c in df.columns if "balance_avg" in c.lower()]
    revenue_cols = [c for c in df.columns if "revenue_avg" in c.lower()]

    df = df.withColumn("total_balance",
        sum([F.coalesce(F.col(c), F.lit(0.0)) for c in balance_cols]))
    df = df.withColumn("total_revenue",
        sum([F.coalesce(F.col(c), F.lit(0.0)) for c in revenue_cols]))
    df = df.withColumn("total_balance_log", F.log1p(F.abs(F.col("total_balance"))))
    df = df.withColumn("total_revenue_log", F.log1p(F.abs(F.col("total_revenue"))))
    df = df.withColumn("revenue_per_product",
        F.when(F.col("nb_subscribed_products") > 0,
               F.col("total_revenue") / F.col("nb_subscribed_products"))
        .otherwise(0.0))
    return df
```

### Pattern: Behavioral Features

```python
def compute_behavioral_features(df):
    """Derive behavioral features from channel usage data.

    New columns: digital_ratio, atm_intensity
    """
    digital = F.coalesce(F.col("digital_services_avg"), F.lit(0.0))
    in_person = F.coalesce(F.col("in_person_interaction_avg"), F.lit(0.0))

    df = df.withColumn("digital_ratio",
        F.when((digital + in_person) > 0, digital / (digital + in_person))
         .otherwise(0.0))
    df = df.withColumn("atm_intensity",
        F.coalesce(F.col("atm_withdrawals_sum_avg"), F.lit(0.0)))
    return df
```

### Pattern: Product Portfolio Features

```python
def compute_product_features(df):
    """Derive product portfolio features.

    New columns: product_churn_rate, net_product_growth, product_type_breadth
    """
    df = df.withColumn("product_churn_rate",
        F.when(F.col("nb_subscribed_products") > 0,
               F.col("nb_terminated_products") / F.col("nb_subscribed_products"))
        .otherwise(0.0))
    df = df.withColumn("net_product_growth",
        F.col("nb_subscribed_product_lag") - F.col("nb_terminated_products_lag"))

    product_count_cols = [c for c in df.columns if "product_count" in c.lower()]
    df = df.withColumn("product_type_breadth",
        sum([F.when(F.coalesce(F.col(c), F.lit(0)) > 0, F.lit(1)).otherwise(F.lit(0))
             for c in product_count_cols]))
    return df
```

### Pattern: Apply All Feature Functions

```python
# Apply all feature engineering in sequence
feature_functions = [
    compute_demographic_features,
    compute_financial_features,
    compute_behavioral_features,
    compute_product_features,
]

for fn in feature_functions:
    customer_full = fn(customer_full)
    customer_ref = fn(customer_ref)
```

### Pattern: Feature Store Registration

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")

# Drop and recreate for clean registration
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.03_customer_features")

fe.create_table(
    name=f"{CATALOG}.{SCHEMA}.03_customer_features",
    primary_keys=["customer_id", "reference_date"],
    df=customer_full,
    description="Customer segmentation features (panel: customer x month). Includes demographic, financial, behavioral, and product portfolio features."
)
```

### Feature Engineering Rules

1. **Log-transform monetary features** before clustering — income, balances, revenues have log-normal distributions
2. **StandardScaler (AVGSTD)** for all numeric features before K-Means
3. **Fill nulls with 0** for counts/flags, never drop customers
4. **Derive ratios** (digital_ratio, product_churn_rate) — more informative than raw counts
5. **Dynamic column discovery** — use patterns like `[c for c in df.columns if "balance_avg" in c]` instead of hardcoding column lists

## Skill 3: Modeling (Development + Validation)

### When to Use
- Running K-Means clustering with elbow/silhouette analysis
- Profiling and auto-labeling clusters
- Validating segment stability over time
- Building transition matrices

### Pattern: Feature Preparation for Clustering

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_clustering_features(df, feature_cols, log_cols):
    """Prepare features: fill nulls, log-transform skewed columns, standardize.

    Args:
        df: pandas DataFrame
        feature_cols: list of numeric column names
        log_cols: subset of feature_cols to log-transform

    Returns:
        X_scaled: numpy array of standardized features
        scaler: fitted StandardScaler (for scoring historical data)
    """
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    X_log = X.copy()
    for idx in [feature_cols.index(c) for c in log_cols]:
        X_log[:, idx] = np.log1p(np.abs(X_log[:, idx]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled, scaler
```

### Pattern: Elbow & Silhouette Analysis

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

K_RANGE = range(3, 9)
results = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=1337)
    labels = km.fit_predict(X_scaled)
    results.append({
        "k": int(k),
        "inertia": float(km.inertia_),
        "silhouette": float(silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
    })

# Visualize all 4 metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [("inertia", "Inertia (Elbow)"), ("silhouette", "Silhouette Score"),
           ("calinski_harabasz", "Calinski-Harabasz"), ("davies_bouldin", "Davies-Bouldin")]
for ax, (metric, title) in zip(axes.flat, metrics):
    values = [r[metric] for r in results]
    ax.plot([r["k"] for r in results], values, "o-", linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel(metric.replace("_", " ").title())
plt.suptitle("K-Means Cluster Evaluation Metrics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

**Metric interpretation:**
- **Inertia**: Look for the "elbow" (diminishing returns)
- **Silhouette**: Higher is better (>0.20 is acceptable for banking data)
- **Calinski-Harabasz**: Higher is better (ratio of between/within cluster variance)
- **Davies-Bouldin**: Lower is better (avg similarity between clusters)
- **Business constraint**: K=4-6. More than 6 segments are hard for marketing to operationalize.

### Pattern: Cluster Profiling Heatmap

```python
def plot_cluster_heatmap(profiles, feature_cols, title="Cluster Feature Profiles"):
    """Heatmap of standardized feature means across clusters."""
    from sklearn.preprocessing import StandardScaler

    plot_data = profiles[feature_cols].copy()
    scaler = StandardScaler()
    plot_data_scaled = pd.DataFrame(
        scaler.fit_transform(plot_data),
        index=plot_data.index,
        columns=plot_data.columns
    )

    fig, ax = plt.subplots(figsize=(16, max(6, len(feature_cols) * 0.35)))
    sns.heatmap(plot_data_scaled.T, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()
```

### Pattern: Auto-Labeling Segments

```python
def label_cluster(row, profiles):
    """Rule-based auto-labeling based on cluster centroid characteristics."""
    high_income = row["income"] > profiles["income"].quantile(0.75)
    high_digital = row.get("digital_services_avg", 0) > profiles.get("digital_services_avg", pd.Series([0])).median()
    many_products = row["nb_subscribed_products"] > profiles["nb_subscribed_products"].median()
    low_products = row["nb_subscribed_products"] < profiles["nb_subscribed_products"].quantile(0.25)

    balance_cols = [c for c in profiles.columns if "balance_avg" in c.lower()]
    total_bal = sum(row.get(c, 0) for c in balance_cols)
    median_bal = sum(profiles[c].median() for c in balance_cols)
    high_balance = total_bal > median_bal * 1.5

    if high_income and high_balance and many_products:
        return "Premium High-Value"
    elif high_digital and many_products:
        return "Digital-First Engaged"
    elif low_products and row["income"] < profiles["income"].quantile(0.25):
        return "Low Engagement At-Risk"
    elif many_products and not high_income:
        return "Multi-Product Middle Market"
    else:
        return "Mainstream Active"
```

### Pattern: Transition Matrix & Stability

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Build transition analysis
w = Window.partitionBy("customer_id").orderBy("reference_date")

transitions = (
    all_scored
    .select("customer_id", "reference_date", "cluster_labels")
    .withColumn("prev_cluster_1m", F.lag("cluster_labels", 1).over(w))
    .withColumn("prev_cluster_3m", F.lag("cluster_labels", 3).over(w))
    .filter(F.col("prev_cluster_1m").isNotNull())
)

# 1-month transition matrix with probabilities
trans_matrix = (
    transitions.groupBy("prev_cluster_1m", "cluster_labels")
    .agg(F.count("*").alias("count"))
    .withColumn("total_from", F.sum("count").over(Window.partitionBy("prev_cluster_1m")))
    .withColumn("probability", F.round(F.col("count") / F.col("total_from"), 4))
)

# Retention rates = diagonal of transition matrix
retention = trans_matrix.filter(F.col("prev_cluster_1m") == F.col("cluster_labels"))
```

**Stability thresholds:**
- 1-month retention > 85% = stable segment
- 1-month retention 70-85% = moderate (review features)
- 1-month retention < 70% = volatile (consider merging or re-engineering features)

### Pattern: Transition Heatmap

```python
def plot_transition_heatmap(trans_pdf, title="Segment Transition Probabilities"):
    """Heatmap showing from -> to transition probabilities."""
    pivot = trans_pdf.pivot_table(
        values="probability", index="from_segment", columns="to_segment", fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("To Segment")
    ax.set_ylabel("From Segment")
    plt.tight_layout()
    plt.show()
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `cache()` fails on serverless | Use `createOrReplaceTempView()` + `spark.table()` |
| `pyspark.ml` not available on serverless | Use `sklearn` with pandas (toPandas/createDataFrame) |
| `dbutils.library.restartPython()` in markdown cell | Must be a separate Python cell |
| Feature Store write fails on NULLs | Primary key columns must be NOT NULL |
| Silhouette score < 0.25 | Normal for banking data — focus on business interpretability |
| Uneven cluster sizes | Check for outlier micro-clusters, consider log-transforms |
| `seaborn-v0_8-whitegrid` not found | Use `seaborn-whitegrid` or `sns.set_theme(style="whitegrid")` |
