# Databricks notebook source

# MAGIC %md
# MAGIC # Customer Segmentation Workshop — AnalyticSoul Series (Lessons 3.3–3.6)
# MAGIC
# MAGIC Welcome to this comprehensive, tutorial-style notebook that walks through the **full customer segmentation pipeline** following the AnalyticSoul Medium tutorial approach, adapted for our banking demo dataset.
# MAGIC
# MAGIC ### What We'll Cover
# MAGIC | Lesson | Topic | Cells |
# MAGIC |--------|-------|-------|
# MAGIC | **3.3** | Feature Engineering — encoding, correlation, selection, standardization | 1–10 |
# MAGIC | **3.4** | K-Means Customer Segmentation — elbow, silhouette, model training | 11–20 |
# MAGIC | **3.5** | Interpreting Results — profiles, radar charts, naming, business insights | 21–35 |
# MAGIC | **3.6** | Dimensionality Reduction with PCA — variance, 2D/3D viz, loadings | 36–45 |
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - Customer features table: `main.banking.04_customer_features_reference`
# MAGIC - Libraries: scikit-learn, seaborn, matplotlib, plotly, mlflow
# MAGIC
# MAGIC > **Tip:** This notebook is designed as a live workshop. Run each cell in order and read the markdown explanations to understand the *why* behind every step.

# COMMAND ----------

# MAGIC %pip install seaborn plotly scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # LESSON 3.3 — Feature Engineering
# MAGIC
# MAGIC Before we can cluster customers, we need to prepare the raw features. This lesson covers:
# MAGIC 1. Loading the feature table
# MAGIC 2. Encoding categorical variables so K-Means can use them
# MAGIC 3. Correlation analysis to understand feature relationships
# MAGIC 4. Feature selection to avoid redundancy
# MAGIC 5. Standardization so all features are on the same scale

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3.1 — Load Customer Features
# MAGIC
# MAGIC We start by reading the pre-computed customer features table into a Pandas DataFrame. This table was built in earlier notebooks and contains one row per customer with dozens of engineered features — demographics, product holdings, transaction behavior, and more.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

CATALOG = "main"
SCHEMA = "banking"
FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.04_customer_features_reference"

# Load into Pandas
df = spark.table(FEATURES_TABLE).toPandas()
print(f"Loaded {len(df):,} customers with {len(df.columns)} features")
print(f"\nColumns:\n{list(df.columns)}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3.2 — Categorical Encoding
# MAGIC
# MAGIC K-Means operates on **numeric distances**, so categorical columns must be converted to numbers first. We'll use **ordinal encoding** — assigning integers that respect the natural ordering of each category.
# MAGIC
# MAGIC | Column | Encoding Strategy |
# MAGIC |--------|------------------|
# MAGIC | `gender` | Binary: Male → 1, Female → 0 |
# MAGIC | `tier` | Ordinal: GOLD → 3, SILVER → 2, BRONZE → 1 |
# MAGIC | `age_bracket` | Ordinal: Young → 1, Middle → 2, Senior → 3, Elderly → 4 |

# COMMAND ----------

# Show original values before encoding
print("=== BEFORE ENCODING ===")
for col in ["gender", "tier", "age_bracket"]:
    if col in df.columns:
        print(f"\n{col}: {df[col].unique()}")

# --- Gender ---
if "gender" in df.columns:
    df["gender_encoded"] = df["gender"].replace({"Male": 1, "M": 1, "Female": 0, "F": 0})
    print(f"\ngender_encoded mapping: Male/M → 1, Female/F → 0")

# --- Tier ---
if "tier" in df.columns:
    tier_map = {"1.GOLD": 3, "GOLD": 3, "2.SILVER": 2, "SILVER": 2, "3.BRONZE": 1, "BRONZE": 1}
    df["tier_encoded"] = df["tier"].map(tier_map).fillna(1).astype(int)
    print(f"tier_encoded mapping: {tier_map}")

# --- Age Bracket ---
if "age_bracket" in df.columns:
    age_map = {"Young": 1, "Middle": 2, "Senior": 3, "Elderly": 4}
    df["age_bracket_encoded"] = df["age_bracket"].map(age_map).fillna(2).astype(int)
    print(f"age_bracket_encoded mapping: {age_map}")

print("\n=== AFTER ENCODING ===")
encoded_cols = [c for c in ["gender_encoded", "tier_encoded", "age_bracket_encoded"] if c in df.columns]
if encoded_cols:
    display(df[encoded_cols].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3.3 — Correlation Analysis
# MAGIC
# MAGIC Before selecting features, we need to understand **how they relate to each other**. Highly correlated features carry redundant information and can bias the clustering — K-Means would effectively "double-count" that dimension.
# MAGIC
# MAGIC We'll compute the full pairwise correlation matrix and visualize it as a heatmap. Look for:
# MAGIC - **Strong positive correlations (> 0.7)**: These features move together — keep only one.
# MAGIC - **Strong negative correlations (< -0.7)**: Inversely related — may also be redundant.
# MAGIC - **Near-zero correlations**: Independent features — good candidates for clustering.

# COMMAND ----------

# Select all numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove ID-like columns
exclude_patterns = ["customer_id", "id", "_key", "_pk"]
numeric_cols = [c for c in numeric_cols if not any(p in c.lower() for p in exclude_patterns)]

corr_matrix = df[numeric_cols].corr()

# Plot correlation heatmap — mask upper triangle for cleaner look
fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 7},
    linewidths=0.5,
    square=True,
    ax=ax,
)
ax.set_title("Feature Correlation Matrix — Banking Customer Features", fontsize=16, fontweight="bold", pad=20)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Insights
# MAGIC
# MAGIC From the heatmap above, we can observe several key patterns:
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|------------|
# MAGIC | `income_log` and `total_balance_log` may be moderately correlated | Wealthier customers tend to hold more — but the relationship isn't perfect, so both add value |
# MAGIC | `nb_active_products` and `product_diversity` often correlate | Customers with more products naturally have higher diversity — we'll keep both but watch the coefficients |
# MAGIC | `digital_ratio` is typically independent of income | Digital adoption cuts across wealth levels — a genuinely different dimension |
# MAGIC | `atm_intensity` tends to be inversely related to `digital_ratio` | Makes sense: heavy ATM users are less digital |
# MAGIC
# MAGIC We'll use these insights to pick a **non-redundant feature set** next.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3.4 — Feature Selection
# MAGIC
# MAGIC Based on our correlation analysis, we select **10-12 features** that capture different dimensions of customer behavior while minimizing redundancy. The goal is to cover:
# MAGIC
# MAGIC | Dimension | Features |
# MAGIC |-----------|----------|
# MAGIC | **Wealth** | `income_log`, `total_balance_log` |
# MAGIC | **Revenue** | `total_revenue_log`, `revenue_per_product` |
# MAGIC | **Behavior** | `digital_ratio`, `atm_intensity` |
# MAGIC | **Products** | `nb_active_products`, `product_diversity`, `product_churn_rate` |
# MAGIC | **Demographics** | `customer_age`, `tenure_months` |

# COMMAND ----------

# Define our selected feature set
SELECTED_FEATURES = [
    "income_log",
    "total_balance_log",
    "total_revenue_log",
    "digital_ratio",
    "revenue_per_product",
    "product_churn_rate",
    "atm_intensity",
    "nb_active_products",
    "product_diversity",
    "customer_age",
    "tenure_months",
]

# Keep only features that exist in our dataframe
available_features = [f for f in SELECTED_FEATURES if f in df.columns]
missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]

print(f"Selected features ({len(available_features)}/{len(SELECTED_FEATURES)}):")
for i, f in enumerate(available_features, 1):
    print(f"  {i:2d}. {f}")

if missing_features:
    print(f"\n⚠ Missing features (will be skipped): {missing_features}")

# Prepare the feature matrix — drop rows with nulls
X_raw = df[available_features].dropna()
print(f"\nFeature matrix shape: {X_raw.shape[0]:,} customers × {X_raw.shape[1]} features")
print(f"Dropped {len(df) - len(X_raw):,} rows with missing values")

# Keep aligned index for later
valid_index = X_raw.index
X_raw.describe().round(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3.5 — Standardization (Feature Scaling)
# MAGIC
# MAGIC **Why standardize?** K-Means uses Euclidean distance to assign customers to clusters. If `income_log` ranges from 8–12 while `digital_ratio` ranges from 0–1, then income would dominate the distance calculation simply because its numbers are bigger — not because it's more important.
# MAGIC
# MAGIC **StandardScaler** transforms each feature to have **mean = 0** and **standard deviation = 1**, putting all features on an equal footing.
# MAGIC
# MAGIC $$z = \frac{x - \mu}{\sigma}$$

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled_df = pd.DataFrame(X_scaled, columns=available_features, index=valid_index)

print("Standardization complete!")
print(f"  Mean of each feature (should be ~0): {X_scaled_df.mean().round(4).tolist()[:3]} ...")
print(f"  Std  of each feature (should be ~1): {X_scaled_df.std().round(4).tolist()[:3]} ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Before vs. After Standardization
# MAGIC
# MAGIC Let's visually compare the distributions of two features before and after scaling. Notice how the shapes are preserved — only the axis scale changes.

# COMMAND ----------

# Pick two features to compare
compare_features = [f for f in ["income_log", "total_balance_log"] if f in available_features]
if len(compare_features) < 2:
    compare_features = available_features[:2]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col_idx, feat in enumerate(compare_features):
    # Original
    axes[0, col_idx].hist(X_raw[feat], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0, col_idx].set_title(f"Original — {feat}", fontsize=12, fontweight="bold")
    axes[0, col_idx].set_ylabel("Count")

    # Standardized
    axes[1, col_idx].hist(X_scaled_df[feat], bins=40, color="#DD8452", edgecolor="white", alpha=0.85)
    axes[1, col_idx].set_title(f"Standardized — {feat}", fontsize=12, fontweight="bold")
    axes[1, col_idx].set_ylabel("Count")

fig.suptitle("Effect of StandardScaler on Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # LESSON 3.4 — K-Means Customer Segmentation
# MAGIC
# MAGIC Now that our features are clean and standardized, we can run K-Means clustering. But first — **how many clusters (K) should we use?** There's no single right answer, but two widely-used techniques help us decide:
# MAGIC
# MAGIC 1. **Elbow Method** — looks for a "bend" in the WCSS curve
# MAGIC 2. **Silhouette Analysis** — measures how well-separated the clusters are
# MAGIC
# MAGIC We'll use both, then train our final model and log it to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4.1 — The Elbow Method
# MAGIC
# MAGIC The **Within-Cluster Sum of Squares (WCSS)** measures how tightly packed each cluster is. As K increases, WCSS always decreases (more clusters = smaller groups = tighter packing). We look for the **"elbow"** — the point where adding more clusters gives diminishing returns.
# MAGIC
# MAGIC $$\text{WCSS} = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

# COMMAND ----------

from sklearn.cluster import KMeans

K_RANGE = range(2, 11)
wcss_values = []

print("Computing WCSS for K = 2 to 10 ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    km.fit(X_scaled)
    wcss_values.append(km.inertia_)
    print(f"  K={k:2d}  |  WCSS = {km.inertia_:,.0f}")

# Plot the elbow curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(K_RANGE), wcss_values, "o-", color="#4C72B0", linewidth=2.5, markersize=8)
ax.set_xlabel("Number of Clusters (K)", fontsize=13)
ax.set_ylabel("WCSS (Inertia)", fontsize=13)
ax.set_title("Elbow Method — Finding the Optimal K", fontsize=15, fontweight="bold")
ax.set_xticks(list(K_RANGE))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the Elbow Plot
# MAGIC
# MAGIC Look for where the curve **bends sharply** — that's the elbow. Beyond that point, each additional cluster adds very little explanatory power. Typically for customer segmentation, the elbow falls around **K = 3–5**.
# MAGIC
# MAGIC > The elbow method gives a rough guide. Next, we'll use silhouette analysis for a more rigorous evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4.2 — Silhouette Analysis
# MAGIC
# MAGIC The **Silhouette Score** measures how similar a customer is to their own cluster vs. the nearest neighboring cluster. It ranges from **-1 to +1**:
# MAGIC
# MAGIC | Score Range | Interpretation |
# MAGIC |-------------|---------------|
# MAGIC | **0.71 – 1.0** | Strong structure |
# MAGIC | **0.51 – 0.70** | Reasonable structure |
# MAGIC | **0.26 – 0.50** | Weak structure, possibly overlapping |
# MAGIC | **≤ 0.25** | No substantial structure |
# MAGIC
# MAGIC $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
# MAGIC
# MAGIC where *a(i)* = avg distance to same-cluster points, *b(i)* = avg distance to nearest other cluster.

# COMMAND ----------

from sklearn.metrics import silhouette_score

silhouette_scores = []

print("Computing Silhouette Scores for K = 2 to 10 ...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  K={k:2d}  |  Silhouette Score = {score:.4f}")

optimal_k = list(K_RANGE)[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)
print(f"\n>>> Optimal K by Silhouette Score: K = {optimal_k} (score = {best_silhouette:.4f})")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(list(K_RANGE), silhouette_scores, color="#4C72B0", edgecolor="white", alpha=0.85)
# Highlight the best
bars[np.argmax(silhouette_scores)].set_color("#DD8452")
bars[np.argmax(silhouette_scores)].set_edgecolor("#C44E52")
ax.set_xlabel("Number of Clusters (K)", fontsize=13)
ax.set_ylabel("Silhouette Score", fontsize=13)
ax.set_title("Silhouette Analysis — Higher is Better", fontsize=15, fontweight="bold")
ax.set_xticks(list(K_RANGE))
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4.3 — Train the Final K-Means Model
# MAGIC
# MAGIC Based on our analysis, we'll use **K = 4** clusters (a common sweet spot for banking segmentation that balances granularity with interpretability). We'll train the final model, assign labels, and log everything to **MLflow** for reproducibility.

# COMMAND ----------

import mlflow
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

FINAL_K = 4

# Set up MLflow experiment
experiment_name = "/Shared/analyticsoul_segmentation"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"kmeans_k{FINAL_K}") as run:
    # Train the model
    kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Compute metrics
    sil_score = silhouette_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    # Log parameters
    mlflow.log_param("k", FINAL_K)
    mlflow.log_param("n_features", len(available_features))
    mlflow.log_param("features", ", ".join(available_features))
    mlflow.log_param("n_customers", len(X_scaled))
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("n_init", 10)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("inertia", kmeans.inertia_)
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("calinski_harabasz_score", ch_score)
    mlflow.log_metric("davies_bouldin_score", db_score)

    # Log model
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"\nModel: KMeans(K={FINAL_K})")
    print(f"  Inertia (WCSS):         {kmeans.inertia_:,.0f}")
    print(f"  Silhouette Score:       {sil_score:.4f}")
    print(f"  Calinski-Harabasz:      {ch_score:,.0f}")
    print(f"  Davies-Bouldin:         {db_score:.4f}")

# Assign cluster labels back to the dataframe
df.loc[valid_index, "cluster_id"] = cluster_labels
df["cluster_id"] = df["cluster_id"].astype("Int64")

print(f"\nCluster labels assigned to {len(valid_index):,} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4.4 — Cluster Size Distribution
# MAGIC
# MAGIC Let's see how customers are distributed across the 4 clusters. Ideally, we want **reasonably balanced** clusters — if one cluster has 90% of customers, the segmentation isn't very useful.

# COMMAND ----------

cluster_counts = df.loc[valid_index, "cluster_id"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
axes[0].bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors[:len(cluster_counts)], edgecolor="white")
for i, (idx, val) in enumerate(cluster_counts.items()):
    axes[0].text(i, val + max(cluster_counts.values) * 0.02, f"{val:,}", ha="center", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Cluster", fontsize=13)
axes[0].set_ylabel("Number of Customers", fontsize=13)
axes[0].set_title("Customers per Cluster", fontsize=15, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

# Pie chart
axes[1].pie(
    cluster_counts.values,
    labels=[f"Cluster {i}" for i in cluster_counts.index],
    colors=colors[:len(cluster_counts)],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 12},
    pctdistance=0.75,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[1].set_title("Cluster Distribution (%)", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.show()

print("\nCluster Sizes:")
for idx, count in cluster_counts.items():
    pct = count / cluster_counts.sum() * 100
    print(f"  Cluster {idx}: {count:>6,} customers ({pct:5.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # LESSON 3.5 — Interpreting the Segmentation Results
# MAGIC
# MAGIC Numbers are useless without **meaning**. In this lesson, we'll transform abstract cluster IDs into actionable business segments by:
# MAGIC 1. Profiling each cluster (what makes them unique)
# MAGIC 2. Visualizing profiles with radar charts and box plots
# MAGIC 3. Assigning business-meaningful names
# MAGIC 4. Developing targeted strategies for each segment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5.1 — Cluster Profiles (Centroid Analysis)
# MAGIC
# MAGIC The cluster **centroids** (means) tell us the "average customer" in each cluster. By comparing centroids across features, we can identify what makes each cluster distinctive.

# COMMAND ----------

# Compute mean of each feature per cluster
df_clustered = df.loc[valid_index].copy()
cluster_profiles = df_clustered.groupby("cluster_id")[available_features].mean()

print("=== Cluster Centroids (Feature Means) ===\n")
display(cluster_profiles.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Centroid Heatmap
# MAGIC
# MAGIC A heatmap makes it easy to spot which features are **high** (red) or **low** (blue) for each cluster at a glance.

# COMMAND ----------

# Standardize the centroids for comparable heatmap colors
from sklearn.preprocessing import MinMaxScaler

profile_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(cluster_profiles),
    index=cluster_profiles.index,
    columns=cluster_profiles.columns,
)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    profile_scaled,
    annot=cluster_profiles.round(2).values,
    fmt="",
    cmap="RdYlBu_r",
    linewidths=1,
    ax=ax,
    cbar_kws={"label": "Normalized Value (0-1)"},
    yticklabels=[f"Cluster {i}" for i in cluster_profiles.index],
)
ax.set_title("Cluster Centroid Heatmap — Feature Averages per Segment", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("")
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5.2 — Radar (Spider) Charts
# MAGIC
# MAGIC Radar charts show the **shape** of each cluster's profile. Each spoke represents a feature, and the polygon's shape reveals the cluster's character at a glance. A "spiky" profile means the cluster is extreme on certain dimensions; a "round" profile means it's balanced.

# COMMAND ----------

from math import pi

# Normalize centroids to 0-1 for radar chart
radar_data = profile_scaled.copy()
categories = list(radar_data.columns)
N = len(categories)

# Compute angle for each feature
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the polygon

fig, axes = plt.subplots(1, FINAL_K, figsize=(5 * FINAL_K, 5), subplot_kw=dict(polar=True))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

if FINAL_K == 1:
    axes = [axes]

for idx, (cluster_id, row) in enumerate(radar_data.iterrows()):
    ax = axes[idx]
    values = row.tolist()
    values += values[:1]  # Close the polygon

    ax.plot(angles, values, "o-", linewidth=2, color=colors[idx], markersize=5)
    ax.fill(angles, values, alpha=0.2, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"Cluster {cluster_id}", fontsize=14, fontweight="bold", pad=20, color=colors[idx])
    ax.grid(True, alpha=0.3)

plt.suptitle("Cluster Radar Profiles", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5.3 — Box Plots by Cluster
# MAGIC
# MAGIC Radar charts show the *average*, but real customers vary. Box plots reveal the **full distribution** within each cluster — including outliers and overlap between clusters. We'll look at four key features.

# COMMAND ----------

# Select features for box plots
box_features = [f for f in ["income_log", "total_balance_log", "digital_ratio", "product_diversity"] if f in available_features]
if len(box_features) < 4:
    box_features = available_features[:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for idx, feat in enumerate(box_features):
    ax = axes[idx // 2][idx % 2]
    bp = df_clustered.boxplot(
        column=feat,
        by="cluster_id",
        ax=ax,
        patch_artist=True,
        showfliers=False,
        return_type="dict",
    )
    # Color the boxes
    for i, box in enumerate(bp[feat]["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.7)
    ax.set_title(feat, fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel(feat, fontsize=11)

fig.suptitle("Feature Distributions by Cluster", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5.4 — Cluster Naming
# MAGIC
# MAGIC Now comes the **most important step**: translating cluster IDs into names that business stakeholders can act on. We examine each cluster's profile and assign a descriptive name based on its dominant characteristics.
# MAGIC
# MAGIC The logic below uses the centroid values to automatically name each cluster, but in practice you'd refine these with domain experts.

# COMMAND ----------

# Analyze centroids to assign names
# Normalize centroids for comparison
profile_norm = profile_scaled.copy()

segment_names = {}
segment_descriptions = {}

for cluster_id in cluster_profiles.index:
    row = profile_norm.loc[cluster_id]

    # Scoring heuristics based on feature levels
    income_level = row.get("income_log", 0.5)
    balance_level = row.get("total_balance_log", 0.5)
    digital_level = row.get("digital_ratio", 0.5)
    churn_level = row.get("product_churn_rate", 0.5)
    products_level = row.get("nb_active_products", 0.5)
    diversity_level = row.get("product_diversity", 0.5)
    revenue_level = row.get("total_revenue_log", 0.5)
    tenure_level = row.get("tenure_months", 0.5)

    # Classification logic
    if income_level > 0.65 and balance_level > 0.65 and products_level > 0.55:
        name = "Premium Wealth"
        desc = "High income, high balances, multiple products. The bank's most valuable customers."
    elif digital_level > 0.6 and income_level > 0.4 and churn_level < 0.5:
        name = "Digital Engaged"
        desc = "Digitally savvy, moderate-to-high income, loyal. Prime candidates for digital product cross-sell."
    elif churn_level > 0.55 or (products_level < 0.35 and revenue_level < 0.35):
        name = "At Risk"
        desc = "Low product engagement, higher churn rates. Require retention campaigns and re-engagement."
    elif tenure_level > 0.55 and income_level < 0.5 and digital_level < 0.45:
        name = "Traditional Loyal"
        desc = "Long-tenured, lower income, prefer branch/ATM. Stable but opportunity to grow digitally."
    elif income_level < 0.4 and balance_level < 0.4:
        name = "Mass Market"
        desc = "Lower income and balances. Cost-efficient servicing and financial wellness programs."
    else:
        name = "Core Mainstream"
        desc = "Middle-of-the-road across most dimensions. The bank's bread-and-butter segment."

    segment_names[cluster_id] = name
    segment_descriptions[cluster_id] = desc

# Apply names to dataframe
df.loc[valid_index, "segment_name"] = df.loc[valid_index, "cluster_id"].map(segment_names)

print("=== SEGMENT NAMING ===\n")
for cid in sorted(segment_names.keys()):
    count = cluster_counts.get(cid, 0)
    pct = count / cluster_counts.sum() * 100
    print(f"Cluster {cid} --> \"{segment_names[cid]}\"")
    print(f"  Customers: {count:,} ({pct:.1f}%)")
    print(f"  Profile:   {segment_descriptions[cid]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5.5 — Business Insights & Recommended Actions
# MAGIC
# MAGIC Each segment warrants a different strategy. Below is a summary of key traits and recommended actions for each named segment.

# COMMAND ----------

# Build summary table
summary_rows = []
for cid in sorted(segment_names.keys()):
    count = cluster_counts.get(cid, 0)
    pct = count / cluster_counts.sum() * 100

    # Determine recommended action based on segment name
    name = segment_names[cid]
    if "Premium" in name:
        action = "White-glove service, wealth management upsell, priority support"
    elif "Digital" in name:
        action = "Mobile-first offers, digital product launches, in-app engagement"
    elif "At Risk" in name:
        action = "Proactive retention campaigns, satisfaction surveys, incentive offers"
    elif "Traditional" in name:
        action = "Branch experience optimization, gentle digital onboarding, loyalty rewards"
    elif "Mass Market" in name:
        action = "Cost-efficient digital servicing, financial wellness tools, fee simplification"
    else:
        action = "Balanced cross-sell campaigns, lifecycle marketing, engagement scoring"

    summary_rows.append({
        "Segment": name,
        "Cluster": cid,
        "Size": f"{count:,}",
        "Share": f"{pct:.1f}%",
        "Key Traits": segment_descriptions[cid],
        "Recommended Action": action,
    })

summary_df = pd.DataFrame(summary_rows)
print("=== SEGMENT STRATEGY SUMMARY ===\n")
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segment Strategy Deep Dive
# MAGIC
# MAGIC | Segment | Strategy | Expected Impact |
# MAGIC |---------|----------|-----------------|
# MAGIC | **Premium Wealth** | Assign dedicated relationship managers, offer investment products, premium credit cards | High revenue retention, 15-20% increase in AUM |
# MAGIC | **Digital Engaged** | Launch mobile-exclusive savings products, enable instant credit decisions, gamify engagement | 25-30% increase in digital transaction volume |
# MAGIC | **At Risk** | Trigger outbound calls, offer fee waivers, personalized win-back offers within 30 days | 10-15% reduction in attrition rate |
# MAGIC | **Traditional Loyal** | Offer tablet-assisted branch experience, simplified mobile app, maintain personal touch | Gradual digital migration, 5-10% cost reduction |
# MAGIC | **Core Mainstream** | Lifecycle-based product bundles, seasonal campaigns, referral programs | Steady growth through cross-sell, 8-12% revenue lift |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # LESSON 3.6 — Dimensionality Reduction with PCA
# MAGIC
# MAGIC We've been working in **11-dimensional space** — impossible to visualize directly. **Principal Component Analysis (PCA)** lets us project our data into 2D or 3D while preserving as much information as possible. This is invaluable for:
# MAGIC
# MAGIC 1. **Visualization** — see the cluster structure with your own eyes
# MAGIC 2. **Validation** — check if clusters are truly separated
# MAGIC 3. **Understanding** — learn which features drive the most variation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6.1 — PCA Explained Variance
# MAGIC
# MAGIC Each principal component captures a portion of the total variance in the data. The first component captures the most, the second captures the next most, and so on. We plot the **cumulative explained variance** to decide how many components we need.

# COMMAND ----------

from sklearn.decomposition import PCA

# Fit PCA with all components
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# Cumulative explained variance
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
individual_var = pca_full.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(12, 5))
n_components = len(cumulative_var)

# Bar chart for individual variance
ax.bar(range(1, n_components + 1), individual_var, color="#4C72B0", alpha=0.6, label="Individual", edgecolor="white")
# Line for cumulative
ax.plot(range(1, n_components + 1), cumulative_var, "o-", color="#C44E52", linewidth=2.5, markersize=7, label="Cumulative")

# Mark 80% and 90% thresholds
for threshold in [80, 90]:
    ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_components + 0.2, threshold, f"{threshold}%", fontsize=10, color="gray")

# Find components for 80% and 90%
n_80 = np.argmax(cumulative_var >= 80) + 1
n_90 = np.argmax(cumulative_var >= 90) + 1

ax.set_xlabel("Principal Component", fontsize=13)
ax.set_ylabel("Explained Variance (%)", fontsize=13)
ax.set_title("PCA — Explained Variance by Component", fontsize=15, fontweight="bold")
ax.set_xticks(range(1, n_components + 1))
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

print(f"\nVariance explained:")
print(f"  2 components: {cumulative_var[1]:.1f}%")
print(f"  3 components: {cumulative_var[2]:.1f}%")
print(f"  {n_80} components explain 80% of variance")
print(f"  {n_90} components explain 90% of variance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6.2 — 2D PCA Visualization
# MAGIC
# MAGIC Let's project all customers into the first two principal components and color them by cluster. If our segmentation is good, we should see **distinct, well-separated groups** in this 2D space.

# COMMAND ----------

# Reduce to 2 components
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Also transform the cluster centers
centers_pca_2d = pca_2d.transform(kmeans.cluster_centers_)

fig, ax = plt.subplots(figsize=(12, 8))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
cluster_labels_arr = cluster_labels

for i in range(FINAL_K):
    mask = cluster_labels_arr == i
    name = segment_names.get(i, f"Cluster {i}")
    ax.scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c=colors[i], label=name,
        alpha=0.5, s=20, edgecolors="none",
    )

# Plot cluster centers
ax.scatter(
    centers_pca_2d[:, 0], centers_pca_2d[:, 1],
    c="black", marker="X", s=200, linewidths=2, edgecolors="white",
    zorder=10, label="Centroids",
)

ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=13)
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=13)
ax.set_title("Customer Segments in 2D PCA Space", fontsize=16, fontweight="bold")
ax.legend(fontsize=11, loc="best", framealpha=0.9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6.3 — 3D PCA Visualization
# MAGIC
# MAGIC Adding a third component captures more variance and can reveal structure hidden in 2D. We use **Plotly** for an interactive 3D scatter plot — rotate, zoom, and explore the clusters in 3D space.

# COMMAND ----------

import plotly.express as px

# Reduce to 3 components
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca_3d[:, 0],
    "PC2": X_pca_3d[:, 1],
    "PC3": X_pca_3d[:, 2],
    "Segment": [segment_names.get(c, f"Cluster {c}") for c in cluster_labels_arr],
    "Cluster": cluster_labels_arr.astype(str),
})

fig = px.scatter_3d(
    plot_df,
    x="PC1", y="PC2", z="PC3",
    color="Segment",
    color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    opacity=0.5,
    title=f"Customer Segments in 3D PCA Space ({pca_3d.explained_variance_ratio_.sum()*100:.1f}% variance explained)",
    labels={
        "PC1": f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)",
        "PC2": f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)",
        "PC3": f"PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)",
    },
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(
    width=900, height=700,
    font=dict(size=12),
    title_font=dict(size=16),
    legend=dict(font=dict(size=12)),
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6.4 — PCA Component Loadings
# MAGIC
# MAGIC The **loadings** tell us how much each original feature contributes to each principal component. This helps us interpret what the axes in our PCA plots actually mean in business terms.

# COMMAND ----------

# Component loadings
loadings = pd.DataFrame(
    pca_full.components_[:3],  # First 3 components
    columns=available_features,
    index=["PC1", "PC2", "PC3"],
)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
    loadings,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=1,
    ax=ax,
    cbar_kws={"label": "Loading"},
)
ax.set_title("PCA Component Loadings — What Drives Each Component?", fontsize=15, fontweight="bold", pad=15)
ax.set_ylabel("Principal Component", fontsize=12)
ax.set_xlabel("")
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# Interpret the top loadings
print("\n=== COMPONENT INTERPRETATION ===\n")
for pc in ["PC1", "PC2", "PC3"]:
    top_pos = loadings.loc[pc].nlargest(3)
    top_neg = loadings.loc[pc].nsmallest(3)
    print(f"{pc}:")
    print(f"  Driven by:   {', '.join([f'{f} (+{v:.2f})' for f, v in top_pos.items()])}")
    print(f"  Opposed to:  {', '.join([f'{f} ({v:.2f})' for f, v in top_neg.items()])}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpreting the Components
# MAGIC
# MAGIC | Component | Business Interpretation |
# MAGIC |-----------|----------------------|
# MAGIC | **PC1** | Likely represents the **wealth/engagement axis** — separates high-income, high-balance customers from lower-value ones |
# MAGIC | **PC2** | Likely captures the **digital vs. traditional axis** — separates digitally active customers from ATM/branch-heavy ones |
# MAGIC | **PC3** | Likely reflects the **product breadth axis** — distinguishes multi-product holders from single-product customers |
# MAGIC
# MAGIC > These interpretations depend on your specific data. Always verify by checking the actual loading values above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6.5 — Final Summary & Save Results
# MAGIC
# MAGIC Let's save the final customer-segment assignments to a Delta table so they can be used by downstream applications (the segmentation app, dashboards, marketing campaigns).

# COMMAND ----------

# Prepare final output
output_df = df.loc[valid_index, ["customer_id", "cluster_id", "segment_name"]].copy()
output_df["pc1"] = X_pca_3d[:, 0]
output_df["pc2"] = X_pca_3d[:, 1]
output_df["pc3"] = X_pca_3d[:, 2]

# Convert to Spark and save
output_table = f"{CATALOG}.{SCHEMA}.13_analyticsoul_segments"

spark_df = spark.createDataFrame(output_df)
spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)

print(f"Saved to: {output_table}")
print(f"  Total customers: {len(output_df):,}")
print(f"  Segments: {output_df['segment_name'].nunique()}")
print(f"  Silhouette Score: {sil_score:.4f}")
print(f"  Variance explained (2 PCs): {cumulative_var[1]:.1f}%")
print(f"  Variance explained (3 PCs): {cumulative_var[2]:.1f}%")

display(spark_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Workshop Complete!
# MAGIC
# MAGIC ### What We Accomplished
# MAGIC
# MAGIC | Step | What | Key Output |
# MAGIC |------|------|------------|
# MAGIC | **Feature Engineering** | Encoded categoricals, analyzed correlations, selected 11 features, standardized | Clean feature matrix |
# MAGIC | **K-Means Clustering** | Elbow + silhouette analysis, trained K=4 model, logged to MLflow | 4 customer clusters |
# MAGIC | **Interpretation** | Centroid profiles, radar charts, box plots, business naming | Actionable segment strategy |
# MAGIC | **PCA Visualization** | Explained variance, 2D/3D projections, component loadings | Visual proof of cluster quality |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC 1. **Feature engineering is 80% of the work** — garbage in, garbage out. Encoding, correlation analysis, and scaling are essential.
# MAGIC 2. **K-Means is simple but powerful** — with good features and proper K selection, it produces actionable segments.
# MAGIC 3. **Interpretation matters more than the algorithm** — clusters only create value when business teams can act on them.
# MAGIC 4. **PCA is your visualization friend** — it lets you "see" high-dimensional clusters and validate your segmentation.
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Feed these segments into the **Customer Intelligence Studio** for interactive exploration
# MAGIC - Build **AI-powered personas** for each segment using LLM-based summarization
# MAGIC - Design **A/B tests** to validate the recommended actions for each segment
# MAGIC - Schedule **periodic re-clustering** to detect segment drift over time
