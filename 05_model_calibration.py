# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Model Calibration
# MAGIC **Catalog:** `customer_segmentation.banking`
# MAGIC
# MAGIC K-Means clustering with elbow/silhouette analysis:
# MAGIC 1. Feature preparation (log transforms, standardization)
# MAGIC 2. Elbow & silhouette analysis (K=3-8) with visualizations
# MAGIC 3. Final model K=4 — score reference + historical data
# MAGIC 4. Cluster profiling, heatmap, radar chart, auto-labeling
# MAGIC 5. Save all output tables

# COMMAND ----------

# MAGIC %pip install scikit-learn matplotlib seaborn --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "banking"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature Preparation

# COMMAND ----------

def prepare_clustering_features(df, feature_cols, log_cols):
    """Prepare features: fill nulls, log-transform skewed columns, standardize."""
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    X_log = X.copy()
    for idx in [feature_cols.index(c) for c in log_cols]:
        X_log[:, idx] = np.log1p(np.abs(X_log[:, idx]))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled, scaler

# COMMAND ----------

df_ref = spark.table(f"{CATALOG}.{SCHEMA}.04_customer_features_reference").toPandas()

exclude_cols = {
    "customer_id", "first_name", "last_name", "gender", "region", "tier",
    "reference_date", "project_reference_date", "age_bracket",
}
feature_cols = [
    c for c in df_ref.columns
    if c not in exclude_cols and df_ref[c].dtype.kind in ("f", "i")
]

# Identify skewed monetary/count columns for log transform
LOG_COLS = [
    c for c in feature_cols
    if any(kw in c.lower() for kw in ["income", "balance", "revenue", "atm_withdrawals_sum"])
]

print(f"Numeric features for clustering ({len(feature_cols)}):")
for i, c in enumerate(feature_cols, 1):
    tag = " [log]" if c in LOG_COLS else ""
    print(f"  {i:>2}. {c}{tag}")

X_scaled, scaler = prepare_clustering_features(df_ref, feature_cols, LOG_COLS)

print(f"\nClustering matrix: {X_scaled.shape[0]:,} samples x {X_scaled.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Elbow & Silhouette Analysis (K = 3 to 8)

# COMMAND ----------

K_RANGE = range(3, 9)
results = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=1337)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    ch  = calinski_harabasz_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    results.append({
        "k": int(k),
        "inertia": float(round(km.inertia_, 0)),
        "silhouette": float(round(sil, 4)),
        "calinski_harabasz": float(round(ch, 0)),
        "davies_bouldin": float(round(db, 4)),
    })
    print(f"  K={k}: Silhouette={sil:.4f}  CH={ch:,.0f}  DB={db:.4f}  Inertia={km.inertia_:,.0f}")

metrics_df = pd.DataFrame(results)
print("\n", metrics_df.to_string(index=False))

# Save metrics table
spark.createDataFrame(results).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_cluster_metrics"
)
print(f"\nSaved 05_cluster_metrics ({len(results)} rows)")

# COMMAND ----------

# --- Visualize 2x2 metric grid ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("K-Means Cluster Selection Metrics", fontsize=14, fontweight="bold")

ks = metrics_df["k"].values

ax = axes[0, 0]
ax.plot(ks, metrics_df["inertia"], "o-", color="#2563eb", linewidth=2)
ax.set_title("Inertia (Elbow Curve)")
ax.set_xlabel("K")
ax.set_ylabel("Inertia")
ax.ticklabel_format(style="plain", axis="y")

ax = axes[0, 1]
ax.plot(ks, metrics_df["silhouette"], "s-", color="#16a34a", linewidth=2)
ax.set_title("Silhouette Score (higher = better)")
ax.set_xlabel("K")
ax.set_ylabel("Silhouette")

ax = axes[1, 0]
ax.plot(ks, metrics_df["calinski_harabasz"], "^-", color="#dc2626", linewidth=2)
ax.set_title("Calinski-Harabasz Index (higher = better)")
ax.set_xlabel("K")
ax.set_ylabel("CH Index")
ax.ticklabel_format(style="plain", axis="y")

ax = axes[1, 1]
ax.plot(ks, metrics_df["davies_bouldin"], "D-", color="#9333ea", linewidth=2)
ax.set_title("Davies-Bouldin Index (lower = better)")
ax.set_xlabel("K")
ax.set_ylabel("DB Index")

for ax in axes.flat:
    ax.axvline(x=4, color="gray", linestyle="--", alpha=0.5, label="K=4")
    ax.legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train Final Model (K = 4)

# COMMAND ----------

OPTIMAL_K = 4

final_km = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=25, max_iter=500, random_state=1337)
df_ref["cluster_labels"] = final_km.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df_ref["cluster_labels"].values)
print(f"Final Model (K={OPTIMAL_K}): Silhouette Score = {sil:.4f}")

dist = df_ref["cluster_labels"].value_counts().sort_index()
dist_pct = (dist / len(df_ref) * 100).round(1)

print(f"\nSegment distribution:")
for c in dist.index:
    print(f"  Cluster {c}: {dist[c]:>6,} customers  ({dist_pct[c]}%)")

# COMMAND ----------

# --- Bar chart of segment sizes ---
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b"]
bars = ax.bar(dist.index, dist.values, color=colors[:OPTIMAL_K], edgecolor="white", linewidth=1.5)
for bar, count, pct in zip(bars, dist.values, dist_pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df_ref) * 0.005,
            f"{count:,}\n({pct}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Customers")
ax.set_title("Segment Size Distribution (K=4)", fontsize=13, fontweight="bold")
ax.set_xticks(dist.index)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score Historical Data

# COMMAND ----------

try:
    df_hist = spark.table(f"{CATALOG}.{SCHEMA}.04_customer_features").filter(
        "reference_date != project_reference_date"
    ).toPandas()
    has_historical = len(df_hist) > 0
except:
    has_historical = False
    df_hist = pd.DataFrame()
    print("No historical panel data found — skipping historical scoring")

if has_historical:
    # Filter feature_cols to those available in historical data
    hist_feature_cols = [c for c in feature_cols if c in df_hist.columns]
    missing_cols = set(feature_cols) - set(hist_feature_cols)
    if missing_cols:
        print(f"  Note: {len(missing_cols)} features missing in historical data, filling with 0:")
        for mc in sorted(missing_cols):
            print(f"    - {mc}")
            df_hist[mc] = 0
        hist_feature_cols = feature_cols  # Now all cols exist

    # Apply same log-transform + scaler.transform
    X_hist = df_hist[hist_feature_cols].fillna(0).values.astype(np.float64)
    X_hist_log = X_hist.copy()
    log_indices = [hist_feature_cols.index(c) for c in LOG_COLS if c in hist_feature_cols]
    for idx in log_indices:
        X_hist_log[:, idx] = np.log1p(np.abs(X_hist_log[:, idx]))
    X_hist_scaled = scaler.transform(X_hist_log)

    df_hist["cluster_labels"] = final_km.predict(X_hist_scaled)

    print(f"Scored historical data: {len(df_hist):,} rows")
    hist_dist = df_hist["cluster_labels"].value_counts().sort_index()
    for c in hist_dist.index:
        print(f"  Cluster {c}: {hist_dist[c]:>6,}")
else:
    print("Historical scoring skipped — no panel data available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cluster Profiling & Auto-Labeling

# COMMAND ----------

# --- Compute mean feature values per cluster ---
profile_cols = [c for c in feature_cols if c in df_ref.columns]
profiles = df_ref.groupby("cluster_labels")[profile_cols].mean().round(2)
profiles["count"] = df_ref.groupby("cluster_labels")["customer_id"].count()
profiles["pct"] = (profiles["count"] / len(df_ref) * 100).round(1)

print("=== Cluster Profiles (key features) ===")
key_display = ["count", "pct", "income", "customer_age", "tenure_months"]
key_display += [c for c in profiles.columns if "balance_avg" in c.lower()][:2]
key_display += [c for c in profiles.columns if "revenue_avg" in c.lower()][:2]
key_display += [c for c in ["nb_subscribed_products", "digital_services_avg"] if c in profiles.columns]
avail = [c for c in key_display if c in profiles.columns]
print(profiles[avail].to_string())

# COMMAND ----------

# --- Heatmap of normalized feature means ---
heatmap_data = profiles[profile_cols].copy()
# Standardize across clusters (column-wise z-score) so patterns are visible
heatmap_norm = heatmap_data.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9), axis=0)

fig, ax = plt.subplots(figsize=(max(12, len(profile_cols) * 0.6), 5))
sns.heatmap(
    heatmap_norm,
    annot=False,
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    yticklabels=[f"Cluster {i}" for i in heatmap_norm.index],
    ax=ax,
)
ax.set_title("Cluster Feature Profiles (z-scored across clusters)", fontsize=13, fontweight="bold")
ax.set_xlabel("Features")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.show()

# COMMAND ----------

# --- Radar / Spider chart of key features per cluster ---
radar_candidates = ["income", "customer_age", "tenure_months", "nb_subscribed_products"]
radar_candidates += [c for c in profile_cols if "balance_avg" in c.lower()][:1]
radar_candidates += [c for c in profile_cols if "revenue_avg" in c.lower()][:1]
radar_candidates += [c for c in profile_cols if "digital_services" in c.lower()][:1]
radar_candidates += [c for c in profile_cols if "transaction" in c.lower() or "nb_of_transactions" in c.lower()][:1]
radar_cols = [c for c in radar_candidates if c in profiles.columns]

# Normalize each feature to 0-1 range for the radar chart
radar_data = profiles[radar_cols].copy()
radar_norm = radar_data.apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-9), axis=0)

num_vars = len(radar_cols)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#9333ea", "#ec4899"]

for i, (cluster_id, row) in enumerate(radar_norm.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, color=colors_radar[i % len(colors_radar)],
            label=f"Cluster {cluster_id}")
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i % len(colors_radar)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.replace("_", "\n") for c in radar_cols], fontsize=8)
ax.set_title("Cluster Radar Chart (normalized features)", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# --- Tier distribution per cluster ---
tier_dist = df_ref.groupby(["cluster_labels", "tier"]).size().unstack(fill_value=0)
tier_pct = tier_dist.div(tier_dist.sum(axis=1), axis=0).round(3)

print("=== Tier Distribution per Cluster (%) ===")
print((tier_pct * 100).round(1).to_string())

# COMMAND ----------

# --- Auto-label segments ---
def label_cluster(row):
    high_income = row["income"] > profiles["income"].quantile(0.75)
    high_digital = row.get("digital_services_avg", 0) > profiles.get(
        "digital_services_avg", pd.Series([0])
    ).median()
    many_products = row["nb_subscribed_products"] > profiles["nb_subscribed_products"].median()
    low_products = row["nb_subscribed_products"] < profiles["nb_subscribed_products"].quantile(0.25)

    balance_cols = [c for c in profiles.columns if "balance_avg" in c.lower() or "balance_last" in c.lower()]
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

profiles["segment_label"] = profiles.apply(label_cluster, axis=1)

# De-duplicate labels — ensure each cluster has a unique label
label_counts = profiles["segment_label"].value_counts()
for dup in label_counts[label_counts > 1].index:
    mask = profiles["segment_label"] == dup
    for i, idx in enumerate(profiles[mask].index.tolist()):
        if i > 0:
            profiles.loc[idx, "segment_label"] = f"{dup} (Group {i + 1})"

segment_map = profiles["segment_label"].to_dict()
df_ref["segment_label"] = df_ref["cluster_labels"].map(segment_map)
if has_historical and len(df_hist) > 0:
    df_hist["segment_label"] = df_hist["cluster_labels"].map(segment_map)

print("=== Segment Labels ===")
for cid, label in sorted(segment_map.items()):
    count = int(profiles.loc[cid, "count"])
    pct = profiles.loc[cid, "pct"]
    print(f"  Cluster {cid}: {label:35s}  {count:>6,} customers  ({pct}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save All Outputs

# COMMAND ----------

# --- 05_cluster_profiles ---
profiles_reset = profiles.reset_index()
for col in profiles_reset.columns:
    if profiles_reset[col].dtype.kind == "f":
        profiles_reset[col] = profiles_reset[col].astype(float)
    elif profiles_reset[col].dtype.kind == "i":
        profiles_reset[col] = profiles_reset[col].astype(int)
spark.createDataFrame(profiles_reset).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_cluster_profiles"
)
print("Saved 05_cluster_profiles")

# --- 05_customer_reference_scored ---
spark.createDataFrame(df_ref).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_customer_reference_scored"
)
print(f"Saved 05_customer_reference_scored: {len(df_ref):,} rows")

# --- 05_customer_historical_scored ---
if has_historical and len(df_hist) > 0:
    spark.createDataFrame(df_hist).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{CATALOG}.{SCHEMA}.05_customer_historical_scored"
    )
    print(f"Saved 05_customer_historical_scored: {len(df_hist):,} rows")
else:
    print("Skipped 05_customer_historical_scored (no historical data)")

# COMMAND ----------

# --- 05_age_demographics ---
ref_scored_spark = spark.createDataFrame(df_ref)
w = Window.orderBy("customer_age")

age_demo = (
    ref_scored_spark
    .withColumn("age_quintile", F.ntile(5).over(w))
    .groupBy("tier", "cluster_labels", "segment_label", "age_quintile")
    .agg(
        F.count("*").alias("customer_count"),
        F.min("customer_age").alias("age_min"),
        F.max("customer_age").alias("age_max"),
    )
    .withColumn("age_bounds", F.concat_ws("-", F.col("age_min").cast("int"), F.col("age_max").cast("int")))
)

age_demo.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_age_demographics"
)
print(f"Saved 05_age_demographics: {age_demo.count()} rows")

# COMMAND ----------

# --- 05_customers_nbo ---
nbo_cols = [
    "customer_id", "first_name", "last_name", "gender", "tier", "income", "region",
    "customer_age", "tenure_months", "cluster_labels", "segment_label",
]
customers_nbo_spark = spark.createDataFrame(df_ref[[c for c in nbo_cols if c in df_ref.columns]])
customers_nbo_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_customers_nbo"
)
print(f"Saved 05_customers_nbo: {customers_nbo_spark.count():,} rows")

# --- 05_customer_segments ---
spark.createDataFrame(
    df_ref[["customer_id", "cluster_labels", "segment_label"]]
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.05_customer_segments"
)
print("Saved 05_customer_segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary

# COMMAND ----------

print("=" * 70)
print("  CUSTOMER SEGMENTATION SUMMARY")
print("=" * 70)
print(f"\n  Model: K-Means  |  K = {OPTIMAL_K}  |  Silhouette = {sil:.4f}")
print(f"  Reference customers scored: {len(df_ref):,}")
print(f"  Historical records scored:  {len(df_hist) if has_historical else 0:,}")
print()

for cid, label in sorted(segment_map.items()):
    count = int(profiles.loc[cid, "count"])
    pct = profiles.loc[cid, "pct"]
    inc = profiles.loc[cid, "income"]
    prods = profiles.loc[cid, "nb_subscribed_products"]

    # Build a short characteristic string
    chars = []
    if inc > profiles["income"].quantile(0.75):
        chars.append("high income")
    elif inc < profiles["income"].quantile(0.25):
        chars.append("low income")
    else:
        chars.append("mid income")

    if prods > profiles["nb_subscribed_products"].median():
        chars.append("many products")
    else:
        chars.append("few products")

    if "digital_services_avg" in profiles.columns:
        dig = profiles.loc[cid, "digital_services_avg"]
        if dig > profiles["digital_services_avg"].median():
            chars.append("digitally active")

    print(f"  Cluster {cid}: {label}")
    print(f"           {count:>6,} customers ({pct}%)  |  {', '.join(chars)}")
    print()

print("  Output tables:")
print(f"    {CATALOG}.{SCHEMA}.05_cluster_metrics")
print(f"    {CATALOG}.{SCHEMA}.05_cluster_profiles")
print(f"    {CATALOG}.{SCHEMA}.05_customer_reference_scored")
print(f"    {CATALOG}.{SCHEMA}.05_customer_historical_scored")
print(f"    {CATALOG}.{SCHEMA}.05_age_demographics")
print(f"    {CATALOG}.{SCHEMA}.05_customers_nbo")
print(f"    {CATALOG}.{SCHEMA}.05_customer_segments")
print("=" * 70)
