# Databricks notebook source
# MAGIC %md
# MAGIC # 06 -- Model Validation: Temporal Segment Stability
# MAGIC **Catalog:** `customer_segmentation.ai_banking`
# MAGIC
# MAGIC Validates the clustering model by analyzing how customer segments evolve over time:
# MAGIC 1. Historical segment analysis (sizes and shares over time)
# MAGIC 2. 1-month transition matrix
# MAGIC 3. 3-month transition matrix
# MAGIC 4. Segment stability / retention rates
# MAGIC 5. Segment migration patterns
# MAGIC 6. Summary of key findings

# COMMAND ----------

# MAGIC %pip install matplotlib seaborn --quiet

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Historical Segment Analysis

# COMMAND ----------

ref_scored = spark.table(f"{CATALOG}.{SCHEMA}.ai_05_customer_reference_scored")

try:
    hist_scored = spark.table(f"{CATALOG}.{SCHEMA}.ai_05_customer_historical_scored")
    has_historical = hist_scored.count() > 0
except:
    has_historical = False

if not has_historical:
    print("No historical scored data available — temporal validation requires panel data.")
    print("Skipping temporal analysis. Run the pipeline with historical panel data to enable this notebook.")
    dbutils.notebook.exit("SKIPPED: No historical data available for temporal validation")

# Combine reference + historical scored data into a single timeline
all_scored = ref_scored.unionByName(hist_scored, allowMissingColumns=True)

# Identify revenue columns dynamically
rev_cols = [c for c in all_scored.columns if "revenue_last" in c.lower()]
if rev_cols:
    total_rev_expr = sum([F.coalesce(F.col(c), F.lit(0.0)) for c in rev_cols])
else:
    total_rev_expr = F.lit(0.0)

historical_segments = (
    all_scored
    .withColumn("total_revenue", total_rev_expr)
    .groupBy("reference_date", "cluster_labels", "segment_label")
    .agg(
        F.count("*").alias("customer_count"),
        F.sum("total_revenue").alias("total_revenue_sum"),
        F.avg("total_revenue").alias("total_revenue_avg"),
    )
    .orderBy("reference_date", "cluster_labels")
)

historical_segments.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_06_historical_segments"
)

row_count = historical_segments.count()
print(f"Saved ai_06_historical_segments: {row_count} rows")
display(historical_segments)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segment Sizes Over Time (Stacked Area Chart)

# COMMAND ----------

hs_pdf = historical_segments.toPandas()
hs_pdf["reference_date"] = pd.to_datetime(hs_pdf["reference_date"])

pivot_counts = hs_pdf.pivot_table(
    index="reference_date", columns="segment_label",
    values="customer_count", aggfunc="sum"
).fillna(0).sort_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.stackplot(
    pivot_counts.index, *[pivot_counts[c] for c in pivot_counts.columns],
    labels=pivot_counts.columns, alpha=0.85
)
ax.set_title("Segment Sizes Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Reference Date")
ax.set_ylabel("Customer Count")
ax.legend(loc="upper left", fontsize=9, frameon=True)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segment Share (%) Over Time

# COMMAND ----------

pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(13, 5))
for col in pivot_pct.columns:
    ax.plot(pivot_pct.index, pivot_pct[col], marker="o", linewidth=2, label=col)
ax.set_title("Segment Share (%) Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Reference Date")
ax.set_ylabel("Share (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="best", fontsize=9, frameon=True)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Transition Analysis
# MAGIC
# MAGIC Using window functions to compute 1-month and 3-month lagged segments per customer.

# COMMAND ----------

w_cust = Window.partitionBy("customer_id").orderBy("reference_date")

transitions_raw = (
    all_scored
    .select("customer_id", "reference_date", "cluster_labels", "segment_label")
    .withColumn("cluster_lag1", F.lag("cluster_labels", 1).over(w_cust))
    .withColumn("cluster_lag3", F.lag("cluster_labels", 3).over(w_cust))
    .withColumn("segment_lag1", F.lag("segment_label", 1).over(w_cust))
    .withColumn("segment_lag3", F.lag("segment_label", 3).over(w_cust))
)

print(f"Total scored records: {transitions_raw.count():,}")
print(f"Records with 1-month lag: {transitions_raw.filter(F.col('cluster_lag1').isNotNull()).count():,}")
print(f"Records with 3-month lag: {transitions_raw.filter(F.col('cluster_lag3').isNotNull()).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. 1-Month Transition Matrix

# COMMAND ----------

trans1 = transitions_raw.filter(F.col("cluster_lag1").isNotNull())

transitions_month = (
    trans1
    .groupBy(
        F.col("segment_lag1").alias("from_segment"),
        F.col("segment_label").alias("to_segment"),
    )
    .agg(F.count("*").alias("count"))
)

w_from = Window.partitionBy("from_segment")
transitions_month = (
    transitions_month
    .withColumn("total_from", F.sum("count").over(w_from))
    .withColumn("transition_probability", F.round(F.col("count") / F.col("total_from"), 4))
    .select("from_segment", "to_segment", "count", "transition_probability")
    .orderBy("from_segment", "to_segment")
)

transitions_month.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_06_transitions_month"
)

print(f"Saved ai_06_transitions_month: {transitions_month.count()} rows")
display(transitions_month)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-Month Transition Heatmap

# COMMAND ----------

tm_pdf = transitions_month.toPandas()
tm_matrix = tm_pdf.pivot_table(
    index="from_segment", columns="to_segment",
    values="transition_probability", aggfunc="sum"
).fillna(0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    tm_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, vmin=0, vmax=1, ax=ax,
    cbar_kws={"label": "Probability"}
)
ax.set_title("1-Month Transition Probability Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("To Segment")
ax.set_ylabel("From Segment")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. 3-Month Transition Matrix

# COMMAND ----------

trans3 = transitions_raw.filter(F.col("cluster_lag3").isNotNull())

transitions_quarter = (
    trans3
    .groupBy(
        F.col("segment_lag3").alias("from_segment"),
        F.col("segment_label").alias("to_segment"),
    )
    .agg(F.count("*").alias("count"))
)

w_from3 = Window.partitionBy("from_segment")
transitions_quarter = (
    transitions_quarter
    .withColumn("total_from", F.sum("count").over(w_from3))
    .withColumn("transition_probability", F.round(F.col("count") / F.col("total_from"), 4))
    .select("from_segment", "to_segment", "count", "transition_probability")
    .orderBy("from_segment", "to_segment")
)

transitions_quarter.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_06_transitions_quarter"
)

print(f"Saved ai_06_transitions_quarter: {transitions_quarter.count()} rows")
display(transitions_quarter)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-Month Transition Heatmap

# COMMAND ----------

tq_pdf = transitions_quarter.toPandas()
tq_matrix = tq_pdf.pivot_table(
    index="from_segment", columns="to_segment",
    values="transition_probability", aggfunc="sum"
).fillna(0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    tq_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, vmin=0, vmax=1, ax=ax,
    cbar_kws={"label": "Probability"}
)
ax.set_title("3-Month Transition Probability Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("To Segment")
ax.set_ylabel("From Segment")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Segment Stability (Retention Rates)

# COMMAND ----------

stability_month = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_06_transitions_month")
    .filter(F.col("from_segment") == F.col("to_segment"))
    .select(
        F.col("from_segment").alias("segment"),
        F.col("transition_probability").alias("retention_rate"),
    )
    .withColumn("window", F.lit("1-month"))
)

stability_quarter = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_06_transitions_quarter")
    .filter(F.col("from_segment") == F.col("to_segment"))
    .select(
        F.col("from_segment").alias("segment"),
        F.col("transition_probability").alias("retention_rate"),
    )
    .withColumn("window", F.lit("3-month"))
)

segment_stability = stability_month.union(stability_quarter)
segment_stability.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_06_segment_stability"
)

stab_pdf = segment_stability.orderBy("segment", "window").toPandas()
print("=" * 55)
print("  SEGMENT STABILITY -- RETENTION RATES")
print("=" * 55)
for _, row in stab_pdf.iterrows():
    print(f"  {row['segment']:30s}  {row['window']:8s}  {row['retention_rate']:.2%}")
print("=" * 55)

# Identify most/least stable
month_stab = stab_pdf[stab_pdf["window"] == "1-month"].sort_values("retention_rate", ascending=False)
if len(month_stab) > 0:
    most_stable = month_stab.iloc[0]
    least_stable = month_stab.iloc[-1]
    print(f"\n  Most stable segment (1-month):   {most_stable['segment']} ({most_stable['retention_rate']:.2%})")
    print(f"  Least stable segment (1-month):  {least_stable['segment']} ({least_stable['retention_rate']:.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retention Rate Comparison (1-Month vs 3-Month)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 5))
segments_sorted = stab_pdf[stab_pdf["window"] == "1-month"].sort_values("retention_rate", ascending=False)["segment"].tolist()

x = np.arange(len(segments_sorted))
bar_w = 0.35

month_vals = []
quarter_vals = []
for seg in segments_sorted:
    m = stab_pdf[(stab_pdf["segment"] == seg) & (stab_pdf["window"] == "1-month")]["retention_rate"]
    q = stab_pdf[(stab_pdf["segment"] == seg) & (stab_pdf["window"] == "3-month")]["retention_rate"]
    month_vals.append(float(m.iloc[0]) if len(m) > 0 else 0)
    quarter_vals.append(float(q.iloc[0]) if len(q) > 0 else 0)

bars1 = ax.bar(x - bar_w / 2, month_vals, bar_w, label="1-Month", color="#4C9BE8")
bars2 = ax.bar(x + bar_w / 2, quarter_vals, bar_w, label="3-Month", color="#E8784C")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

ax.set_title("Segment Retention Rates: 1-Month vs 3-Month", fontsize=14, fontweight="bold")
ax.set_ylabel("Retention Rate")
ax.set_xticks(x)
ax.set_xticklabels(segments_sorted, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(frameon=True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Segment Migration Patterns
# MAGIC
# MAGIC For each segment, where do customers come **from** (inflow) and where do they **go** (outflow)?

# COMMAND ----------

# --- Inflow / Outflow analysis using 1-month transitions ---
print("=" * 70)
print("  SEGMENT MIGRATION PATTERNS (1-Month)")
print("=" * 70)

for seg in sorted(tm_pdf["from_segment"].unique()):
    print(f"\n{'=' * 50}")
    print(f"  Segment: {seg}")
    print(f"{'=' * 50}")

    # Outflow: where customers FROM this segment go
    outflow = tm_pdf[tm_pdf["from_segment"] == seg].sort_values("transition_probability", ascending=False)
    print(f"\n  OUTFLOW (where {seg} customers go):")
    for _, row in outflow.iterrows():
        marker = " <-- retention" if row["from_segment"] == row["to_segment"] else ""
        print(f"    -> {row['to_segment']:30s}  {row['transition_probability']:.2%}  ({int(row['count']):,} customers){marker}")

    # Inflow: where customers TO this segment come from
    inflow = tm_pdf[tm_pdf["to_segment"] == seg].sort_values("transition_probability", ascending=False)
    print(f"\n  INFLOW (where {seg} customers come from):")
    for _, row in inflow.iterrows():
        marker = " <-- retention" if row["from_segment"] == row["to_segment"] else ""
        print(f"    <- {row['from_segment']:30s}  {row['transition_probability']:.2%}  ({int(row['count']):,} customers){marker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top Migration Paths (Grouped Bar Chart)

# COMMAND ----------

# Exclude retention (diagonal) to focus on actual migrations
migration_paths = tm_pdf[tm_pdf["from_segment"] != tm_pdf["to_segment"]].copy()
migration_paths = migration_paths.sort_values("count", ascending=False).head(15)
migration_paths["path"] = migration_paths["from_segment"] + " -> " + migration_paths["to_segment"]

fig, ax = plt.subplots(figsize=(13, 6))
colors = sns.color_palette("Set2", n_colors=len(migration_paths))
bars = ax.barh(
    migration_paths["path"][::-1],
    migration_paths["count"][::-1],
    color=colors,
)

for bar, prob in zip(bars, migration_paths["transition_probability"][::-1]):
    ax.text(
        bar.get_width() + max(migration_paths["count"]) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{prob:.1%}",
        va="center", fontsize=9,
    )

ax.set_title("Top Migration Paths (1-Month, Excluding Retention)", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Customers")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Summary

# COMMAND ----------

print("=" * 70)
print("  MODEL VALIDATION SUMMARY")
print("=" * 70)

# Most / least stable
month_retention = stab_pdf[stab_pdf["window"] == "1-month"].sort_values("retention_rate", ascending=False)
quarter_retention = stab_pdf[stab_pdf["window"] == "3-month"].sort_values("retention_rate", ascending=False)

if len(month_retention) > 0:
    most_stable_1m = month_retention.iloc[0]
    least_stable_1m = month_retention.iloc[-1]
    avg_retention_1m = month_retention["retention_rate"].mean()

    print(f"\n  1-MONTH RETENTION")
    print(f"    Most stable segment:   {most_stable_1m['segment']}  ({most_stable_1m['retention_rate']:.2%})")
    print(f"    Least stable segment:  {least_stable_1m['segment']}  ({least_stable_1m['retention_rate']:.2%})")
    print(f"    Average retention:     {avg_retention_1m:.2%}")

if len(quarter_retention) > 0:
    most_stable_3m = quarter_retention.iloc[0]
    least_stable_3m = quarter_retention.iloc[-1]
    avg_retention_3m = quarter_retention["retention_rate"].mean()

    print(f"\n  3-MONTH RETENTION")
    print(f"    Most stable segment:   {most_stable_3m['segment']}  ({most_stable_3m['retention_rate']:.2%})")
    print(f"    Least stable segment:  {least_stable_3m['segment']}  ({least_stable_3m['retention_rate']:.2%})")
    print(f"    Average retention:     {avg_retention_3m:.2%}")

# Top migration paths
print(f"\n  TOP MIGRATION PATHS (1-Month)")
top_migrations = tm_pdf[tm_pdf["from_segment"] != tm_pdf["to_segment"]].sort_values("count", ascending=False).head(5)
for _, row in top_migrations.iterrows():
    print(f"    {row['from_segment']} -> {row['to_segment']}:  {row['transition_probability']:.2%} ({int(row['count']):,} customers)")

# Overall stability assessment
if len(month_retention) > 0:
    print(f"\n  OVERALL MODEL STABILITY ASSESSMENT")
    if avg_retention_1m >= 0.80:
        assessment = "STRONG -- Average 1-month retention >= 80%. Segments are highly stable."
    elif avg_retention_1m >= 0.60:
        assessment = "MODERATE -- Average 1-month retention 60-80%. Segments are reasonably stable with some migration."
    else:
        assessment = "WEAK -- Average 1-month retention < 60%. Segments show significant churn; consider model revision."
    print(f"    {assessment}")

print("\n" + "=" * 70)
print("  Tables saved:")
print(f"    {CATALOG}.{SCHEMA}.ai_06_historical_segments")
print(f"    {CATALOG}.{SCHEMA}.ai_06_transitions_month")
print(f"    {CATALOG}.{SCHEMA}.ai_06_transitions_quarter")
print(f"    {CATALOG}.{SCHEMA}.ai_06_segment_stability")
print("=" * 70)
