# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Segment Activation
# MAGIC **Catalog:** `main.main`
# MAGIC
# MAGIC Actionable outputs from the segmentation:
# MAGIC 1. Product mix analysis → `07_product_mix`
# MAGIC 2. Cross-sell analysis → `07_crosssell_summary`, `07_crosssell_by_product`
# MAGIC 3. Segment value analysis
# MAGIC 4. Next-Best-Action recommendations → `07_segment_nba`

# COMMAND ----------

# MAGIC %pip install matplotlib seaborn --quiet

# COMMAND ----------

CATALOG = "main"
SCHEMA = "banking"

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Product Mix Analysis

# COMMAND ----------

# Load scored customer reference and product holdings
ref_scored = spark.table(f"{CATALOG}.{SCHEMA}.05_customer_reference_scored")
ref_labels = ref_scored.select("customer_id", "tier", "cluster_labels", "segment_label", "reference_date")

prod_holdings = spark.table(f"{CATALOG}.{SCHEMA}.01_raw_product_holdings")

# Build product portfolio fingerprints — which product types each customer holds
product_portfolios = (
    prod_holdings
    .select("customer_id", "product_type")
    .distinct()
    .groupBy("customer_id")
    .agg(
        F.sort_array(F.collect_set("product_type")).alias("product_types"),
        F.countDistinct("product_type").alias("product_type_count"),
    )
    .withColumn("product_portfolio", F.array_join("product_types", ", "))
    .drop("product_types")
)

product_mix = (
    ref_labels
    .join(product_portfolios, "customer_id", "left")
    .groupBy("cluster_labels", "segment_label", "product_portfolio", "tier")
    .agg(F.count("*").alias("customer_count"))
    .orderBy("cluster_labels", F.desc("customer_count"))
)

product_mix.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.07_product_mix"
)
mix_count = product_mix.count()
print(f"07_product_mix: {mix_count:,} rows saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Type Penetration by Segment (Stacked Bar)

# COMMAND ----------

# Compute penetration: for each segment, what % of customers hold each product type
cust_segment = ref_labels.select("customer_id", "segment_label").distinct()
cust_products = prod_holdings.select("customer_id", "product_type").distinct()

penetration_df = (
    cust_segment.join(cust_products, "customer_id", "inner")
    .groupBy("segment_label", "product_type")
    .agg(F.countDistinct("customer_id").alias("holders"))
)

segment_sizes = (
    cust_segment.groupBy("segment_label")
    .agg(F.countDistinct("customer_id").alias("segment_size"))
)

penetration_df = (
    penetration_df.join(segment_sizes, "segment_label")
    .withColumn("penetration_pct", F.round(F.col("holders") / F.col("segment_size") * 100, 1))
)

pen_pd = penetration_df.select("segment_label", "product_type", "penetration_pct").toPandas()

# Pivot for stacked bar
pivot = pen_pd.pivot(index="segment_label", columns="product_type", values="penetration_pct").fillna(0)
pivot = pivot.sort_index()

fig, ax = plt.subplots(figsize=(14, 6))
pivot.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Penetration (%)")
ax.set_xlabel("")
ax.set_title("Product Type Penetration by Segment", fontsize=14, fontweight="bold")
ax.legend(title="Product Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Combination Frequency Heatmap by Segment

# COMMAND ----------

# Build co-occurrence matrix: for each segment, how often pairs of product types appear together
cust_prod_pairs = cust_products.alias("a").join(
    cust_products.alias("b"),
    (F.col("a.customer_id") == F.col("b.customer_id")),
).select(
    F.col("a.customer_id"),
    F.col("a.product_type").alias("product_a"),
    F.col("b.product_type").alias("product_b"),
).filter(F.col("product_a") <= F.col("product_b"))

combo_by_segment = (
    cust_prod_pairs
    .join(cust_segment, cust_prod_pairs["customer_id"] == cust_segment["customer_id"])
    .drop(cust_segment["customer_id"])
    .groupBy("segment_label", "product_a", "product_b")
    .agg(F.countDistinct("a.customer_id").alias("combo_count"))
)

combo_pd = combo_by_segment.toPandas()
segments = sorted(combo_pd["segment_label"].unique())
n_seg = len(segments)

fig, axes = plt.subplots(1, n_seg, figsize=(5 * n_seg, 5), sharey=True)
if n_seg == 1:
    axes = [axes]

for ax, seg in zip(axes, segments):
    seg_data = combo_pd[combo_pd["segment_label"] == seg]
    # Build symmetric matrix
    all_prods = sorted(set(seg_data["product_a"]) | set(seg_data["product_b"]))
    mat = pd.DataFrame(0, index=all_prods, columns=all_prods)
    for _, row in seg_data.iterrows():
        mat.loc[row["product_a"], row["product_b"]] = row["combo_count"]
        mat.loc[row["product_b"], row["product_a"]] = row["combo_count"]
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=False, linewidths=0.5)
    short_label = seg.replace(" ", "\n") if len(seg) > 18 else seg
    ax.set_title(short_label, fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

fig.suptitle("Product Combination Frequency by Segment", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cross-Sell Analysis

# COMMAND ----------

cust_rev = spark.table(f"{CATALOG}.{SCHEMA}.01_raw_revenues").withColumn("date", F.to_date("date"))
cust_prod_tbl = spark.table(f"{CATALOG}.{SCHEMA}.01_raw_products")

# Revenue per customer per product type
customer_product_rev = (
    cust_rev.join(cust_prod_tbl, "product_id", "left")
    .groupBy("customer_id", "product_type")
    .agg(F.avg("revenue").alias("avg_revenue"))
    .filter(F.col("avg_revenue") > 0)
)

# Cross-sell breadth per customer
xsell_per_customer = (
    customer_product_rev.groupBy("customer_id")
    .agg(
        F.countDistinct("product_type").alias("xsell_breadth"),
        F.sum("avg_revenue").alias("total_avg_revenue"),
    )
)

ref_labels_simple = ref_scored.select("customer_id", "tier", "cluster_labels", "segment_label")
xsell_enriched = xsell_per_customer.join(ref_labels_simple, "customer_id", "inner")

# Summary: segment x tier
crosssell_summary = (
    xsell_enriched.groupBy("tier", "segment_label")
    .agg(
        F.count("*").alias("customer_count"),
        F.sum("total_avg_revenue").alias("total_revenue"),
        F.avg("total_avg_revenue").alias("avg_revenue"),
        F.avg("xsell_breadth").alias("avg_xsell"),
    )
)
crosssell_summary.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.07_crosssell_summary"
)

# Revenue by segment x tier x product_type
crosssell_by_product = (
    customer_product_rev.join(ref_labels_simple, "customer_id", "inner")
    .groupBy("segment_label", "tier", "product_type")
    .agg(F.sum("avg_revenue").alias("total_revenue"))
)
crosssell_by_product.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.07_crosssell_by_product"
)

print(f"07_crosssell_summary:    {crosssell_summary.count():,} rows saved")
print(f"07_crosssell_by_product: {crosssell_by_product.count():,} rows saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Avg Cross-Sell Breadth by Segment and Tier

# COMMAND ----------

cs_pd = crosssell_summary.toPandas()
cs_pd["label"] = cs_pd["segment_label"] + " — " + cs_pd["tier"]
cs_pd = cs_pd.sort_values("avg_xsell", ascending=True)

fig, ax = plt.subplots(figsize=(12, max(5, len(cs_pd) * 0.45)))
colors = sns.color_palette("viridis", n_colors=len(cs_pd))
bars = ax.barh(cs_pd["label"], cs_pd["avg_xsell"], color=colors, edgecolor="white")
for bar, val in zip(bars, cs_pd["avg_xsell"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8)
ax.set_xlabel("Avg # Distinct Product Types Held")
ax.set_title("Average Cross-Sell Breadth by Segment and Tier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Total Revenue by Segment and Product Type

# COMMAND ----------

cbp_pd = crosssell_by_product.toPandas()
rev_pivot = cbp_pd.groupby(["segment_label", "product_type"])["total_revenue"].sum().reset_index()
rev_pivot_wide = rev_pivot.pivot(index="segment_label", columns="product_type", values="total_revenue").fillna(0)
rev_pivot_wide = rev_pivot_wide.loc[rev_pivot_wide.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(14, 6))
rev_pivot_wide.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Total Revenue")
ax.set_xlabel("")
ax.set_title("Total Revenue by Segment and Product Type", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(title="Product Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-Sell Opportunity Gaps

# COMMAND ----------

# Identify under-penetrated product types per segment
all_product_types = sorted(pen_pd["product_type"].unique())
all_segments_list = sorted(pen_pd["segment_label"].unique())

print("=" * 80)
print("CROSS-SELL OPPORTUNITY SUMMARY")
print("=" * 80)

for seg in all_segments_list:
    seg_pen = pen_pd[pen_pd["segment_label"] == seg].set_index("product_type")["penetration_pct"]
    overall_pen = pen_pd.groupby("product_type")["penetration_pct"].mean()
    gaps = []
    for pt in all_product_types:
        seg_val = seg_pen.get(pt, 0)
        avg_val = overall_pen.get(pt, 0)
        if seg_val < avg_val * 0.8:  # under-penetrated: more than 20% below average
            gaps.append((pt, seg_val, avg_val))
    print(f"\n  {seg}")
    if gaps:
        for pt, sv, av in sorted(gaps, key=lambda x: x[1]):
            print(f"    - {pt:20s}  segment: {sv:5.1f}%  |  avg: {av:5.1f}%  (gap: {av - sv:+.1f}pp)")
    else:
        print("    No significant gaps — well-diversified portfolio")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Segment Value Analysis

# COMMAND ----------

# For each segment: total revenue, avg revenue per customer, customer count
segment_value = (
    xsell_enriched.groupBy("segment_label")
    .agg(
        F.count("*").alias("customer_count"),
        F.sum("total_avg_revenue").alias("total_revenue"),
        F.avg("total_avg_revenue").alias("avg_revenue"),
    )
    .orderBy(F.desc("total_revenue"))
)

sv_pd = segment_value.toPandas()

# Print ranking
print("=" * 80)
print("SEGMENT VALUE RANKING (by Total Revenue)")
print("=" * 80)
for i, row in sv_pd.iterrows():
    rank = i + 1
    print(f"  #{rank}  {row['segment_label']:35s}  |  Customers: {row['customer_count']:>6,}  |  "
          f"Avg Rev: ${row['avg_revenue']:>10,.2f}  |  Total Rev: ${row['total_revenue']:>14,.2f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bubble Chart: Segment Value Overview

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 7))

# Normalize bubble sizes for visibility
max_rev = sv_pd["total_revenue"].max()
bubble_sizes = (sv_pd["total_revenue"] / max_rev) * 3000 + 200

cmap = plt.cm.get_cmap("Set2", len(sv_pd))
colors_list = [cmap(i) for i in range(len(sv_pd))]

for i, row in sv_pd.iterrows():
    ax.scatter(
        row["customer_count"], row["avg_revenue"],
        s=bubble_sizes.iloc[i], alpha=0.7, color=colors_list[i],
        edgecolors="black", linewidth=0.8, zorder=3,
    )
    ax.annotate(
        row["segment_label"], (row["customer_count"], row["avg_revenue"]),
        textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

ax.set_xlabel("Customer Count", fontsize=11)
ax.set_ylabel("Avg Revenue per Customer ($)", fontsize=11)
ax.set_title("Segment Value Overview\n(bubble size = total revenue)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Next-Best-Action Recommendations

# COMMAND ----------

nba_rules = {
    "Premium High-Value": {
        "action": "Wealth Management Upgrade",
        "detail": "Dedicated relationship manager; wealth advisory; securities portfolio review; premium credit card upgrade",
        "channel": "Relationship Manager + In-Branch",
        "priority": "Critical",
        "risk_assessment": "Low churn risk, highest lifetime value",
    },
    "Digital-First Engaged": {
        "action": "Digital Product Expansion",
        "detail": "Mobile-first products: instant loans, digital securities, budgeting tools; beta-test new features",
        "channel": "App Push + Email",
        "priority": "High",
        "risk_assessment": "Low risk, high digital engagement",
    },
    "Low Engagement At-Risk": {
        "action": "Win-Back Campaign",
        "detail": "Fee waiver for 6 months; simplified product bundle; satisfaction survey; proactive outreach",
        "channel": "Phone + Email + SMS",
        "priority": "High",
        "risk_assessment": "High churn risk — act within 30 days",
    },
    "Multi-Product Middle Market": {
        "action": "Cross-sell & Deepen",
        "detail": "Credit review for secured lending; competitive savings rate; bundle pricing discount",
        "channel": "Email + App",
        "priority": "Medium",
        "risk_assessment": "Stable segment — growth opportunity",
    },
    "Mainstream Active": {
        "action": "Engagement & Loyalty",
        "detail": "Loyalty rewards; cashback on debit purchases; financial health check-up; digital migration nudge",
        "channel": "App + Email",
        "priority": "Medium",
        "risk_assessment": "Moderate — keep engaged with consistent value",
    },
}

# Get segment counts from scored data
ref_pd = ref_scored.select("segment_label").toPandas()

nba_rows = []
for seg_label, rules in nba_rules.items():
    matching = ref_pd[ref_pd["segment_label"].str.startswith(seg_label)]
    nba_rows.append({
        "segment_label": seg_label,
        "customer_count": int(matching.shape[0]),
        "recommended_action": rules["action"],
        "action_detail": rules["detail"],
        "channel": rules["channel"],
        "priority": rules["priority"],
        "risk_assessment": rules["risk_assessment"],
    })

nba_df = spark.createDataFrame(nba_rows)
nba_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.07_segment_nba"
)
print(f"07_segment_nba: {len(nba_rows)} rows saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next-Best-Action Summary

# COMMAND ----------

nba_display = pd.DataFrame(nba_rows)

# Priority color mapping
priority_colors = {"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#1976d2"}

fig, ax = plt.subplots(figsize=(18, len(nba_display) * 1.2 + 1.5))
ax.axis("off")

columns = ["Segment", "Customers", "Action", "Channel", "Priority", "Risk Assessment"]
col_widths = [0.18, 0.08, 0.16, 0.16, 0.08, 0.34]

table_data = []
cell_colors = []
for _, row in nba_display.iterrows():
    table_data.append([
        row["segment_label"],
        f"{row['customer_count']:,}",
        row["recommended_action"],
        row["channel"],
        row["priority"],
        row["risk_assessment"],
    ])
    p_color = priority_colors.get(row["priority"], "#e0e0e0")
    # Light tint for row background based on priority
    alpha = 0.12
    rgb = tuple(int(p_color.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    bg = tuple(c * alpha + 1 * (1 - alpha) for c in rgb)
    cell_colors.append([bg] * len(columns))

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    colWidths=col_widths,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

# Style header row
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor("#263238")
    cell.set_text_props(color="white", fontweight="bold")

# Style data rows
for i in range(len(table_data)):
    for j in range(len(columns)):
        cell = table[i + 1, j]
        cell.set_facecolor(cell_colors[i][j])
        if j == 4:  # Priority column — bold
            cell.set_text_props(fontweight="bold")
        if j == 0:  # Segment column — bold
            cell.set_text_props(fontweight="bold")
        if j in (3, 5):  # Wider text columns — left align
            cell.set_text_props(ha="left")

ax.set_title("Next-Best-Action Recommendations by Segment",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verification

# COMMAND ----------

print("=" * 70)
print(f"CUSTOMER SEGMENTATION PIPELINE — {CATALOG}.{SCHEMA}")
print("=" * 70)

all_tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
pipeline_tables = [t for t in all_tables if not t[1].startswith("_tmp_")]
total_rows = 0

for t in sorted(pipeline_tables, key=lambda x: x[1]):
    count = spark.sql(f"SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.{t[1]}").collect()[0][0]
    total_rows += count
    print(f"  {t[1]:45s} {count:>10,} rows")

print("-" * 70)
print(f"  Total tables: {len(pipeline_tables)}")
print(f"  Total rows:   {total_rows:,}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline Summary

# COMMAND ----------

print("=" * 70)
print("SEGMENT ACTIVATION COMPLETE")
print("=" * 70)
print()
print("  Tables produced in this notebook:")
print(f"    07_product_mix          — Product portfolio fingerprints by segment")
print(f"    07_crosssell_summary    — Cross-sell breadth and revenue by segment x tier")
print(f"    07_crosssell_by_product — Revenue breakdown by segment x product type")
print(f"    07_segment_nba          — Next-Best-Action recommendations per segment")
print()
print("  Key analyses:")
print("    1. Product penetration rates and combination heatmaps")
print("    2. Cross-sell breadth and revenue gaps")
print("    3. Segment value ranking (bubble chart)")
print("    4. NBA recommendations with priority and channel guidance")
print()
print("=" * 70)
