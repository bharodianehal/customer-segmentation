# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Campaign Targeting & AI-Generated Copy
# MAGIC **Catalog:** `main.banking`
# MAGIC
# MAGIC Combines segments + propensity scores to generate targeted promotion lists,
# MAGIC then uses `ai_query()` to create personalized campaign copy per segment.

# COMMAND ----------

CATALOG = "main"
SCHEMA = "banking"

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Segments + Propensity Scores

# COMMAND ----------

df_scored = spark.table(f"{CATALOG}.{SCHEMA}.05_customer_reference_scored")
df_propensity = spark.table(f"{CATALOG}.{SCHEMA}.08_customer_propensity_scores")

print(f"Scored customers: {df_scored.count():,}")
print(f"Propensity scores: {df_propensity.count():,}")

# COMMAND ----------

# Join segments with propensity
df_combined = (
    df_scored.select("customer_id", "segment_label", "income", "region", "tier")
    .join(df_propensity.select("customer_id", "propensity_score", "propensity_decile", "predicted_responder"), "customer_id", "left")
    .fillna({"propensity_score": 0.0, "propensity_decile": 1, "predicted_responder": 0})
)

print(f"Combined: {df_combined.count():,} customers")
print(f"\nCustomers per segment:")
df_combined.groupBy("segment_label").count().orderBy("segment_label").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Promotion Strategies

# COMMAND ----------

SEGMENT_STRATEGIES = {
    "Premium High-Value": {
        "campaign_type": "VIP Exclusive",
        "description": "Exclusive wealth management, private banking, premium rewards",
        "priority": 1,
        "expected_response": 0.35
    },
    "Digital-First Engaged": {
        "campaign_type": "Digital Upsell",
        "description": "Mobile-first offers, digital product bundles, fintech features",
        "priority": 2,
        "expected_response": 0.28
    },
    "Mainstream Active": {
        "campaign_type": "Cross-Sell Bundle",
        "description": "Product bundles, loyalty rewards, savings accounts",
        "priority": 3,
        "expected_response": 0.18
    },
    "Low Engagement At-Risk": {
        "campaign_type": "Retention Offer",
        "description": "Fee waivers, rate improvements, win-back incentives",
        "priority": 1,
        "expected_response": 0.12
    }
}

for seg, strategy in SEGMENT_STRATEGIES.items():
    print(f"  {seg}: {strategy['campaign_type']} (expected response: {strategy['expected_response']:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter to Top Propensity Deciles per Segment

# COMMAND ----------

# Target deciles 7-10 (top 40% propensity) for each segment
df_targets = (
    df_combined
    .filter(F.col("propensity_decile") >= 7)
    .withColumn("priority_rank",
        F.row_number().over(
            Window.partitionBy("segment_label").orderBy(F.desc("propensity_score"))
        ))
)

# Add campaign type
segment_campaign_map = {seg: info["campaign_type"] for seg, info in SEGMENT_STRATEGIES.items()}
mapping_expr = F.create_map([F.lit(x) for pair in segment_campaign_map.items() for x in pair])
df_targets = df_targets.withColumn("campaign_type", mapping_expr[F.col("segment_label")])
df_targets = df_targets.fillna({"campaign_type": "General Promotion"})

print(f"Total targeted customers: {df_targets.count():,}")
print(f"\nTargeted by segment:")
df_targets.groupBy("segment_label", "campaign_type").count().orderBy("segment_label").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save Campaign Targets

# COMMAND ----------

df_targets_save = df_targets.select(
    "customer_id", "segment_label", "propensity_score", "propensity_decile",
    "campaign_type", "priority_rank", "income", "region", "tier"
)

df_targets_save.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.09_campaign_targets"
)
print(f"Saved 09_campaign_targets: {spark.table(f'{CATALOG}.{SCHEMA}.09_campaign_targets').count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate AI Campaign Copy per Segment

# COMMAND ----------

campaign_rows = []
for segment_name, strategy in SEGMENT_STRATEGIES.items():
    prompt = f"""You are a marketing copywriter for a regional bank. Write campaign copy for the "{segment_name}" customer segment.

Campaign type: {strategy['campaign_type']}
Strategy: {strategy['description']}

Generate EXACTLY this JSON format (no markdown, no code blocks):
{{"email_subject": "...", "email_body": "2-3 sentence email body", "sms_message": "Under 160 chars SMS", "recommended_offer": "Specific offer description"}}"""

    try:
        result = spark.sql(f"""
            SELECT ai_query('databricks-meta-llama-3-3-70b-instruct', '{prompt.replace("'", "''")}') as response
        """).collect()[0][0]

        import json as _json
        try:
            parsed = _json.loads(result)
        except:
            parsed = {
                "email_subject": f"{strategy['campaign_type']} - Exclusive Offer",
                "email_body": f"As a valued {segment_name} customer, we have a special offer for you. {strategy['description']}.",
                "sms_message": f"{strategy['campaign_type']}: Special offer just for you. Reply YES to learn more.",
                "recommended_offer": strategy['description']
            }

        for channel in ["email", "sms"]:
            campaign_rows.append({
                "segment": segment_name,
                "channel": channel,
                "campaign_type": strategy["campaign_type"],
                "subject_line": parsed.get("email_subject", ""),
                "body_text": parsed.get("email_body", "") if channel == "email" else parsed.get("sms_message", ""),
                "recommended_offer": parsed.get("recommended_offer", ""),
                "expected_response_rate": strategy["expected_response"]
            })
        print(f"  Generated copy for: {segment_name}")
    except Exception as e:
        print(f"  ai_query failed for {segment_name}: {e}")
        for channel in ["email", "sms"]:
            campaign_rows.append({
                "segment": segment_name,
                "channel": channel,
                "campaign_type": strategy["campaign_type"],
                "subject_line": f"{strategy['campaign_type']} - Special Offer",
                "body_text": strategy["description"],
                "recommended_offer": strategy["description"],
                "expected_response_rate": strategy["expected_response"]
            })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Campaign Copy

# COMMAND ----------

import pandas as pd
df_copy = spark.createDataFrame(pd.DataFrame(campaign_rows))
df_copy.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.09_campaign_copy"
)
print(f"Saved 09_campaign_copy: {spark.table(f'{CATALOG}.{SCHEMA}.09_campaign_copy').count():,} rows")
display(spark.table(f"{CATALOG}.{SCHEMA}.09_campaign_copy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Campaign Summary

# COMMAND ----------

total_targeted = spark.table(f"{CATALOG}.{SCHEMA}.09_campaign_targets").count()
total_customers = df_scored.count()

print("=" * 70)
print("  CAMPAIGN TARGETING COMPLETE")
print("=" * 70)
print(f"  Total customers:    {total_customers:,}")
print(f"  Targeted customers: {total_targeted:,} ({100*total_targeted/total_customers:.1f}%)")
print(f"  Segments covered:   {len(SEGMENT_STRATEGIES)}")
print()

for seg, strategy in SEGMENT_STRATEGIES.items():
    seg_count = df_targets.filter(F.col("segment_label") == seg).count()
    print(f"  {seg}:")
    print(f"    Campaign: {strategy['campaign_type']}")
    print(f"    Targeted: {seg_count:,}")
    print(f"    Expected response: {strategy['expected_response']:.0%}")
    print()

print("=" * 70)
