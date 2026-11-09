# Databricks notebook source
# MAGIC %md
# MAGIC # 03b — Feature Derivation: Products
# MAGIC **Catalog:** `customer_segmentation.ai_banking`
# MAGIC
# MAGIC This notebook derives product portfolio features from raw product holdings.
# MAGIC
# MAGIC ### Features Derived
# MAGIC | Feature | Logic | Type | Purpose |
# MAGIC |---------|-------|------|---------|
# MAGIC | `nb_subscribed_products` | COUNT(*) where start_date <= ref_date | Aggregation | Total products ever subscribed |
# MAGIC | `nb_terminated_products` | COUNT(*) where end_date IS NOT NULL | Aggregation | Total products terminated |
# MAGIC | `nb_active_products` | subscribed - terminated | Aggregation | Currently active products |
# MAGIC | `product_diversity` | COUNT(DISTINCT product_type) where active | Aggregation | Distinct product categories held |
# MAGIC | `product_churn_rate` | terminated / subscribed | Derived Ratio | Attrition risk indicator |
# MAGIC | `net_product_growth` | active - terminated | Composite | Engagement trajectory |
# MAGIC
# MAGIC ### Output Table
# MAGIC `ai_03_products_features` — one row per customer with product portfolio metrics

# COMMAND ----------

CATALOG = "customer_segmentation"
SCHEMA = "ai_banking"

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Product Holdings

# COMMAND ----------

raw_holdings = (
    spark.table(f"{CATALOG}.{SCHEMA}.ai_01_raw_product_holdings")
    .withColumn("start_date", F.to_date(F.col("start_date")))
    .withColumn("end_date", F.to_date(F.col("end_date")))
)

reference_date = raw_holdings.agg(F.max("start_date")).collect()[0][0]
print(f"Product holdings: {raw_holdings.count():,} rows")
print(f"Reference date: {reference_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Derive Aggregation Features

# COMMAND ----------

# Filter to products that existed by reference date
holdings_current = raw_holdings.filter(F.col("start_date") <= F.lit(reference_date))

# Flag active vs terminated
holdings_flagged = (
    holdings_current
    .withColumn("is_active",
        F.when(F.col("end_date").isNull(), F.lit(1))
         .when(F.col("end_date") > F.lit(reference_date), F.lit(1))
         .otherwise(F.lit(0)))
)

# Aggregate per customer
products_features = (
    holdings_flagged
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("nb_subscribed_products"),
        F.sum(F.when(F.col("is_active") == 0, 1).otherwise(0)).alias("nb_terminated_products"),
        F.sum(F.col("is_active")).alias("nb_active_products"),
        F.countDistinct(
            F.when(F.col("is_active") == 1, F.col("product_type"))
        ).alias("product_diversity"),
    )
)

print(f"Base product features: {products_features.count():,} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Derived Ratio: Product Churn Rate
# MAGIC
# MAGIC `product_churn_rate = nb_terminated / nb_subscribed` — Attrition risk indicator

# COMMAND ----------

products_features = products_features.withColumn(
    "product_churn_rate",
    F.when(F.col("nb_subscribed_products") > 0,
           F.col("nb_terminated_products") / F.col("nb_subscribed_products"))
    .otherwise(0.0)
)

print("Applied: product_churn_rate = nb_terminated / nb_subscribed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Composite: Net Product Growth
# MAGIC
# MAGIC `net_product_growth = active - terminated` — Engagement trajectory

# COMMAND ----------

products_features = products_features.withColumn(
    "net_product_growth",
    F.col("nb_active_products") - F.col("nb_terminated_products")
)

print("Applied: net_product_growth = nb_active_products - nb_terminated_products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary & Save

# COMMAND ----------

print(f"Products features: {products_features.count():,} customers x {len(products_features.columns)} cols")
display(products_features.limit(5))

# COMMAND ----------

products_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ai_03_products_features"
)

saved = spark.table(f"{CATALOG}.{SCHEMA}.ai_03_products_features").count()
print(f"Saved ai_03_products_features: {saved:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC **Output:** `ai_03_products_features` — `customer_id, nb_subscribed_products, nb_terminated_products, nb_active_products, product_diversity, product_churn_rate, net_product_growth`
