# Databricks notebook source
# MAGIC %md
# MAGIC # ai_00 — Copy Raw Data to ai_banking Volume
# MAGIC
# MAGIC This utility notebook copies the 7 source TSV files from `banking` volume
# MAGIC to the `ai_banking` volume so the AI-generated pipeline has its own data.

# COMMAND ----------

SOURCE_VOL = "/Volumes/customer_segmentation/banking/raw_data"
TARGET_VOL = "/Volumes/customer_segmentation/ai_banking/raw_data"

# COMMAND ----------

files = [
    "additional_information.csv",
    "balances.csv",
    "customers.csv",
    "dates.csv",
    "product_holdings.csv",
    "products.csv",
    "revenues.csv",
]

for f in files:
    dbutils.fs.cp(f"dbfs:{SOURCE_VOL}/{f}", f"dbfs:{TARGET_VOL}/{f}", recurse=False)
    print(f"  Copied: {f}")

print(f"\nAll {len(files)} files copied to {TARGET_VOL}")

# COMMAND ----------

# Verify
for f in files:
    info = dbutils.fs.ls(f"dbfs:{TARGET_VOL}/{f}")
    size_kb = info[0].size / 1024
    print(f"  {f}: {size_kb:.1f} KB")
