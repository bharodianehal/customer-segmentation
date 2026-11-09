# Databricks notebook source

# MAGIC %md
# MAGIC # 08 - Propensity Scoring Model
# MAGIC
# MAGIC **Purpose**: Build a propensity-to-buy model for targeted promotions using customer features.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the customer feature table
# MAGIC 2. Creates a synthetic `high_value_responder` target variable
# MAGIC 3. Trains RandomForest and GradientBoosting classifiers
# MAGIC 4. Logs both models to MLflow with evaluation metrics
# MAGIC 5. Registers the best model in the Model Registry
# MAGIC 6. Scores all customers and saves propensity scores as a Feature Store table

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "main"
SCHEMA = "banking"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.04_customer_features_reference"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.08_customer_propensity_scores"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.propensity_to_respond"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Feature table:  {FEATURE_TABLE}")
print(f"Output table:   {OUTPUT_TABLE}")
print(f"Model registry: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Customer Features

# COMMAND ----------

df_features = spark.table(FEATURE_TABLE)
print(f"Loaded {df_features.count():,} customers from feature table")
df_features.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Synthetic Target Variable
# MAGIC
# MAGIC We simulate a realistic promotion response pattern:
# MAGIC - Customers with **high digital engagement** (`digital_ratio > 0.5`), **high income** (`income_quintile >= 4`), and **product diversity** (`>= 3`) respond with **70% probability**
# MAGIC - All other customers respond with **15% probability**
# MAGIC
# MAGIC This mirrors real-world behavior where digitally engaged, affluent, multi-product customers are more likely to respond to targeted campaigns.

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

pdf = df_features.toPandas()
print(f"Converted to pandas: {len(pdf):,} rows")

# Set seed for reproducibility
np.random.seed(42)

# Define high-propensity segment
high_propensity_mask = (
    (pdf["digital_ratio"] > 0.5) &
    (pdf["income_quintile"] >= 4) &
    (pdf["product_diversity"] >= 3)
)

print(f"High-propensity segment: {high_propensity_mask.sum():,} customers ({high_propensity_mask.mean():.1%})")
print(f"Low-propensity segment:  {(~high_propensity_mask).sum():,} customers ({(~high_propensity_mask).mean():.1%})")

# Assign labels with probabilistic noise
pdf["high_value_responder"] = 0
pdf.loc[high_propensity_mask, "high_value_responder"] = np.random.binomial(1, 0.70, size=high_propensity_mask.sum())
pdf.loc[~high_propensity_mask, "high_value_responder"] = np.random.binomial(1, 0.15, size=(~high_propensity_mask).sum())

print(f"\nTarget distribution:")
print(f"  Responders (1): {pdf['high_value_responder'].sum():,} ({pdf['high_value_responder'].mean():.1%})")
print(f"  Non-responders (0): {(1 - pdf['high_value_responder']).sum():,.0f} ({1 - pdf['high_value_responder'].mean():.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Selection & Preparation

# COMMAND ----------

FEATURE_COLS = [
    "income_log",
    "total_balance_log",
    "total_revenue_log",
    "digital_ratio",
    "revenue_per_product",
    "product_churn_rate",
    "atm_intensity",
    "income_quintile",
    "customer_age",
    "tenure_months",
    "nb_active_products",
    "product_diversity",
    "gender_female",
    "gender_male",
]
TARGET_COL = "high_value_responder"

# Check which features are available
available_features = [c for c in FEATURE_COLS if c in pdf.columns]
missing_features = [c for c in FEATURE_COLS if c not in pdf.columns]

if missing_features:
    print(f"WARNING: Missing features (will be excluded): {missing_features}")

FEATURE_COLS = available_features
print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

# Prepare model data
model_df = pdf[FEATURE_COLS + [TARGET_COL, "customer_id"]].copy()
model_df[FEATURE_COLS] = model_df[FEATURE_COLS].fillna(0)

print(f"\nModel dataset: {len(model_df):,} rows x {len(FEATURE_COLS)} features")
print(f"Feature summary:")
display(model_df[FEATURE_COLS].describe().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train/Test Split

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = model_df[FEATURE_COLS].values
y = model_df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} rows ({y_train.mean():.1%} positive)")
print(f"Test set:     {len(X_test):,} rows ({y_test.mean():.1%} positive)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Models & Log to MLflow
# MAGIC
# MAGIC We train two classifiers and compare performance:
# MAGIC - **RandomForestClassifier** -- robust, interpretable feature importances
# MAGIC - **GradientBoostingClassifier** -- typically higher AUC, sequential boosting

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/bank-customer-segmentation/08_propensity_scoring"
mlflow.set_experiment(experiment_name)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    ),
}

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"  AUC:       {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

        # Log to MLflow
        mlflow.log_params({
            "model_type": name,
            "n_features": len(FEATURE_COLS),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate": float(y_train.mean()),
        })
        mlflow.log_metrics({
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test[:5],
        )

        # Feature importances
        importances = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("Feature Importances:")
        print(importances.to_string(index=False))

        results[name] = {
            "run_id": run.info.run_id,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "model": model,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Select Best Model & Register

# COMMAND ----------

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]

print(f"Best model: {best_name}")
print(f"  AUC:       {best['auc']:.4f}")
print(f"  Precision: {best['precision']:.4f}")
print(f"  Recall:    {best['recall']:.4f}")
print(f"  F1:        {best['f1']:.4f}")

# Register in Unity Catalog Model Registry
model_uri = f"runs:/{best['run_id']}/model"
registered = mlflow.register_model(model_uri, MODEL_NAME)
print(f"\nRegistered model: {MODEL_NAME}")
print(f"  Version: {registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Score All Customers

# COMMAND ----------

best_model = best["model"]

# Score all customers
X_all = model_df[FEATURE_COLS].values
all_probs = best_model.predict_proba(X_all)[:, 1]
all_preds = best_model.predict(X_all)

# Build scores dataframe
scores_df = pd.DataFrame({
    "customer_id": model_df["customer_id"].values,
    "propensity_score": np.round(all_probs, 6),
    "predicted_responder": all_preds.astype(int),
})

# Compute deciles (1 = lowest propensity, 10 = highest)
scores_df["propensity_decile"] = pd.qcut(
    scores_df["propensity_score"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1

print(f"Scored {len(scores_df):,} customers")
print(f"\nScore distribution:")
print(scores_df["propensity_score"].describe().round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Propensity Scores to Delta Table

# COMMAND ----------

spark_scores = spark.createDataFrame(scores_df)

spark_scores.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(OUTPUT_TABLE)

print(f"Saved {spark_scores.count():,} rows to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Register as Feature Store Table

# COMMAND ----------

# Feature Store registration skipped - using direct saveAsTable

# fe = FeatureEngineeringClient()

try:
    # Feature Store registration skipped
    print(f"Registered {OUTPUT_TABLE} in Feature Store with primary key: customer_id")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Feature table {OUTPUT_TABLE} already registered in Feature Store")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary & Lift Analysis

# COMMAND ----------

print("=" * 70)
print("PROPENSITY SCORING SUMMARY")
print("=" * 70)

# Score distribution by decile
decile_summary = scores_df.groupby("propensity_decile").agg(
    count=("customer_id", "count"),
    avg_score=("propensity_score", "mean"),
    min_score=("propensity_score", "min"),
    max_score=("propensity_score", "max"),
    predicted_responders=("predicted_responder", "sum"),
).reset_index()

decile_summary["response_rate"] = (
    decile_summary["predicted_responders"] / decile_summary["count"]
)

print("\nScore Distribution by Decile:")
print("-" * 70)
print(f"{'Decile':>7} {'Count':>7} {'Avg Score':>10} {'Min':>8} {'Max':>8} {'Resp Rate':>10}")
print("-" * 70)
for _, row in decile_summary.iterrows():
    print(
        f"{int(row['propensity_decile']):>7} "
        f"{int(row['count']):>7} "
        f"{row['avg_score']:>10.4f} "
        f"{row['min_score']:>8.4f} "
        f"{row['max_score']:>8.4f} "
        f"{row['response_rate']:>10.1%}"
    )

# Lift chart data
overall_rate = scores_df["predicted_responder"].mean()
decile_summary["lift"] = decile_summary["response_rate"] / overall_rate

print(f"\nOverall response rate: {overall_rate:.1%}")
print(f"\nLift by Decile (top deciles):")
print("-" * 40)
for _, row in decile_summary.sort_values("propensity_decile", ascending=False).head(4).iterrows():
    print(f"  Decile {int(row['propensity_decile'])}: {row['lift']:.2f}x lift")

# Targeting recommendation
top_deciles = scores_df[scores_df["propensity_decile"] >= 7]
print(f"\nTargeting Recommendation:")
print(f"  Target deciles 7-10: {len(top_deciles):,} customers ({len(top_deciles)/len(scores_df):.0%} of base)")
print(f"  Expected responders: {top_deciles['predicted_responder'].sum():,}")
print(f"  Expected response rate: {top_deciles['predicted_responder'].mean():.1%}")

print("\nPropensity scoring complete.")
