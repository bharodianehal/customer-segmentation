# Bank Customer Segmentation on Databricks

An end-to-end retail-banking customer segmentation pipeline built as Databricks notebooks. It covers ingestion, EDA,
feature engineering, Feature Store registration, K-Means calibration, temporal stability validation, activation,
propensity, churn, CLV, dynamic segmentation and AI-generated campaign copy. It also ships **Databricks Assistant
skills / prompts** that regenerate the same pipeline with the AI assistant. All data is **synthetic**.

> See [DISCLAIMER.md](DISCLAIMER.md). This is a demonstration project, not a supported product.

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_ingestion.py` | Load raw files from a UC Volume into `01_raw_*` Delta tables with basic DQ checks |
| 02a-d | `02*_explore_*.py` | EDA: customers, products, financials (balances & revenue), behavioural |
| 03a-d | `03*_features_*.py` | Feature derivation per domain |
| 04 | `04_feature_join.py` | Join features and register them in the Unity Catalog Feature Store |
| 05 | `05_model_calibration.py` | K-Means calibration (k selection, scaling), MLflow tracking |
| 06 | `06_model_validation.py` | Temporal segment stability |
| 07 | `07_activation.py` | Segment activation tables |
| 08 | `08_propensity_scoring.py` | Product propensity model (MLflow) |
| 09 | `09_campaign_targeting.py` | Campaign targeting + AI-generated campaign copy (`ai_query`) |
| 10 | `10_dynamic_segmentation.py` | Segment drift / dynamic re-segmentation |
| 11 | `11_churn_prediction.py` | Churn risk scoring |
| 12 | `12_customer_lifetime_value.py` | Customer lifetime value |
| 13 | `13_analyticsoul_segmentation.py` | Tutorial-style walkthrough following the public AnalyticSoul segmentation lessons |

`ai-notebooks/` holds the AI-assistant-generated equivalents (prefix `ai_`). `ai-assistant-skills/` and
`skills/` hold the prompts and skill definitions used to generate them. `DEMO_SCRIPT.md` is a presenter script,
and `data transformation.docx` describes the input elements and feature logic.

## Setup

1. Create a catalog/schema (the notebooks default to `customer_segmentation.banking` or `main.banking`. Edit the
   `CATALOG` / `SCHEMA` constants at the top of each notebook to match your environment).
2. Create a Volume `raw_data` in that schema and upload the source files described in notebook 01 (customers,
   products, product holdings, balances, revenues, additional behavioural information, dates). **No data files are
   included in this repo.** Use your own synthetic data with the documented columns.
3. Import the notebooks into a Databricks workspace (or use a Git folder) and run them in order on serverless or a
   cluster with Databricks Runtime ML. MLflow experiments are written under `/Shared/`.
4. Optionally deploy the companion [segmentation-app](https://github.com/bharodianehal/segmentation-app) on top of
   the output tables.

## Disclaimer

Educational/demo use only, synthetic data, not credit/marketing/regulatory advice, no warranty. See
[DISCLAIMER.md](DISCLAIMER.md).
