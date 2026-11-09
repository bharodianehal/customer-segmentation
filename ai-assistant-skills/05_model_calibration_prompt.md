# Databricks Assistant Prompt: 05 Model Calibration

## Prompt

Create a Databricks notebook called "05 -- Model Calibration" that performs K-Means clustering with elbow/silhouette analysis. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install and Setup**
- First cell: `%pip install scikit-learn matplotlib seaborn --quiet`
- Second cell: `dbutils.library.restartPython()`
- Third cell: Set CATALOG and SCHEMA. Import: `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`, `StandardScaler` from sklearn.preprocessing, `KMeans` from sklearn.cluster, `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score` from sklearn.metrics, `pyspark.sql.functions as F`, `Window`. Set `sns.set_style("whitegrid")` and `plt.rcParams.update({"figure.dpi": 120, "font.size": 10})`. Suppress warnings.

**Step 2: Feature Preparation**
- Define a `prepare_clustering_features(df, feature_cols, log_cols)` function that:
  1. Extracts feature_cols as numpy array, fills NaN with 0.
  2. Applies `np.log1p(np.abs(...))` to columns in log_cols.
  3. Applies `StandardScaler().fit_transform()`.
  4. Returns the scaled matrix and the scaler.
- Load `ai_04_customer_features_reference` to pandas.
- Build `feature_cols` by excluding non-numeric columns: `{"customer_id", "first_name", "last_name", "gender", "region", "tier", "reference_date", "project_reference_date", "age_bracket"}`. Only keep columns with numeric dtype (`df[c].dtype.kind in ("f", "i")`).
- Build `LOG_COLS` as columns containing keywords: "income", "balance", "revenue", "atm_withdrawals_sum".
- Print all feature columns with `[log]` tag for log-transformed ones.
- Call `prepare_clustering_features()`. Print the matrix shape.

**Step 3: Elbow and Silhouette Analysis (K=3 to 8)**
- Loop K from 3 to 8. For each K:
  - `KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=1337)`
  - Compute `silhouette_score` (with `sample_size=min(5000, len(X_scaled))`), `calinski_harabasz_score`, `davies_bouldin_score`.
  - Store results in a list of dicts with keys: k, inertia, silhouette, calinski_harabasz, davies_bouldin.
  - Print each K's metrics.
- Save metrics as `ai_05_cluster_metrics` Delta table.
- **Visualization**: Create a 2x2 subplot grid (12x8 figsize):
  - Top-left: Inertia (Elbow Curve) with blue "o-" line.
  - Top-right: Silhouette Score with green "s-" line.
  - Bottom-left: Calinski-Harabasz Index with red "^-" line.
  - Bottom-right: Davies-Bouldin Index with purple "D-" line.
  - Add a vertical dashed gray line at K=4 on all subplots.
  - Use `plt.tight_layout()` and `plt.show()`.

**Step 4: Train Final Model (K=4)**
- Set `OPTIMAL_K = 4`.
- Train: `KMeans(n_clusters=4, init="k-means++", n_init=25, max_iter=500, random_state=1337)`.
- Add `cluster_labels` column to df_ref.
- Print final silhouette score.
- Print segment distribution (count and percentage per cluster).
- **Bar chart**: Segment size distribution with colors `["#2563eb", "#16a34a", "#dc2626", "#f59e0b"]`. Add count and percentage labels above each bar.

**Step 5: Cluster Profiling and Auto-Labeling**
- Compute mean feature values per cluster. Add `count` and `pct` columns.
- Print key feature profiles.
- **Heatmap**: Normalize profiles column-wise (z-score across clusters). Use `sns.heatmap` with `cmap="RdYlGn"`, `center=0`. Title: "Cluster Feature Profiles (z-scored across clusters)".
- **Radar chart**: Select key features for radar (income, customer_age, tenure_months, nb_subscribed_products, plus first balance_avg, revenue_avg, digital_services column). Normalize to 0-1 range. Plot polar chart with filled areas.
- **Tier distribution**: Cross-tab of cluster_labels x tier, shown as percentages.
- **Auto-label segments** using rule-based logic:
  - Define `label_cluster(row)` function that checks:
    - `high_income`: income > 75th percentile across clusters.
    - `high_digital`: digital_services_avg > median across clusters.
    - `many_products`: nb_subscribed_products > median.
    - `low_products`: nb_subscribed_products < 25th percentile.
    - `high_balance`: total balance columns > 1.5x median.
  - Assign labels: "Premium High-Value", "Digital-First Engaged", "Low Engagement At-Risk", "Multi-Product Middle Market", or "Mainstream Active".
  - De-duplicate labels to ensure uniqueness.
  - Map labels back to df_ref.
  - Print segment labels with counts and percentages.

**Step 6: Save All Outputs**
- `ai_05_cluster_profiles`: Save profiles DataFrame (reset index, cast types).
- `ai_05_customer_reference_scored`: Save df_ref with cluster_labels and segment_label.
- `ai_05_customer_segments`: Save just `customer_id, cluster_labels, segment_label`.
- `ai_05_age_demographics`: Create from scored data -- add `age_quintile` via `ntile(5).over(Window.orderBy("customer_age"))`, group by `(tier, cluster_labels, segment_label, age_quintile)`, aggregate customer_count, age_min, age_max, add `age_bounds` column.
- `ai_05_customers_nbo`: Save subset columns: `customer_id, first_name, last_name, gender, tier, income, region, customer_age, tenure_months, cluster_labels, segment_label`.

**Step 7: Summary**
- Print a formatted summary with model details (K, silhouette), counts, and segment descriptions with characteristics (income level, product count, digital activity).
- List all output tables.

## Expected Output

A notebook that runs K-Means clustering (K=3-8 sweep then final K=4), produces 4 visualizations (2x2 metrics grid, segment bar chart, heatmap, radar chart), auto-labels segments, and saves 6 output tables: ai_05_cluster_metrics, ai_05_cluster_profiles, ai_05_customer_reference_scored, ai_05_customer_segments, ai_05_age_demographics, ai_05_customers_nbo.
