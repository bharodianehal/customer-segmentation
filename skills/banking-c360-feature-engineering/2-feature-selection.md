# Feature Selection for Banking Segmentation

Techniques to reduce dimensionality and remove multicollinearity before clustering. Banking C360 data typically starts with 60+ candidate features and needs reduction to 10-15 for effective K-Means clustering.

## Why Feature Selection Matters for Clustering

Unlike supervised learning (where models can internally weight features), K-Means treats all features equally. Correlated features effectively get double-weighted. Feature selection is not optional for clustering -- it directly impacts segment quality.

## Step 1: Correlation Analysis

Remove one of each highly correlated pair (|r| > 0.70).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr_matrix = feature_pdf[feature_columns].corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.70:
            high_corr_pairs.append({
                "feature_1": corr_matrix.columns[i],
                "feature_2": corr_matrix.columns[j],
                "correlation": round(corr_matrix.iloc[i, j], 3)
            })

corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)
print("Highly correlated pairs (|r| > 0.70):")
print(corr_df.to_string(index=False))

# Common banking correlations to watch for:
# - checking_balance <-> total_deposit_balance (0.85+)
# - mobile_logins <-> digital_engagement_score (0.80+)
# - frequency_30d <-> frequency_90d (0.90+)
# - monetary_30d <-> monetary_90d (0.90+)
# - num_deposit_accounts <-> product_count (0.75+)

# Resolution: Keep the broader feature, drop the narrow one
# Keep total_deposit_balance, drop checking_balance
# Keep digital_engagement_score, drop mobile_logins
# Keep frequency_90d, drop frequency_30d (unless computing trend)

# Visualize
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

## Step 2: Variance Inflation Factor (VIF)

Iteratively remove features with VIF > 5 (conservative) or > 10 (lenient).

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Scale first (VIF is sensitive to scale)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(feature_pdf[feature_columns]),
    columns=feature_columns
)

def compute_vif(df):
    """Compute VIF for all features."""
    vif_data = pd.DataFrame({
        "feature": df.columns,
        "VIF": [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    }).sort_values("VIF", ascending=False)
    return vif_data

# Iterative VIF removal
remaining_features = feature_columns.copy()
removed = []

while True:
    vif = compute_vif(X_scaled[remaining_features])
    max_vif = vif.iloc[0]
    if max_vif["VIF"] <= 5:
        break
    print(f"Removing {max_vif['feature']} (VIF={max_vif['VIF']:.1f})")
    removed.append(max_vif["feature"])
    remaining_features.remove(max_vif["feature"])

print(f"\nRemoved {len(removed)} features: {removed}")
print(f"Remaining {len(remaining_features)} features:")
print(compute_vif(X_scaled[remaining_features]).to_string(index=False))
```

## Step 3: Variable Clustering (VARCLUS Equivalent)

Groups correlated variables and selects one representative per cluster using hierarchical variable clustering.

```python
# Option A: Using varclushi (hierarchical variable clustering)
# pip install varclushi
from varclushi import VarClusHi

vc = VarClusHi(
    feature_pdf[feature_columns],
    maxeigval2=1,      # Split clusters when 2nd eigenvalue > 1
    maxclus=None        # Let algorithm determine optimal number
)
vc.varclus()

# View cluster assignments
cluster_assignments = vc.rsquare
print("\nVariable Clusters:")
print(cluster_assignments[["Cluster", "Variable", "RS_Own", "RS_Next", "RS_Ratio"]]
      .sort_values(["Cluster", "RS_Ratio"])
      .to_string(index=False))

# Selection rule: Within each cluster, pick the variable with the
# LOWEST (1 - RS_Own) / (1 - RS_Next) ratio
# This is the variable that best represents its cluster while being
# least related to other clusters
selected = (
    cluster_assignments
    .sort_values("RS_Ratio")
    .groupby("Cluster")
    .first()
    .reset_index()
)
selected_features = selected["Variable"].tolist()
print(f"\nSelected {len(selected_features)} representative features:")
for _, row in selected.iterrows():
    print(f"  Cluster {row['Cluster']}: {row['Variable']} "
          f"(RS_Own={row['RS_Own']:.3f}, RS_Ratio={row['RS_Ratio']:.3f})")
```

```python
# Option B: Using sklearn hierarchical clustering on features (if varclushi unavailable)
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# Convert correlation to distance
corr_matrix = feature_pdf[feature_columns].corr().abs()
distance_matrix = 1 - corr_matrix
condensed = squareform(distance_matrix, checks=False)

# Hierarchical clustering on features
Z = linkage(condensed, method="ward")

# Plot dendrogram
plt.figure(figsize=(14, 8))
dendrogram(Z, labels=feature_columns, leaf_rotation=90)
plt.title("Feature Dendrogram (Ward's Method)")
plt.ylabel("Distance")
plt.axhline(y=0.7, color='r', linestyle='--', label='Cut threshold')
plt.tight_layout()
plt.show()

# Cut at desired threshold
feature_clusters = fcluster(Z, t=0.7, criterion='distance')
feature_cluster_df = pd.DataFrame({
    "feature": feature_columns,
    "cluster": feature_clusters
})

# Select one feature per cluster (highest variance with target or highest business priority)
for cluster_id in sorted(feature_cluster_df["cluster"].unique()):
    members = feature_cluster_df[feature_cluster_df["cluster"] == cluster_id]["feature"].tolist()
    print(f"Feature Cluster {cluster_id}: {members}")
    # Select the most interpretable or highest-variance member
```

## Step 4: PCA (Optional -- for Visualization or Dimensionality Reduction)

Use PCA to understand variance structure. Generally NOT recommended to cluster on PCA components in banking (loses interpretability), but useful for visualization.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=min(15, len(feature_columns)))
X_pca = pca.fit_transform(X_scaled)

# Scree plot
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, alpha=0.7, label="Individual")
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'ro-', label="Cumulative")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.axhline(y=0.80, color='g', linestyle='--', label='80% threshold')
plt.legend()
plt.show()

# Feature loadings for top components
loadings = pd.DataFrame(
    pca.components_[:5].T,
    columns=[f"PC{i+1}" for i in range(5)],
    index=feature_columns
)
print("\nTop PCA Loadings:")
for pc in loadings.columns:
    top = loadings[pc].abs().nlargest(5)
    print(f"\n{pc} (explains {pca.explained_variance_ratio_[int(pc[2:])-1]:.1%}):")
    for feat, val in top.items():
        print(f"  {feat}: {loadings.loc[feat, pc]:.3f}")
```

## Recommended Selection Pipeline

Run these steps in order:

```
1. Start with 60+ candidate features from feature catalog
2. Remove features with > 30% nulls
3. Remove near-zero-variance features (< 1% unique values)
4. Correlation analysis: remove one of each pair with |r| > 0.70
5. VIF analysis: iteratively remove VIF > 5
6. VARCLUS: group remaining into variable clusters, select 1 per cluster
7. Business review: ensure all selected features are explainable
8. Final set: 10-15 features for clustering
```

Typical reduction: 66 features -> ~40 (after nulls/variance) -> ~25 (after correlation) -> ~18 (after VIF) -> ~12-15 (after VARCLUS) -> 10-15 (after business review).
