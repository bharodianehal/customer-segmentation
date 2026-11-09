# Databricks Assistant Prompt: 06 Model Validation

## Prompt

Create a Databricks notebook called "06 -- Model Validation: Temporal Segment Stability" that validates the clustering model by analyzing how customer segments evolve over time. Use `CATALOG = "customer_segmentation"` and `SCHEMA = "ai_banking"`.

**Step 1: Install and Setup**
- First cell: `%pip install matplotlib seaborn --quiet`
- Second cell: Set CATALOG and SCHEMA.
- Import: `pyspark.sql.functions as F`, `Window` from pyspark.sql.window, `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `matplotlib.ticker as mticker`, `seaborn as sns`.
- Set `sns.set_theme(style="whitegrid", palette="Set2")` and `plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})`.

**Step 2: Load Data and Early Exit Check**
- Load `ai_05_customer_reference_scored`.
- Try to load `ai_05_customer_historical_scored`. If it does not exist or has 0 rows, set `has_historical = False`.
- If no historical data: print a message and call `dbutils.notebook.exit("SKIPPED: No historical data available for temporal validation")`.
- Combine reference + historical using `ref_scored.unionByName(hist_scored, allowMissingColumns=True)`.

**Step 3: Historical Segment Analysis**
- Dynamically identify revenue columns (containing "revenue_last").
- Compute `total_revenue` as sum of revenue columns (or 0 if none).
- Group by `(reference_date, cluster_labels, segment_label)` and aggregate: customer_count, total_revenue_sum, total_revenue_avg.
- Save as `ai_06_historical_segments`. Display it.
- **Stacked area chart**: Pivot on segment_label, values=customer_count. Use `ax.stackplot()`. Title: "Segment Sizes Over Time". Format y-axis with comma separator.
- **Segment share line chart**: Convert to percentages. Plot each segment as a line with markers. Title: "Segment Share (%) Over Time". Use `PercentFormatter`.

**Step 4: Transition Analysis**
- Create window `w_cust = Window.partitionBy("customer_id").orderBy("reference_date")`.
- Add lag columns: `cluster_lag1` (lag 1), `cluster_lag3` (lag 3), `segment_lag1`, `segment_lag3`.
- Print total records, records with 1-month lag, records with 3-month lag.

**Step 5: 1-Month Transition Matrix**
- Filter to rows with non-null `cluster_lag1`.
- Group by `(segment_lag1 as from_segment, segment_label as to_segment)`, count.
- Add `total_from` (window sum by from_segment) and `transition_probability = count / total_from` (rounded to 4 decimals).
- Save as `ai_06_transitions_month`. Display it.
- **Heatmap**: Pivot to matrix form. Use `sns.heatmap` with `annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1`. Title: "1-Month Transition Probability Matrix".

**Step 6: 3-Month Transition Matrix**
- Same pattern as 1-month but using `cluster_lag3` and `segment_lag3`.
- Save as `ai_06_transitions_quarter`. Display it.
- **Heatmap**: Same style. Title: "3-Month Transition Probability Matrix".

**Step 7: Segment Stability (Retention Rates)**
- From `ai_06_transitions_month`, filter where `from_segment == to_segment` to get 1-month retention rates. Add `window = "1-month"`.
- From `ai_06_transitions_quarter`, same for 3-month. Add `window = "3-month"`.
- Union both. Save as `ai_06_segment_stability`.
- Print formatted retention rates for each segment and window.
- Print most stable and least stable segments (1-month).
- **Bar chart**: Side-by-side bars comparing 1-month vs 3-month retention for each segment. Blue for 1-month, orange for 3-month. Add percentage labels. Title: "Segment Retention Rates: 1-Month vs 3-Month". Use `PercentFormatter(xmax=1)`.

**Step 8: Migration Patterns**
- For each segment, print outflow (where customers go) and inflow (where customers come from) based on 1-month transitions. Mark retention rows.
- **Horizontal bar chart**: Top 15 migration paths (excluding retention/diagonal). Show customer counts with transition probability labels. Title: "Top Migration Paths (1-Month, Excluding Retention)".

**Step 9: Summary**
- Print overall model stability assessment:
  - If avg 1-month retention >= 80%: "STRONG"
  - If 60-80%: "MODERATE"
  - If < 60%: "WEAK"
- Print top 5 migration paths.
- List all saved tables: ai_06_historical_segments, ai_06_transitions_month, ai_06_transitions_quarter, ai_06_segment_stability.

## Expected Output

A notebook that validates segment temporal stability with: stacked area chart of segment sizes, segment share line chart, 1-month and 3-month transition matrix heatmaps, retention rate comparison bar chart, migration path analysis, and saves 4 output tables: ai_06_historical_segments, ai_06_transitions_month, ai_06_transitions_quarter, ai_06_segment_stability. Gracefully exits if no historical data is available.
