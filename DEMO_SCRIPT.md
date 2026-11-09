# Bank Customer Segmentation Demo Script

## Demo Overview

**Duration:** 30-40 minutes
**Audience:** Regional bank analytics/marketing leaders evaluating Databricks
**Workspace:** customer_intelligence catalog on Databricks Unity Catalog
**Value Prop:** A modern, AI-assisted, regulatory-compliant segmentation platform on Databricks

---

## Before You Start

- Open the Databricks workspace: https://<workspace>.cloud.databricks.com
- Navigate to Catalog Explorer and show the **customer_intelligence** catalog
- Have all 6 notebooks ready in tabs (01 through 06)
- Confirm the MLflow experiment exists at /Users/{user}/customer_segmentation

---

## Act 1: The Data Foundation (5 min)

### Open NB01 - Data Generation

**What to say:**

"We start with a realistic banking dataset for a fictional US regional bank. This notebook generates synthetic data that mirrors production-quality CIF and core banking extracts."

**Key points to highlight:**

- **Scale:** 100,000 customers, ~180K accounts, ~10M transactions (12-month history), ~85K loans, 100K digital engagement records, ~250K product holdings
- **Geographic footprint:** 12 states - NY (35%), MD (18%), PA (14%), CT, NJ, and others
- **Data quality built in:** Credit scores calibrated to typical US averages (720-740), income correlated with age/education, home ownership correlated with age
- **Loan realism:** Mortgage and HELOC restricted to homeowners only, credit score at origination drifts from current score based on loan age
- **Household linkage:** Married customers in the same city share household IDs - enabling household-level aggregation

**Show in Catalog Explorer:**

- Open **customer_intelligence.customer_segmentation** schema
- Click into the **customers** table - show 100K rows, column types
- Click into **transactions** - show 10M+ rows, 12-month date range

**Transition:** "With our data foundation in place, let's see what the Data Science Agent discovers in the data."

---

## Act 2: AI-Assisted EDA (5-7 min)

### Open NB02 - EDA Data Analysis

**What to say:**

"This is where Databricks differentiates. Instead of manually writing 200 lines of matplotlib code, we designed this notebook for the Databricks Data Science Agent. The Agent reads the notebook structure and generates analysis cells interactively."

**Key sections to scroll through (don't run - show structure):**

1. Data Overview and Quality Check
2. Customer Demographics Analysis
3. Financial Profile Analysis
4. Transaction Pattern Analysis
5. Digital Engagement Analysis
6. Lending Portfolio Analysis
7. Correlation Analysis and Feature Selection Hints
8. Key Findings Summary

**What to highlight:**

- "The Agent identifies that digital adoption is heavily age-correlated - 94% mobile enrollment for under-30s vs. 35% for 65+"
- "It flags that transaction channel usage has shifted: mobile is now 45% of transactions, branch is down to 8%"
- "The correlation analysis automatically surfaces feature pairs with |r| > 0.70, giving us a head start on feature selection"

**Demo tip:** If time allows, open a new cell and type a question for the Data Science Agent like "What is the relationship between tenure and product count?" to show the interactive AI experience.

**Transition:** "The EDA confirms our hypothesis: we need both a value model and a behavioral model. Let's build the features."

---

## Act 3: Feature Engineering with Custom Skill (7-10 min)

### Show the Custom Skill (Databricks Assistant)

**What to say:**

"Before we run feature engineering, let me show you something unique. We built a custom skill for the Databricks Assistant that encodes our banking domain expertise - 66 features across 7 domains with SQL expressions and business rationale."

**Open the Assistant sidebar and demonstrate:**

- Ask: "What RFM features should I compute for a 12-month customer segmentation?"
- The skill responds with specific SQL expressions, window definitions, and rationale
- Ask: "How should I handle feature selection for clustering?"
- Shows the Correlation + VIF + VARCLUS pipeline

**What to highlight:**

- "This skill codifies institutional knowledge. When your senior data scientist retires, their expertise doesn't leave with them."
- "New team members can ask the Assistant and get production-quality feature engineering patterns instantly."

### Open NB03 - Feature Engineering

**Walk through the 6 feature domains:**

| Domain | Features | Business Purpose |
|--------|----------|-----------------|
| RFM (Recency, Frequency, Monetary) | 12-month primary window with 30d/90d sub-windows | Core transaction behavior |
| Financial Profile | Balances, products, deposit-to-loan ratio | Relationship depth |
| Trend and Momentum | 30-day balance trends, growing/declining flags | Trajectory signals |
| Life Stage | Age bracket, career stage, household size | Demographic context |
| Digital and Channel | Digital engagement score, channel diversity, adoption score | Channel preference |
| CLV and Attrition | Estimated monthly revenue, dormancy flag, days since last deposit | Retention risk |

**Key talking points:**

- "We compute 66 candidate features, then use automated feature selection to reduce to the best 15-18"
- "Feature selection uses Pearson correlation (remove pairs with |r| > 0.70) followed by iterative VIF removal (threshold 5.0)"
- "Two output tables: one for the Value model, one for the Behavioral model - this two-model approach is the industry standard for banking segmentation"

**Show the output tables in Catalog Explorer:**

- **customer_value_features** - financial and relationship features
- **customer_behavior_features** - digital and channel features

**Transition:** "Features are ready. Now let's train and track our models with MLflow."

---

## Act 4: Model Training with MLflow (5-7 min)

### Open NB04 - Model Training

**What to say:**

"We run K-Means clustering for K equals 4 through 6. Why this range? In banking, more than 6 segments is too many for marketing to operationalize, and fewer than 4 misses important distinctions."

**Key results to highlight:**

- **Elbow plot:** Show the inertia curve - point to where diminishing returns begin
- **Silhouette scores:** Best K selected automatically by highest silhouette score
- **MLflow experiment:** Click into the MLflow experiment to show all runs tracked with parameters, metrics, and artifacts

**What to say about MLflow:**

"Every model run is fully tracked. We log the K value, feature list, all four metrics, the trained model, and the scaler. If a regulator asks 'why K=5 instead of K=4?' - we have the complete audit trail."

**Show the segment labels (auto-generated):**

**Value Segments:**
- High Value Core - highest balances, longest tenure, most products
- Growth Potential - moderate balances, active and trending up
- Mass Market Active - lower balances but frequent transactions
- New and Developing - short tenure, building relationship
- At Risk / Declining - negative trends, low engagement
- Dormant / Low Value - minimal activity

**Behavioral Segments:**
- Digital First - high mobile and online engagement
- Omnichannel Active - uses all channels actively
- Branch Dependent - heavy branch usage, low digital
- Digital Adopter - recently shifting to digital
- Low Engagement - minimal interaction across all channels

**Cross-Reference Matrix:**

"This is where it gets powerful. We cross the Value and Behavioral segments to create actionable micro-segments."

| | Digital First | Branch Dependent | Low Engagement |
|---|---|---|---|
| **High Value** | Retain and deepen | Digital migration | Re-engage urgently |
| **Growth** | Cross-sell digitally | Educate on digital | Onboarding push |
| **At Risk** | Digital retention | Branch retention | Managed attrition |

**Transition:** "Before we activate these segments, we need to validate them - especially for fair lending compliance."

---

## Act 5: Validation and Fair Lending (5-7 min)

### Open NB05 - Model Validation

**What to say:**

"This is the notebook that separates a serious banking segmentation from a weekend project. We validate across four dimensions: statistical quality, business utility, segment stability, and fair lending compliance."

**Statistical validation:**

- Silhouette plot: "Each cluster should have mostly positive silhouette values. Clusters with more than 10% negative values have assignment uncertainty at the boundaries."
- Bootstrap stability: "We resample 80% of the data 10 times and re-cluster. An Adjusted Rand Index above 0.80 means our segments are stable - they're not artifacts of the particular sample."

**Fair lending compliance (THE differentiator):**

**What to say:**

"This is critical for any regulated financial institution. Under ECOA and the Fair Housing Act, we cannot create segments that have disparate impact on protected classes. We test four proxy discrimination vectors."

| Test | Method | What We Check |
|------|--------|---------------|
| Geographic concentration | Chi-squared test | Is any segment disproportionately concentrated in specific states? (Geography can proxy for race/ethnicity) |
| Age distribution | Bracket ratio analysis | Does any segment systematically exclude an age group? (Age is protected under ECOA) |
| Gender distribution | Proportion check | Does any segment exceed 65% of one gender? |
| Marital status | Cross-tabulation | Are married/unmarried customers disproportionately sorted? |

**What to say:**

"We document all of this per SR 11-7 - the Fed's Model Risk Management guidance. This notebook IS the validation report your model risk management team needs."

**Regulatory references to mention:**

- Equal Credit Opportunity Act (ECOA) - 15 U.S.C. 1691
- SR 11-7: Guidance on Model Risk Management
- Fair Housing Act - 42 U.S.C. 3601-3619

**Transition:** "Segments are validated and compliant. Now let's put them to work."

---

## Act 6: Campaign Activation (5-7 min)

### Open NB06 - Activation and Monitoring

**What to say:**

"The best segmentation in the world is useless if it sits in a data warehouse. This notebook builds the activation layer - Next Best Action rules, campaign audience lists, and ongoing monitoring."

**Next Best Action engine (18 rules):**

"Each customer gets a personalized recommendation based on their Value and Behavioral segment intersection."

Examples:
- High Value + Branch Dependent: "Offer dedicated digital concierge service" - Priority: HIGH
- At Risk + Any: "Retention offer with branch manager outreach within 30 days" - Priority: CRITICAL
- Growth + Digital First: "Pre-approved credit card via mobile app notification" - Priority: HIGH
- New + Low Engagement: "First 90 days engagement program" - Priority: CRITICAL

**Three pilot campaigns:**

| Campaign | Target Segment | Audience Size | Objective |
|----------|---------------|---------------|-----------|
| Digital Migration | High Value + Branch Dependent | ~15-20% of base | Move branch-heavy HNW to digital |
| Retention | At Risk / Declining | ~10-15% of base | Prevent attrition with 30-day outreach |
| Cross-Sell | Growth + Digital | ~15-20% of base | Deepen relationship via digital offers |

**Reverse ETL to Salesforce Marketing Cloud:**

"The segment_export_sfmc table is designed for Hightouch Reverse ETL. It syncs daily to SFMC Data Extensions, which feed Journey Builder for automated email, SMS, push, and branch alert orchestration."

**PSI drift monitoring:**

"We compute Population Stability Index monthly. PSI below 0.10 means segments are stable. Between 0.10 and 0.25 we investigate. Above 0.25 we recalibrate the model."

---

## Closing: Why Databricks (3 min)

**What to say:**

"Let me summarize what you just saw and why this matters."

| Capability | Traditional Tools | Databricks |
|-----------|-------------------|------------|
| AI-Assisted EDA | Manual coding | Data Science Agent generates analysis interactively |
| Feature Engineering | Manual, no institutional memory | Custom Skills encode domain expertise, reusable across team |
| Model Tracking | Limited audit trail | MLflow - full experiment tracking, model registry, deployment |
| Fair Lending Validation | Separate manual process | Built into the pipeline, automated proxy testing |
| Campaign Activation | Export CSV, manual upload | Reverse ETL to SFMC via Hightouch - automated daily sync |
| Drift Monitoring | Manual, periodic | PSI computed automatically, alerts on threshold breach |
| Collaboration | Per-user licensing | Unity Catalog - governed, shared, no per-seat licensing for data |
| Scalability | Memory-limited | Serverless - 10M transactions processed in minutes |
| Total Cost | High licensing costs | Consumption-based, typically 40-60% lower TCO |

**Closing statement:**

"This entire pipeline - from data generation to campaign activation - runs on serverless compute with Unity Catalog governance. Every table, every model, every segment assignment is governed, auditable, and ready for regulatory review. And your team can build it with AI assistance."

---

## Common Customer Questions

**Q: How long did this take to build?**
A: The pipeline was built in days, not months. With the custom Databricks Assistant skill, a junior data scientist can replicate this pattern for any segmentation use case.

**Q: Can this handle our actual data volumes?**
A: Absolutely. This demo uses 10M transactions on serverless. Databricks scales linearly - 100M or 1B transactions just means slightly longer compute time, with no code changes.

**Q: What about real-time scoring?**
A: MLflow model serving can score new customers in real-time via REST API. The batch pipeline refreshes segments nightly, and real-time scoring handles intra-day events.

**Q: How do we integrate with our existing martech stack?**
A: Hightouch Reverse ETL connects to 150+ destinations: Salesforce, Adobe, Braze, HubSpot, etc. The segment_export_sfmc table is the integration point.

**Q: What about model governance and approval workflows?**
A: MLflow Model Registry provides staging, production, and archived stages with approval gates. Unity Catalog provides lineage tracking from raw data to final segment assignment.

**Q: Is this compliant with OCC/Fed model risk requirements?**
A: NB05 implements SR 11-7 validation requirements: model purpose documentation, validation testing, fair lending analysis, and ongoing monitoring. The notebook itself serves as the validation report.
