# Banking C360 Feature Catalog

Complete inventory of features for customer segmentation, organized by domain. Each feature includes its SQL/PySpark expression, business rationale, and which segmentation model it serves.

## Feature Domains

| Domain | Feature Count | Primary Model |
|--------|--------------|---------------|
| RFM (Recency, Frequency, Monetary) | 12 | Value Segmentation |
| Financial Profile | 14 | Value Segmentation |
| Lending & Credit | 10 | Value Segmentation |
| Digital Engagement | 10 | Behavioral Segmentation |
| Channel Mix | 8 | Behavioral Segmentation |
| Product Usage | 6 | Both Models |
| Trend & Momentum | 6 | Both Models |

---

## Domain 1: RFM Features

### Recency

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 1 | `recency_days` | `DATEDIFF(current_date, MAX(transaction_date))` | Days since last transaction. Core recency signal. Lower = more engaged. |
| 2 | `recency_deposit_days` | `DATEDIFF(current_date, MAX(CASE WHEN txn_type IN ('credit','deposit','transfer_in') THEN transaction_date END))` | Days since last deposit/inflow. Signals active income relationship. |
| 3 | `recency_digital_days` | `DATEDIFF(current_date, MAX(CASE WHEN channel IN ('mobile','online') THEN transaction_date END))` | Days since last digital interaction. Captures digital engagement recency. |

### Frequency

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 4 | `frequency_90d` | `COUNT(transaction_id) WHERE date >= current - 90` | Transaction count in 90 days. Core frequency signal. |
| 5 | `frequency_30d` | `COUNT(transaction_id) WHERE date >= current - 30` | Short-term frequency. Captures recent behavior changes. |
| 6 | `active_weeks_90d` | `COUNT(DISTINCT DATE_TRUNC('week', transaction_date))` | Weeks with at least one transaction. More robust than raw count. |
| 7 | `txn_regularity_cv` | `STDDEV(weekly_txn_count) / AVG(weekly_txn_count)` | Coefficient of variation of weekly transaction counts. Low CV = consistent customer. |

### Monetary

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 8 | `monetary_90d` | `SUM(amount) WHERE txn_type IN ('debit','pos_purchase','bill_pay')` | Total spend in 90 days. Core monetary signal. |
| 9 | `monetary_30d` | Same as above for 30 days | Short-term monetary for trend comparison. |
| 10 | `avg_txn_amount` | `AVG(amount)` | Average transaction size. Differentiates high-value vs. high-frequency. |
| 11 | `median_txn_amount` | `PERCENTILE_APPROX(amount, 0.5)` | Median transaction. More robust than mean for skewed data. |
| 12 | `large_txn_count_90d` | `COUNT(*) WHERE amount > PERCENTILE(amount, 0.90)` | Count of transactions above 90th percentile. Signals significant financial activity. |

---

## Domain 2: Financial Profile

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 13 | `total_deposit_balance` | `SUM(current_balance) WHERE account_type IN ('checking','savings','money_market','cd')` | Total deposits. Primary value indicator. Apply `log1p` before scaling. |
| 14 | `checking_balance` | `SUM(current_balance) WHERE account_type = 'checking'` | Operating account balance. Signals primary banking relationship. |
| 15 | `savings_balance` | `SUM(current_balance) WHERE account_type = 'savings'` | Savings buffer. Differentiates savers from transactors. |
| 16 | `high_yield_balance` | `SUM(current_balance) WHERE account_type IN ('money_market','cd')` | Rate-sensitive deposits. Signals investment orientation. |
| 17 | `checking_to_savings_ratio` | `checking_balance / NULLIF(savings_balance, 0)` | High ratio = transactor; Low ratio = saver. Key behavioral indicator. |
| 18 | `num_deposit_accounts` | `COUNT(DISTINCT account_id) WHERE status = 'active'` | Account breadth. More accounts = deeper relationship. |
| 19 | `avg_balance_30d` | `AVG(avg_balance_30d) across accounts` | Recent average. Smoother than point-in-time balance. |
| 20 | `avg_balance_90d` | `AVG(avg_balance_90d) across accounts` | Quarterly average. Used for trend calculation. |
| 21 | `total_relationship_value` | `total_deposit_balance + COALESCE(total_loan_balance, 0)` | Full wallet size. Include both sides of balance sheet. |
| 22 | `annual_income` | Direct from customer table | Income bracket. Normalize to income-to-balance ratio. |
| 23 | `balance_to_income_ratio` | `total_deposit_balance / NULLIF(annual_income, 0)` | Savings rate proxy. High ratio = strong saver relative to income. |
| 24 | `credit_score` | Direct from customer table | FICO score. Represents creditworthiness. |
| 25 | `tenure_months` | Direct from customer table | Relationship length. Longer tenure = stickier customer. |
| 26 | `tenure_years` | `tenure_months / 12.0` | Human-readable tenure for profiling. |

---

## Domain 3: Lending & Credit

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 27 | `total_loan_balance` | `SUM(current_balance) WHERE loan_status IN ('current','delinquent')` | Total outstanding debt with bank. |
| 28 | `num_active_loans` | `COUNT(loan_id) WHERE loan_status IN ('current','delinquent')` | Lending breadth. |
| 29 | `has_mortgage` | `MAX(CASE WHEN loan_type = 'mortgage' THEN 1 ELSE 0 END)` | Anchor product flag. Mortgage customers have highest retention. |
| 30 | `has_credit_card` | `MAX(CASE WHEN loan_type = 'credit_card' THEN 1 ELSE 0 END)` | Card holder flag. Key cross-sell indicator. |
| 31 | `has_auto_loan` | `MAX(CASE WHEN loan_type = 'auto_loan' THEN 1 ELSE 0 END)` | Auto borrower flag. |
| 32 | `credit_utilization` | `SUM(CASE WHEN loan_type='credit_card' THEN current_balance END) / SUM(CASE WHEN loan_type='credit_card' THEN original_amount END)` | Card utilization. High = financial stress or heavy usage. |
| 33 | `deposit_to_loan_ratio` | `total_deposit_balance / NULLIF(total_loan_balance, 0)` | > 1 = net depositor; < 1 = net borrower. Key profitability indicator. |
| 34 | `avg_loan_rate` | `AVG(interest_rate)` | Average borrowing cost. Signals credit quality tier. |
| 35 | `has_delinquency` | `MAX(CASE WHEN delinquency_days > 0 THEN 1 ELSE 0 END)` | Any active delinquency flag. Risk signal. |
| 36 | `max_delinquency_days` | `MAX(delinquency_days)` | Worst delinquency. Differentiates severity. |

---

## Domain 4: Digital Engagement

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 37 | `mobile_logins_30d` | Direct from digital_engagement | Mobile app usage intensity. |
| 38 | `online_logins_30d` | Direct from digital_engagement | Desktop/web usage intensity. |
| 39 | `total_digital_logins_30d` | `mobile_logins_30d + online_logins_30d` | Combined digital activity. |
| 40 | `mobile_to_total_login_ratio` | `mobile_logins_30d / NULLIF(total_digital_logins_30d, 0)` | Mobile preference. High = mobile-first customer. |
| 41 | `digital_engagement_score` | Direct from digital_engagement (0-100) | Composite engagement metric. |
| 42 | `digital_adoption_score` | `mobile_enrolled + online_enrolled + bill_pay + direct_deposit` (0-4) | Feature adoption breadth. More features = stickier. |
| 43 | `mobile_app_enrolled` | Direct flag (0/1) | Mobile adoption. |
| 44 | `bill_pay_enrolled` | Direct flag (0/1) | Bill pay signals primary banking relationship. |
| 45 | `direct_deposit_enrolled` | Direct flag (0/1) | DD is the #1 retention driver. Customers with DD churn at 1/3 the rate. |
| 46 | `mobile_transactions_30d` | Direct from digital_engagement | Mobile transaction volume. Beyond just logins. |

---

## Domain 5: Channel Mix

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 47 | `mobile_txns_90d` | `COUNT(*) WHERE channel = 'mobile'` | Mobile transaction volume. |
| 48 | `online_txns_90d` | `COUNT(*) WHERE channel = 'online'` | Web transaction volume. |
| 49 | `branch_txns_90d` | `COUNT(*) WHERE channel = 'branch'` | Branch transaction volume. |
| 50 | `atm_txns_90d` | `COUNT(*) WHERE channel = 'atm'` | ATM usage volume. |
| 51 | `digital_txn_ratio` | `(mobile_txns + online_txns) / total_txns` | % of transactions via digital channels. Core behavioral signal. |
| 52 | `branch_dependency_ratio` | `branch_txns / total_txns` | Branch reliance. High = candidate for digital migration campaign. |
| 53 | `channel_diversity` | `COUNT(DISTINCT channel)` | Number of distinct channels used. Omnichannel vs. single-channel. |
| 54 | `preferred_channel` | Direct from digital_engagement | Dominant channel for this customer. Use for campaign delivery. |

---

## Domain 6: Product Usage

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 55 | `product_count` | `COUNT(product_id) WHERE status = 'active'` | Total active products. #1 retention predictor -- 3+ products = 90%+ retention. |
| 56 | `deposit_product_count` | `COUNT(*) WHERE product_category = 'deposit'` | Deposit product depth. |
| 57 | `lending_product_count` | `COUNT(*) WHERE product_category = 'lending'` | Lending product depth. |
| 58 | `digital_product_count` | `COUNT(*) WHERE product_category = 'digital'` | Digital feature adoption count. |
| 59 | `has_both_deposit_and_loan` | `CASE WHEN deposit_count > 0 AND lending_count > 0 THEN 1 ELSE 0 END` | Cross-category relationship. Signals full banking relationship. |
| 60 | `product_penetration_score` | `product_count / max_eligible_products` | What % of available products does this customer hold? |

---

## Domain 7: Trend & Momentum

| # | Feature Name | Expression | Rationale |
|---|-------------|------------|-----------|
| 61 | `balance_trend_30d` | `avg_balance_30d - avg_balance_90d` | Positive = growing; Negative = declining. Key early warning for attrition. |
| 62 | `frequency_trend` | `frequency_30d * 3 - frequency_90d` | Annualized 30d frequency vs 90d. Positive = accelerating activity. |
| 63 | `monetary_trend` | `monetary_30d * 3 - monetary_90d` | Spend trajectory. Positive = increasing wallet share. |
| 64 | `digital_adoption_trend` | `digital_logins_30d * 3 - digital_logins_90d` (if 90d available) | Digital migration direction. |
| 65 | `is_growing` | `CASE WHEN balance_trend > 0 AND frequency_trend > 0 THEN 1 ELSE 0 END` | Composite growth flag. Growing customers are prime cross-sell targets. |
| 66 | `is_declining` | `CASE WHEN balance_trend < 0 AND frequency_trend < 0 THEN 1 ELSE 0 END` | Composite decline flag. Declining customers need retention intervention. |

---

## Feature Selection for Each Model

### Customer Value Model (Recommended 12-15 features)

```
recency_days, frequency_90d, monetary_90d, avg_txn_amount,
total_deposit_balance (log1p), savings_balance (log1p),
total_loan_balance (log1p), product_count, tenure_months,
credit_score, annual_income (log1p), balance_trend_30d,
deposit_to_loan_ratio (capped at 10), has_mortgage, has_credit_card
```

### Behavioral/Channel Model (Recommended 10-12 features)

```
digital_engagement_score, digital_txn_ratio, branch_dependency_ratio,
mobile_logins_30d, channel_diversity, digital_adoption_score,
mobile_to_total_login_ratio, bill_pay_enrolled, direct_deposit_enrolled,
product_count, engagement_intensity, frequency_90d
```
