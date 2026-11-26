# Project 2: Customer Churn & Retention Modeling

## Problem & Framing
Predicts customer churn using 5-fold stratified cross-validation and evaluates retention outreach at a `0.40` probability threshold, balancing **recall** (catching churning customers) against **precision** (avoiding unnecessary retention offers).

## Features & Pipeline
- **Engineered features:** `avg_monthly_from_total` (`TotalCharges / (tenure + 1)`) and `charge_ratio` (`MonthlyCharges / (avg_monthly_from_total + 1)`).
- **Preprocessing:** Median imputation + standard scaling for numeric features; mode imputation + one-hot encoding for categorical features inside a `ColumnTransformer` + `Pipeline`.

## 5-Fold Out-of-Fold Results (Threshold = 0.40)
| Model | ROC-AUC | Recall | Precision | F1 |
| :--- | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.709 | 0.636 | 0.564 | 0.598 |
| HistGradientBoosting | 0.644 | 0.572 | 0.524 | 0.547 |

## Limitations & Next Steps
- `LogisticRegression` outperformed `HistGradientBoosting` across all reported metrics at threshold `0.40`, showing why linear baselines should not be dismissed.
- Threshold selection (`0.40`) should be tuned using actual retention campaign cost vs. customer lifetime value before deployment.
