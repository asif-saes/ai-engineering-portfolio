# Project 1: Educational Risk & Outcome Analytics

## Problem & Framing
Identifies students at risk of scoring below `10.0` (on a `0–20` scale) using 5-fold stratified cross-validation. For early intervention, **recall** on at-risk students is prioritized alongside **precision** and **F1** so struggling learners are not overlooked.

## Data & Feature Regimes
- **Dataset:** Reproducible synthetic student dataset (`n = 400`, seed `42`) generated in `src/train_risk_model.py` (not real student records).
- **Engineered features:** `study_failure_ratio` (`studytime / (failures + 1)`) and `engagement_index` (`homework_rate * (studytime + 1) / (absences + 1)`).
- **Early-Warning (No Midterms):** Uses pre-exam study, engagement, failure, and support features without `midterm_1` or `midterm_2`.
- **Mid-Term (With Midterms):** Adds interim `midterm_1` and `midterm_2` scores to measure how later information improves risk detection.

## 5-Fold Out-of-Fold Results
| Regime | Model | Recall | Precision | F1 |
| :--- | :--- | :---: | :---: | :---: |
| Early-Warning (No Midterms) | LogisticRegression | 0.757 | 0.773 | 0.765 |
| Early-Warning (No Midterms) | HistGradientBoosting | 0.693 | 0.784 | 0.736 |
| Mid-Term (With Midterms) | LogisticRegression | 0.841 | 0.837 | 0.839 |
| Mid-Term (With Midterms) | HistGradientBoosting | 0.815 | 0.842 | 0.828 |

## Limitations & Next Steps
- Trained on synthetic data with fixed relationships; real institutional data would require privacy safeguards, calibration checks, and domain validation before classroom use.
