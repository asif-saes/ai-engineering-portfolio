# AI Engineering Portfolio

Reproducible machine learning and NLP projects built in Python (`scikit-learn`, `pandas`, `numpy`, `matplotlib`) with 5-fold stratified cross-validation, saved output tables, and diagnostic charts.

## Projects

1. **Educational Risk & Outcome Analytics** (`01-educational-risk-analytics/`)
   - Compares Early-Warning (no midterms) vs. Mid-Term (with midterms) risk detection (`final_score < 10.0`).
   - Best Early-Warning F1: `0.765` (`LogisticRegression`, recall `0.757`, precision `0.773`); Best Mid-Term F1: `0.839` (`LogisticRegression`).
2. **Customer Churn & Retention Modeling** (`02-customer-churn-analytics/`)
   - Evaluates churn prediction at a `0.40` probability threshold using engineered billing/tenure ratios.
   - Best 5-fold result: `LogisticRegression` (`ROC-AUC = 0.709`, `recall = 0.636`, `precision = 0.564`, `F1 = 0.598`).
3. **Multiclass Support Ticket Triage (NLP)** (`03-nlp-ticket-triage/`)
   - Combines unigram/bigram `TfidfVectorizer` with `MultinomialNB` and `LinearSVC`.
   - Best 5-fold result: `MultinomialNB` (`macro_f1 = 0.923`, `weighted_f1 = 0.924`).

## Notes on Data & Scope
- Scripts include reproducible synthetic fallbacks (`seed=42`) so all pipelines execute cleanly without broken external URLs; synthetic experiments validate modeling and evaluation workflows rather than real-world production telemetry.
