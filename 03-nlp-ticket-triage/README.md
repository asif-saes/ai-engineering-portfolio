# Project 3: Multiclass Support Ticket Triage (NLP)

## Problem & Framing
Classifies text tickets/topics across multiple categories using 5-fold stratified cross-validation, tracking both **macro F1** (equal weight across classes) and **weighted F1** (accounts for class support).

## Pipeline & Features
- **Text representation:** `TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")` inside a `Pipeline`.
- **Models compared:** `MultinomialNB(alpha=0.3)` vs. `LinearSVC(C=1.0)`.

## 5-Fold Out-of-Fold Results
| Model | Macro F1 | Weighted F1 |
| :--- | :---: | :---: |
| MultinomialNB | 0.923 | 0.924 |
| LinearSVC | 0.919 | 0.921 |

## Limitations & Next Steps
- TF-IDF captures lexical n-grams well but does not model deep context, slang, or multilingual phrasing without pretrained transformers or domain-specific training data.
