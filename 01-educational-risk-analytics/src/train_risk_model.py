from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT_DIR = Path("01-educational-risk-analytics/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Create a reproducible synthetic CS/math student dataset (no external URL needed)
rng = np.random.default_rng(42)
n = 400

studytime = rng.integers(1, 5, size=n)           # 1 (<2h) to 4 (>10h)
failures = rng.choice([0, 1, 2, 3], size=n, p=[0.72, 0.16, 0.08, 0.04])
absences = rng.poisson(lam=6, size=n).clip(0, 35)
homework_rate = rng.beta(a=5.0, b=2.0, size=n).round(2)
quiz_avg = np.clip(rng.normal(loc=12.0, scale=3.2, size=n), 0, 20).round(1)
school_support = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
internet = rng.choice(["yes", "no"], size=n, p=[0.85, 0.15])

# Interim / midterm scores (used only in Mid-Term regime)
midterm_1 = np.clip(0.45 * quiz_avg + 1.1 * studytime - 1.8 * failures - 0.12 * absences + 5.0 * homework_rate + rng.normal(0, 1.5, n), 0, 20)
midterm_2 = np.clip(0.75 * midterm_1 + 0.25 * quiz_avg - 0.08 * absences + rng.normal(0, 1.2, n), 0, 20)

# Final score on a 0-20 scale (<10 = at-risk)
final_score = np.clip(0.65 * midterm_2 + 0.25 * midterm_1 + 1.5 * homework_rate - 0.6 * failures + rng.normal(0, 1.4, n), 0, 20)

df = pd.DataFrame({
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "homework_rate": homework_rate,
    "quiz_avg": quiz_avg,
    "school_support": school_support,
    "internet": internet,
    "midterm_1": midterm_1.round(1),
    "midterm_2": midterm_2.round(1),
    "final_score": final_score.round(1),
})

# 2. Domain feature engineering
df["study_failure_ratio"] = df["studytime"] / (df["failures"] + 1.0)
df["engagement_index"] = df["homework_rate"] * (df["studytime"] + 1.0) / (df["absences"] + 1.0)
df["at_risk"] = (df["final_score"] < 10.0).astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
regimes = {
    "Early-Warning (No Midterms)": [
        c for c in df.columns if c not in {"midterm_1", "midterm_2", "final_score", "at_risk"}
    ],
    "Mid-Term (With Midterms)": [
        c for c in df.columns if c not in {"final_score", "at_risk"}
    ],
}

rows = []
early_preds = None

for regime, cols in regimes.items():
    X = df[cols].copy()
    y = df["at_risk"].astype(int)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42),
    }

    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        rows.append({
            "regime": regime,
            "model": name,
            "recall_at_risk": round(recall_score(y, preds), 3),
            "precision_at_risk": round(precision_score(y, preds), 3),
            "f1_at_risk": round(f1_score(y, preds), 3),
        })
        if regime == "Early-Warning (No Midterms)" and name == "HistGradientBoosting":
            early_preds = preds

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_DIR / "metrics_comparison.csv", index=False)

cm = confusion_matrix(df["at_risk"], early_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On-Track", "At-Risk"])
fig, ax = plt.subplots(figsize=(5.5, 4.5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Project 1: Early-Warning At-Risk (5-Fold OOF)")
plt.tight_layout()
plt.savefig(OUT_DIR / "risk_confusion_matrix.png", dpi=180)
plt.close()

print(metrics_df.to_string(index=False))
