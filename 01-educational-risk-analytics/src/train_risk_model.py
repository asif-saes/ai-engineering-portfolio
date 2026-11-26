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

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
OUT_DIR = Path("01-educational-risk-analytics/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(URL, sep=";")
df["study_failure_ratio"] = df["studytime"] / (df["failures"] + 1.0)
df["absence_rate_proxy"] = df["absences"] / (df["studytime"] + 1.0)
df["at_risk"] = (df["G3"] < 10).astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
regimes = {
    "Early-Warning (No G1/G2)": [c for c in df.columns if c not in {"G1", "G2", "G3", "at_risk"}],
    "Mid-Term (With G1 & G2)": [c for c in df.columns if c not in {"G3", "at_risk"}],
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
        if regime == "Early-Warning (No G1/G2)" and name == "HistGradientBoosting":
            early_preds = preds

pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_comparison.csv", index=False)

cm = confusion_matrix(df["at_risk"], early_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On-Track", "At-Risk"])
fig, ax = plt.subplots(figsize=(5.5, 4.5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Project 1: Early-Warning At-Risk (5-Fold OOF)")
plt.tight_layout()
plt.savefig(OUT_DIR / "risk_confusion_matrix.png", dpi=180)
plt.close()
print(pd.DataFrame(rows).to_string(index=False))
