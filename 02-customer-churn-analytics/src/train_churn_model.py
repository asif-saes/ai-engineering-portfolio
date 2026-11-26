from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4r/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = Path("02-customer-churn-analytics/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_churn_data() -> tuple[pd.DataFrame, pd.Series, str]:
    try:
        df = pd.read_csv(URL)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["avg_monthly_from_total"] = df["TotalCharges"] / (df["tenure"] + 1.0)
        df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_from_total"] + 1.0)
        y = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
        X = df.drop(columns=["customerID", "Churn"], errors="ignore")
        return X, y, "external_telco_csv"
    except Exception:
        rng = np.random.default_rng(42)
        n = 600
        tenure = rng.integers(1, 61, size=n)
        monthly_charges = np.round(rng.uniform(25.0, 115.0, size=n), 2)
        support_tickets = rng.poisson(lam=1.4, size=n).clip(0, 8)
        contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
        internet_service = rng.choice(["Fiber", "DSL", "None"], size=n, p=[0.5, 0.35, 0.15])
        paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])

        total_charges = np.round(monthly_charges * tenure * rng.uniform(0.92, 1.05, size=n), 2)
        avg_monthly_from_total = total_charges / (tenure + 1.0)
        charge_ratio = monthly_charges / (avg_monthly_from_total + 1.0)

        contract_risk = np.where(contract == "Month-to-month", 1.1, np.where(contract == "One year", 0.2, -0.4))
        logit = (
            -1.3
            - 0.045 * tenure
            + 0.018 * monthly_charges
            + 0.35 * support_tickets
            + contract_risk
            + rng.normal(0, 0.65, size=n)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.uniform(0, 1, size=n) < prob).astype(int), name="churn")

        X = pd.DataFrame({
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "support_tickets": support_tickets,
            "avg_monthly_from_total": avg_monthly_from_total.round(2),
            "charge_ratio": charge_ratio.round(3),
            "Contract": contract,
            "InternetService": internet_service,
            "PaperlessBilling": paperless_billing,
        })
        return X, y, "synthetic_fallback_seed_42"


X, y, source_used = load_churn_data()
print(f"Dataset source: {source_used} | Rows: {len(X)} | Churn rate: {y.mean():.3f}")

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
threshold = 0.40
rows = []
best_probs = None

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=200, random_state=42),
}

for name, est in models.items():
    pipe = Pipeline([("prep", prep), ("model", est)])
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    preds = (probs >= threshold).astype(int)
    rows.append({
        "model": name,
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(y, probs), 3),
        "recall_churn": round(recall_score(y, preds), 3),
        "precision_churn": round(precision_score(y, preds), 3),
        "f1_churn": round(f1_score(y, preds), 3),
    })
    if name == "HistGradientBoosting":
        best_probs = probs

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_DIR / "churn_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
PrecisionRecallDisplay.from_predictions(y, best_probs, ax=ax, name="HistGradientBoosting")
ax.set_title("Project 2: Churn Precision-Recall (5-Fold OOF)")
plt.tight_layout()
plt.savefig(OUT_DIR / "churn_precision_recall.png", dpi=180)
plt.close()

print("\n" + metrics_df.to_string(index=False))
