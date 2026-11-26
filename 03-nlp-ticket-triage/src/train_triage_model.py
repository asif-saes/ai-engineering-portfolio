from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

OUT_DIR = Path("03-nlp-ticket-triage/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_triage_data() -> tuple[pd.Series, pd.Series, list[str], str]:
    categories = [
        "comp.graphics",
        "sci.space",
        "rec.sport.hockey",
        "talk.politics.misc",
    ]
    try:
        dataset = fetch_20newsgroups(
            subset="all",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        texts = pd.Series(dataset.data).fillna("")
        y = pd.Series(dataset.target)
        valid = texts.str.strip().str.len() > 0
        return (
            texts[valid].reset_index(drop=True),
            y[valid].reset_index(drop=True),
            list(dataset.target_names),
            "fetch_20newsgroups",
        )
    except Exception:
        rng = np.random.default_rng(42)
        label_names = ["billing_account", "technical_bug", "feature_request", "shipping_access"]
        templates = {
            "billing_account": [
                "Charged twice on my invoice this month and need a refund for duplicate billing.",
                "Cannot update my payment card or download tax receipts from account billing.",
                "Subscription auto-renewed unexpectedly; please explain the invoice charge.",
            ],
            "technical_bug": [
                "App crashes with a 500 error when exporting CSV reports from the dashboard.",
                "Login page freezes and throws a timeout exception after submitting credentials.",
                "Chart rendering breaks on mobile browser and fails to load recent logs.",
            ],
            "feature_request": [
                "Would love to export filtered analytics tables directly to PDF or Excel.",
                "Please add dark mode and custom webhook notifications for team alerts.",
                "Feature suggestion: allow bulk role assignment and saved filter presets.",
            ],
            "shipping_access": [
                "Order delivery is delayed past the estimated arrival window and tracking stalled.",
                "Team member cannot access workspace permissions or invite link expired.",
                "Package arrived damaged and account access is locked after password reset.",
            ],
        }
        rows_text, rows_label = [], []
        for idx, name in enumerate(label_names):
            for _ in range(120):
                base = rng.choice(templates[name])
                suffix = f" Ticket ref #{rng.integers(1000, 9999)} priority {rng.choice(['normal', 'high'])}."
                rows_text.append(base + suffix)
                rows_label.append(idx)
        order = rng.permutation(len(rows_text))
        texts = pd.Series([rows_text[i] for i in order])
        y = pd.Series([rows_label[i] for i in order])
        return texts, y, label_names, "synthetic_support_tickets_seed_42"


texts, y, target_names, source_used = load_triage_data()
print(f"Dataset source: {source_used} | Samples: {len(texts)} | Classes: {len(target_names)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
svc_preds = None

models = {
    "MultinomialNB": MultinomialNB(alpha=0.3),
    "LinearSVC": LinearSVC(C=1.0, random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=15000,
                stop_words="english",
            ),
        ),
        ("clf", clf),
    ])
    preds = cross_val_predict(pipe, texts, y, cv=cv, n_jobs=-1)
    rows.append({
        "model": name,
        "macro_f1": round(f1_score(y, preds, average="macro"), 3),
        "weighted_f1": round(f1_score(y, preds, average="weighted"), 3),
    })
    if name == "LinearSVC":
        svc_preds = preds

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_DIR / "nlp_metrics.csv", index=False)

cm = confusion_matrix(y, svc_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp.plot(ax=ax, cmap="Purples", values_format="d", xticks_rotation=25)
ax.set_title("Project 3: Topic/Ticket Triage Confusion Matrix (5-Fold OOF)")
plt.tight_layout()
plt.savefig(OUT_DIR / "nlp_confusion_matrix.png", dpi=180)
plt.close()

print("\n" + metrics_df.to_string(index=False))
