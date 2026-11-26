from pathlib import Path
import matplotlib.pyplot as plt
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

categories = ["comp.graphics", "sci.space", "rec.sport.hockey", "talk.politics.misc"]
dataset = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
texts = pd.Series(dataset.data).fillna("")
y = pd.Series(dataset.target)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
svc_preds = None

models = {
    "MultinomialNB": MultinomialNB(alpha=0.3),
    "LinearSVC": LinearSVC(C=1.0, random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")),
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

pd.DataFrame(rows).to_csv(OUT_DIR / "nlp_metrics.csv", index=False)

cm = confusion_matrix(y, svc_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp.plot(ax=ax, cmap="Purples", values_format="d", xticks_rotation=25)
ax.set_title("Project 3: Topic/Ticket Triage Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "nlp_confusion_matrix.png", dpi=180)
plt.close()
print(pd.DataFrame(rows).to_string(index=False))
