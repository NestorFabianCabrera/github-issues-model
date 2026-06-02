import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/data/github_issues_processed.csv")
df["title"]        = df["title"].fillna("").astype(str)
df["body"]         = df["body"].fillna("").astype(str)
df["labels"]       = df["labels"].fillna("").astype(str)
df["has_assignee"] = df["has_assignee"].fillna(0).astype(int)
df["has_milestone"]= df["has_milestone"].fillna(0).astype(int)
df["comments"]     = df["comments"].fillna(0).astype(int)
df["day_of_week"]  = df["day_of_week"].fillna(0).astype(int)
df["hour_of_day"]  = df["hour_of_day"].fillna(0).astype(int)

# ── Richer features ───────────────────────────────────────────────────────────
df["title_word_count"]  = df["title"].str.split().str.len()
df["body_word_count"]   = df["body"].str.split().str.len()
df["has_code_block"]    = df["body"].str.contains("```").astype(int)
df["has_url"]           = df["body"].str.contains("http").astype(int)
df["has_question"]      = df["title"].str.contains(r"\?").astype(int)
df["label_count"]       = df["labels"].apply(lambda x: len([i for i in x.split("|") if i]) if x else 0)
df["title_len"]         = df["title"].str.len()

NUM_FEATURES = [
    "has_assignee", "has_milestone", "comments", "day_of_week",
    "hour_of_day", "title_word_count", "body_word_count",
    "has_code_block", "has_url", "has_question",
    "label_count", "title_len"
]

# ── 3-class target (más balanceado y realista) ────────────────────────────────
# rapido: <1d  |  normal: 1-14d  |  lento: >14d
def classify3(days):
    if days < 1:    return "rapido"
    elif days < 14: return "normal"
    else:           return "lento"

df["label3"] = df["resolution_days"].apply(classify3)
print("3-class distribution:")
print(df["label3"].value_counts())
print(df["label3"].value_counts(normalize=True).round(3))

# ── Split ─────────────────────────────────────────────────────────────────────
X_title = df["title"]
X_body  = df["body"]
X_num   = df[NUM_FEATURES]
y       = df["label3"]

# Build combined text: title weighted more
df["text_combined"] = df["title"] + " " + df["title"] + " " + df["body"]

X = pd.concat([df["text_combined"].rename("text"), X_num], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── TF-IDF + numeric ──────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=3,
)
X_train_tfidf = tfidf.fit_transform(X_train["text"])
X_test_tfidf  = tfidf.transform(X_test["text"])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[NUM_FEATURES])
X_test_num  = scaler.transform(X_test[NUM_FEATURES])

X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_test_final  = hstack([X_test_tfidf,  csr_matrix(X_test_num)])

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\nTraining Random Forest v2...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_final, y_train)
y_pred = rf.predict(X_test_final)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")
print(f"RF v2 — Accuracy: {acc:.3f} | F1 weighted: {f1:.3f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["rapido","normal","lento"])
print(f"Confusion matrix:\n{cm}")

# ── Top features ──────────────────────────────────────────────────────────────
feature_names = tfidf.get_feature_names_out().tolist() + NUM_FEATURES
importances   = rf.feature_importances_
top_idx       = np.argsort(importances)[::-1][:20]
top_features  = [(feature_names[i], round(float(importances[i]), 5)) for i in top_idx]
print(f"\nTop 20 features: {top_features}")

# ── Save ──────────────────────────────────────────────────────────────────────
artifacts = {
    "tfidf": tfidf,
    "scaler": scaler,
    "model": rf,
    "num_features": NUM_FEATURES,
    "classes": ["rapido", "normal", "lento"],
}
with open("/data/model_artifacts_v2.pkl", "wb") as f:
    pickle.dump(artifacts, f)

report = classification_report(y_test, y_pred, output_dict=True)
metrics = {
    "accuracy": round(acc, 4),
    "f1_weighted": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "top_features": top_features,
    "class_labels": ["rapido", "normal", "lento"],
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "class_counts": df["label3"].value_counts().to_dict(),
}
with open("/data/model_metrics_v2.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved v2 model and metrics.")
