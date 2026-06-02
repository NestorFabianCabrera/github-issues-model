import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/data/github_issues_processed.csv")
df["text"]         = df["text"].fillna("").astype(str)
df["labels"]       = df["labels"].fillna("").astype(str)
df["has_assignee"] = df["has_assignee"].fillna(0).astype(int)
df["has_milestone"]= df["has_milestone"].fillna(0).astype(int)
df["comments"]     = df["comments"].fillna(0).astype(int)
df["day_of_week"]  = df["day_of_week"].fillna(0).astype(int)
df["hour_of_day"]  = df["hour_of_day"].fillna(0).astype(int)
df["text_len"]     = df["text"].str.len()
df["title_len"]    = df["title"].fillna("").str.len()
df["label_count"]  = df["labels"].apply(lambda x: len([i for i in x.split("|") if i]) if x else 0)

NUM_FEATURES = ["has_assignee","has_milestone","comments","day_of_week",
                "hour_of_day","text_len","title_len","label_count"]

X_text = df["text"]
X_num  = df[NUM_FEATURES]
y      = df["label"]

print(f"Dataset: {len(df)} rows | Classes: {y.value_counts().to_dict()}")

# ── Custom transformer for numeric features ───────────────────────────────────
class NumericSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.values if hasattr(X, "values") else X

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# ── Split ─────────────────────────────────────────────────────────────────────
X = pd.concat([X_text.rename("text"), X_num], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Pipeline ──────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train["text"])
X_test_tfidf  = tfidf.transform(X_test["text"])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[NUM_FEATURES])
X_test_num  = scaler.transform(X_test[NUM_FEATURES])

X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_test_final  = hstack([X_test_tfidf,  csr_matrix(X_test_num)])

# ── Train Random Forest ───────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_final, y_train)
y_pred_rf = rf.predict(X_test_final)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average="weighted")
print(f"Random Forest — Accuracy: {acc_rf:.3f} | F1 weighted: {f1_rf:.3f}")
print(classification_report(y_test, y_pred_rf))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_rf, labels=["rapido","normal","lento","muy_lento"])
print(f"Confusion matrix:\n{cm}")

# ── Feature importance (top numeric) ─────────────────────────────────────────
feature_names = tfidf.get_feature_names_out().tolist() + NUM_FEATURES
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
top_features = [(feature_names[i], round(float(importances[i]), 5)) for i in top_idx]
print(f"\nTop 20 features: {top_features}")

# ── Save model artifacts ──────────────────────────────────────────────────────
artifacts = {
    "tfidf": tfidf,
    "scaler": scaler,
    "model": rf,
    "num_features": NUM_FEATURES,
    "classes": ["rapido", "normal", "lento", "muy_lento"],
}
with open("/data/model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# ── Save metrics JSON ─────────────────────────────────────────────────────────
report = classification_report(y_test, y_pred_rf, output_dict=True)
metrics = {
    "accuracy": round(acc_rf, 4),
    "f1_weighted": round(f1_rf, 4),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "top_features": top_features,
    "class_labels": ["rapido","normal","lento","muy_lento"],
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}
with open("/data/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel and metrics saved.")
