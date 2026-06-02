import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("/data/github_issues_combined.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nNulls:\n{df.isnull().sum()}")

# ── 2. Compute resolution time (days) ────────────────────────────────────────
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df["closed_at"]  = pd.to_datetime(df["closed_at"],  utc=True, errors="coerce")

df = df.dropna(subset=["created_at", "closed_at"])
df["resolution_days"] = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 86400
df = df[df["resolution_days"] >= 0]

print(f"\nResolution days stats:")
print(df["resolution_days"].describe())

# ── 3. Assign labels ─────────────────────────────────────────────────────────
def classify(days):
    if days < 1:
        return "rapido"
    elif days < 7:
        return "normal"
    elif days < 30:
        return "lento"
    else:
        return "muy_lento"

df["label"] = df["resolution_days"].apply(classify)
print(f"\nClass distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))

# ── 4. Feature engineering ───────────────────────────────────────────────────
df["title"]  = df["title"].fillna("").astype(str)
df["body"]   = df["body"].fillna("").astype(str)
df["text"]   = df["title"] + " " + df["body"]
df["text_len"]   = df["text"].str.len()
df["title_len"]  = df["title"].str.len()
df["label_count"] = df["labels"].fillna("").apply(lambda x: len(x.split("|")) if x else 0)

print(f"\nSample rows:")
print(df[["repo","label","resolution_days","has_assignee","has_milestone","comments"]].head(10))

# ── 5. Save processed ─────────────────────────────────────────────────────────
df.to_csv("/data/github_issues_processed.csv", index=False)
print(f"\nSaved processed: {len(df)} rows")

# ── 6. Save EDA stats as JSON for later use ───────────────────────────────────
stats = {
    "total_issues": int(len(df)),
    "repos": int(df["repo"].nunique()),
    "class_counts": df["label"].value_counts().to_dict(),
    "class_pcts": df["label"].value_counts(normalize=True).round(3).to_dict(),
    "resolution_stats": df["resolution_days"].describe().round(2).to_dict(),
    "top_repos": df["repo"].value_counts().head(10).to_dict(),
}
with open("/data/eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nEDA stats saved.")
