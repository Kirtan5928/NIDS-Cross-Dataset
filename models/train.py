import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import time

print("Loading merged dataset...")
df = pd.read_csv("data/merged_dataset.csv")
print(f"Shape: {df.shape}")

# ── Prepare features and label ──────────────────────────────────────────────
X = df.drop(columns=["Label", "source"])
y = df["Label"]

# ── Train/test split — 80% train, 20% test ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

# ── Random Forest ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,        # use all CPU cores
    class_weight="balanced"  # handles class imbalance
)

start = time.time()
rf.fit(X_train, y_train)
print(f"Training time: {round(time.time() - start, 2)}s")

rf_preds = rf.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_test, rf_preds) * 100, 2)}%")
print(f"\nClassification Report:\n{classification_report(y_test, rf_preds)}")

# ── XGBoost ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric="logloss",
    verbosity=0
)

start = time.time()
xgb.fit(X_train, y_train)
print(f"Training time: {round(time.time() - start, 2)}s")

xgb_preds = xgb.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_test, xgb_preds) * 100, 2)}%")
print(f"\nClassification Report:\n{classification_report(y_test, xgb_preds)}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Random Forest Accuracy : {round(accuracy_score(y_test, rf_preds) * 100, 2)}%")
print(f"XGBoost Accuracy       : {round(accuracy_score(y_test, xgb_preds) * 100, 2)}%")

import joblib

# Save both models
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
print("\nModels saved to models/")