import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle
import os

# ── Load data ─────────────────────────────────────────────────────
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Label distribution:\n{y.value_counts()}\n")

# ── Train/test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

# ── Cross-validation setup ────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Models ───────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
}

results = {}

for name, model in models.items():
    print(f"=== {name} ===")

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1:        {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    results[name] = {
        "model":        model,
        "cv_mean":      cv_scores.mean(),
        "cv_std":       cv_scores.std(),
        "accuracy":     acc,
        "precision":    prec,
        "recall":       rec,
        "f1":           f1,
    }

# ── Comparison table ──────────────────────────────────────────────
print("\n=== Model Comparison ===")
print(f"{'Model':<25} {'CV Acc':>8} {'Test Acc':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<25} {r['cv_mean']:>8.4f} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

# ── Save best model ───────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
print(f"\nBest model by F1: {best_name} ({results[best_name]['f1']:.4f})")

os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save all models too
for name, r in results.items():
    fname = name.lower().replace(" ", "_")
    with open(f"models/{fname}.pkl", "wb") as f:
        pickle.dump(r['model'], f)

print("Saved all models to models/")