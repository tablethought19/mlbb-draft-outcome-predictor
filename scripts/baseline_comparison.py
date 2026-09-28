import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ── Load data (same files/order used in train_models.py) ─────────
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()

# ── Recreate the exact same train/test split used in train_models.py ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load the deployed model ────────────────────────────────────────
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# ── Model predictions on the full test set ─────────────────────────
y_pred = model.predict(X_test)
model_acc = accuracy_score(y_test, y_pred)
model_f1 = f1_score(y_test, y_pred)

# ── Naive baselines ──────────────────────────────────────────────
majority_class = y_train.mode()[0]  # whichever class (0 or 1) is more common in training
baseline_preds = pd.Series([majority_class] * len(y_test), index=y_test.index)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)

print(f"Full test set size: {len(y_test)}")
print(f"Majority class in training set: {'Team 1' if majority_class == 1 else 'Team 2'}")
print()
print(f"{'':20} {'Accuracy':>10} {'F1-score':>10}")
print(f"{'Random Forest':20} {model_acc:>10.4f} {model_f1:>10.4f}")
print(f"{'Naive baseline':20} {baseline_acc:>10.4f} {baseline_f1:>10.4f}")
print()
print(f"Improvement over baseline: {(model_acc - baseline_acc)*100:+.2f} percentage points (accuracy)")
from sklearn.metrics import classification_report, f1_score

print("\n=== Random Forest — per-class breakdown ===")
print(classification_report(y_test, y_pred, target_names=["Team 2 win", "Team 1 win"]))

print("=== Naive baseline — per-class breakdown ===")
print(classification_report(y_test, baseline_preds, target_names=["Team 2 win", "Team 1 win"], zero_division=0))

# Macro F1 = average of F1 across BOTH classes, not just class 1
model_macro_f1 = f1_score(y_test, y_pred, average='macro')
baseline_macro_f1 = f1_score(y_test, baseline_preds, average='macro', zero_division=0)
print(f"Random Forest macro-F1 (both classes averaged): {model_macro_f1:.4f}")
print(f"Naive baseline macro-F1 (both classes averaged): {baseline_macro_f1:.4f}")