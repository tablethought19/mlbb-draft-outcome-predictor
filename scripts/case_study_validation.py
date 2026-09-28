import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# ── Load data (same files/order used in train_models.py) ─────────
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()
df_clean = pd.read_csv("data/raw/mlbb_drafts_clean.csv")  # same row order as X/y

# ── Recreate the exact same train/test split used in train_models.py ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load the deployed model ────────────────────────────────────────
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# ── Identify MPL PH S17 matches within the held-out test set ──────
test_indices = X_test.index
mpl_ph_s17_mask = df_clean.index.isin(test_indices) & (df_clean['tournament'] == "MPL Philippines Season 17")
case_study_indices = df_clean[mpl_ph_s17_mask].index

print(f"Case study matches found: {len(case_study_indices)}")

# ── Run predictions on each held-out MPL PH S17 match ──────────────
pick_cols = [f"team{t}pick{n}" for t in [1, 2] for n in range(1, 6)]

results = []
for idx in case_study_indices:
    row_features = X.loc[[idx]]
    actual_winner = y.loc[idx]  # 1 = Team 1 win, 0 = Team 2 win

    pred = model.predict(row_features)[0]
    prob = model.predict_proba(row_features)[0]

    match_info = df_clean.loc[idx]
    team1_picks = ", ".join(match_info[pick_cols[:5]].tolist())
    team2_picks = ", ".join(match_info[pick_cols[5:]].tolist())

    results.append({
        "match_id":        match_info.get("match_id", idx),
        "date":            match_info.get("date", ""),
        "patch":           match_info.get("patch", ""),
        "team1_picks":     team1_picks,
        "team2_picks":     team2_picks,
        "actual_winner":   "Team 1" if actual_winner == 1 else "Team 2",
        "predicted_winner":"Team 1" if pred == 1 else "Team 2",
        "team1_win_prob":  round(prob[1] * 100, 1),
        "team2_win_prob":  round(prob[0] * 100, 1),
        "correct":         "Yes" if pred == actual_winner else "No",
    })

results_df = pd.DataFrame(results)

# ── Print a compact summary table (without the long pick columns) ──
print("\n" + results_df[[
    "match_id", "date", "actual_winner", "predicted_winner",
    "team1_win_prob", "team2_win_prob", "correct"
]].to_string(index=False))

# ── Overall case study accuracy ─────────────────────────────────────
correct_count = (results_df["correct"] == "Yes").sum()
total_count = len(results_df)
accuracy = correct_count / total_count
print(f"\nCase study accuracy: {accuracy:.2%} ({correct_count}/{total_count} correct)")

# ── Save full results (with picks/bans) for use in the report ──────
results_df.to_csv("data/processed/case_study_mpl_ph_s17.csv", index=False)
print("Saved to data/processed/case_study_mpl_ph_s17.csv")