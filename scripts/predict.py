import pandas as pd
import numpy as np
import pickle

# ── Load model and feature columns ───────────────────────────────
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns from training data
X = pd.read_csv("data/processed/X.csv")
FEATURE_COLS = X.columns.tolist()

# ── Hero name normaliser ──────────────────────────────────────────
def hero_to_col(name):
    return str(name).lower()\
        .replace(" ", "_")\
        .replace("'", "")\
        .replace(".", "")\
        .replace("-", "_")

# ── Patch encoding ────────────────────────────────────────────────
PATCH_ORDER = [
    "1.8.30B", "1.8.56B", "1.8.66", "1.8.78", "1.8.92B",
    "1.9.06",  "1.9.06C", "1.9.42", "1.9.42E","1.9.64",
    "1.9.64B", "1.9.91",  "1.9.99", "1.9.99a","2.1.18",
    "2.1.40a", "2.1.61"
]
PATCH_MAP = {p: i for i, p in enumerate(PATCH_ORDER)}

def predict_draft(
    team1_picks:   list,
    team2_picks:   list,
    team1_bans:    list,
    team2_bans:    list,
    team1_is_blue: bool,
    patch:         str
):
    # Build empty feature row
    row = {col: 0 for col in FEATURE_COLS}

    # Encode picks
    for hero in team1_picks:
        col = f"pick_{hero_to_col(hero)}"
        if col in row:
            row[col] = 1
        else:
            print(f"Warning: unknown pick hero '{hero}' (col: {col}) — skipping")

    for hero in team2_picks:
        col = f"pick_{hero_to_col(hero)}"
        if col in row:
            row[col] = -1
        else:
            print(f"Warning: unknown pick hero '{hero}' (col: {col}) — skipping")

    # Encode bans
    for hero in team1_bans:
        col = f"ban_{hero_to_col(hero)}"
        if col in row:
            row[col] = 1
        else:
            print(f"Warning: unknown ban hero '{hero}' (col: {col}) — skipping")

    for hero in team2_bans:
        col = f"ban_{hero_to_col(hero)}"
        if col in row:
            row[col] = -1
        else:
            print(f"Warning: unknown ban hero '{hero}' (col: {col}) — skipping")

    # Encode side
    row['team1_is_blue'] = 1 if team1_is_blue else 0

    # Encode patch
    if patch in PATCH_MAP:
        row['patch_encoded'] = PATCH_MAP[patch]
    else:
        print(f"Warning: unknown patch '{patch}' — defaulting to latest")
        row['patch_encoded'] = max(PATCH_MAP.values())

    # Predict
    X_input = pd.DataFrame([row])[FEATURE_COLS]
    prob = model.predict_proba(X_input)[0]
    pred = model.predict(X_input)[0]

    return {
        "prediction":     "Team 1 wins" if pred == 1 else "Team 2 wins",
        "team1_win_prob": round(prob[1] * 100, 1),
        "team2_win_prob": round(prob[0] * 100, 1),
    }

# ── Test match ────────────────────────────────────────────────────
if __name__ == "__main__":
    team1_picks = [
        "phoveus",
        "harley",
        "zhuxin",
        "claude",
        "hylos",
    ]
    team2_picks = [
        "lapu-lapu",
        "fredrinn",
        "yve",
        "karrie",
        "hilda",
    ]
    team1_bans = [
        "kalea",
        "guinevere",
        "marcel",
        "uranus",
        "moskov",
    ]
    team2_bans = [
        "freya",
        "baxia",
        "suyou",
        "gord",
        "gloo",
    ]

    result = predict_draft(
        team1_picks   = team1_picks,
        team2_picks   = team2_picks,
        team1_bans    = team1_bans,
        team2_bans    = team2_bans,
        team1_is_blue = True,
        patch         = "2.1.61"
    )

    print("\n=== Draft Prediction ===")
    print(f"Prediction:       {result['prediction']}")
    print(f"Team 1 win prob:  {result['team1_win_prob']}%")
    print(f"Team 2 win prob:  {result['team2_win_prob']}%")