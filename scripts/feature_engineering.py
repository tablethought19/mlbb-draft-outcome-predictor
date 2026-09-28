import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/mlbb_drafts_clean.csv")

# ── Helper: clean hero name → valid column name ────────────────────
def hero_to_col(name):
    return str(name).lower()\
        .replace(" ", "_")\
        .replace("'", "")\
        .replace(".", "")\
        .replace("-", "_")

# ── Pick columns ───────────────────────────────────────────────────
pick_cols = [f"team{t}pick{n}" for t in [1,2] for n in range(1,6)]
ban_cols  = [f"team{t}ban{n}"  for t in [1,2] for n in range(1,6)]

# Get all unique heroes across picks only (bans use same hero pool)
all_heroes = sorted(set(
    pd.concat([df[c] for c in pick_cols + ban_cols]).dropna().unique()
))
all_heroes_clean = [hero_to_col(h) for h in all_heroes]

print(f"Total unique heroes (picks + bans): {len(all_heroes)}")

# ── One-hot encode picks ───────────────────────────────────────────
# For picks: 1 if hero appears in team1 picks, -1 if in team2 picks
for hero, col in zip(all_heroes, all_heroes_clean):
    team1_picked = df[pick_cols[:5]].isin([hero]).any(axis=1).astype(int)
    team2_picked = df[pick_cols[5:]].isin([hero]).any(axis=1).astype(int)
    df[f"pick_{col}"] = team1_picked - team2_picked

# ── One-hot encode bans ────────────────────────────────────────────
# Bans: 1 if hero banned by team1, -1 if banned by team2
team1_ban_cols = [f"team1ban{n}" for n in range(1,6)]
team2_ban_cols = [f"team2ban{n}" for n in range(1,6)]

for hero, col in zip(all_heroes, all_heroes_clean):
    team1_banned = df[team1_ban_cols].isin([hero]).any(axis=1).astype(int)
    team2_banned = df[team2_ban_cols].isin([hero]).any(axis=1).astype(int)
    df[f"ban_{col}"] = team1_banned - team2_banned

# ── Encode side ───────────────────────────────────────────────────
# team1side: blue=1, red=0
df['team1_is_blue'] = (df['team1side'].str.lower() == 'blue').astype(int)

# ── Encode patch ──────────────────────────────────────────────────
# Label encode in chronological order
patch_order = sorted(df['patch'].dropna().unique())
patch_map = {p: i for i, p in enumerate(patch_order)}
df['patch_encoded'] = df['patch'].map(patch_map)
print(f"\nPatch encoding:")
for p, i in patch_map.items():
    print(f"  {p} → {i}")

# ── Build final feature matrix ────────────────────────────────────
pick_feature_cols = [f"pick_{hero_to_col(h)}" for h in all_heroes]
ban_feature_cols  = [f"ban_{hero_to_col(h)}"  for h in all_heroes]
feature_cols = pick_feature_cols + ban_feature_cols + ['team1_is_blue', 'patch_encoded']

X = df[feature_cols]
y = df['winner'].map({1: 1, 2: 0})  # team1 wins = 1, team2 wins = 0

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label distribution:\n{y.value_counts()}")
print(f"\nAny nulls in X: {X.isnull().sum().sum()}")
print(f"Any nulls in y: {y.isnull().sum()}")

# ── Save ──────────────────────────────────────────────────────────
X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)
print("\nSaved to data/processed/X.csv and y.csv")