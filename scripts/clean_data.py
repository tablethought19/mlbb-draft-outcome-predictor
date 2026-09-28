import pandas as pd

df = pd.read_csv("data/raw/mlbb_drafts_raw.csv")
df['date'] = pd.to_datetime(df['date'])
print(f"Rows before cleaning: {len(df)}")

# ── Date filter ────────────────────────────────────────────────────
# M5 was Nov-Dec 2023, worth including as world championship data
df = df[df['date'] >= '2023-11-01']
print(f"After date filter (2023-11 onward): {len(df)}")

# ── Tournament filter ──────────────────────────────────────────────
# Tournament names verified by querying Liquipedia's /match endpoint 
# with date-range filters to confirm exact naming (e.g. M6 = "M6 World Championship - Swiss Stage")
KEEP_TOURNAMENTS = [
    #Season 17
    "MPL Indonesia Season 17",
    "MPL Philippines Season 17",
    "MPL Malaysia Season 17",
    # MPL Philippines
    "MPL Philippines Season 13 - Regular Season",
    "MPL Philippines Season 13 - Playoffs",
    "MPL Philippines Season 14 - Playoffs",
    "MPL PH Season 14 - Regular Season",
    "MPL Philippines Season 15 - Playoffs",
    "MPL PH Season 15 - Regular Season",
    "MPL Philippines Season 16",
    # MPL Indonesia
    "MPL ID Season 13 - Regular Season",
    "MPL ID Season 13 - Playoffs",
    "MPL ID Season 14 - Regular Season",
    "MPL ID Season 14 - Playoffs",
    "MPL ID Season 15 - Regular Season",
    "MPL ID Season 15 - Playoffs",
    "MPL Indonesia Season 16",
    # MPL Malaysia
    "MPL Malaysia Season 13 - Regular Season",
    "MPL Malaysia Season 13 - Playoffs",
    "MPL Malaysia Season 14 - Regular Season",
    "MPL Malaysia Season 14 - Playoffs",
    "MPL Malaysia Season 15 - Regular Season",
    "MPL Malaysia Season 15 - Playoffs",
    "MPL Malaysia Season 16",
    # MPL Singapore
    "MPL Singapore Season 7 - Regular Season",
    "MPL Singapore Season 7 - Playoffs",
    "MPL Singapore Season 8 - Regular Season",
    "MPL Singapore Season 8 - Playoffs",
    "MPL Singapore Season 9 - Regular Season",
    "MPL Singapore Season 9 - Playoffs",
    "MPL Singapore Season 10",
    "MPL Singapore Season 11",
    # MPL MENA
    "MPL MENA Season 5 - Regular Season",
    "MPL MENA Season 5 - Playoffs",
    "MPL MENA Season 6 - Regular Season",
    "MPL MENA Season 6 - Playoffs",
    "MPL MENA Season 7 - Regular Season",
    "MPL MENA Season 7 - Playoffs",
    "MPL MENA Season 8",
    "MPL MENA Season 9",
    # MPL Turkiye
    "MTC Turkiye Championship Season 6",
    "MTC Turkiye Championship Season 7",
    # MPL Cambodia
    "MPL Cambodia Season 9",
    "MPL Cambodia Season 10",
    # World Championships
    "M5 World Championship - Group Stage",
    "M5 World Championship - Knockout",
    "M5 World Championship - Wildcard",
    "M6 World Championship - Swiss Stage",
    "M6 World Championship - Knockout Stage",
    "M6 World Championship - Wildcard",
    "M7 World Championship",
    # Mid Season Cup
    "Mid Season Cup 2024 - Group Stage",
    "Mid Season Cup 2024 - Playoffs",
    "Mid Season Cup 2024 - Wildcard",
    "Mid Season Cup 2025 - Group Stage",
    "Mid Season Cup 2025 - Wildcard",
    "MLBB Mid Season Cup 2025",
    # Continental
    "MLBB Continental Championships Season 6",
    "MLBB Continental Championships Season 7",
    # IESF
    "IESF World Esports Championship 2024 Group Stage",
    "IESF World Esports Championship 2024 Final Stage",
    "IESF World Esports Championship 2024 South East Asia",
]

before = len(df)
df = df[df['tournament'].isin(KEEP_TOURNAMENTS)]
print(f"After tournament filter: {len(df)}")
print(f"Removed: {before - len(df)}")

# ── Drop thin patches ──────────────────────────────────────────────
MIN_ROWS_PER_PATCH = 5
patch_counts = df['patch'].value_counts()
valid_patches = patch_counts[patch_counts >= MIN_ROWS_PER_PATCH].index
df = df[df['patch'].isin(valid_patches)]
print(f"After dropping thin patches: {len(df)}")

# ── Drop nulls in critical columns ────────────────────────────────
critical_cols = (
    [f"team{t}pick{n}" for t in [1,2] for n in range(1,6)] +
    ['winner', 'patch', 'date']
)
before = len(df)
# ── Backfill missing patches using date ranges ─────────────────────
TOURNAMENT_PATCH_MAP = {
    "MPL Philippines Season 13 - Regular Season": "1.8.66",
    "MPL Malaysia Season 13 - Regular Season":    "1.8.66",
    "MPL MENA Season 5 - Regular Season":         "1.8.66",
    "MPL MENA Season 7 - Regular Season":         "1.9.06",
    "MTC Turkiye Championship Season 6":          "1.9.06",
    "MTC Turkiye Championship Season 7":          "1.9.42",
}

for tournament, patch in TOURNAMENT_PATCH_MAP.items():
    mask = (df['tournament'] == tournament) & (df['patch'].isnull())
    df.loc[mask, 'patch'] = patch
    filled = mask.sum()
    if filled > 0:
        print(f"Filled {filled} null patches for {tournament} → {patch}")
df = df.dropna(subset=critical_cols)
print(f"After dropping nulls: {len(df)} (removed {before - len(df)})")

# ── Normalise text ────────────────────────────────────────────────
hero_cols = (
    [f"team{t}pick{n}" for t in [1,2] for n in range(1,6)] +
    [f"team{t}ban{n}"  for t in [1,2] for n in range(1,6)]
)
for col in hero_cols:
    df[col] = df[col].str.strip().str.lower()

df['winner'] = df['winner'].astype(int)
df['team1side'] = df['team1side'].str.strip().str.lower()
df['team2side'] = df['team2side'].str.strip().str.lower()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n=== Clean dataset summary ===")
print(f"Total rows: {len(df)}")
print(f"\nTournament breakdown:\n{df['tournament'].value_counts().to_string()}")
print(f"\nPatch distribution:\n{df['patch'].value_counts()}")
print(f"\nWinner distribution:\n{df['winner'].value_counts()}")
print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")

df.to_csv("data/raw/mlbb_drafts_clean.csv", index=False)
print("\nSaved to data/raw/mlbb_drafts_clean.csv")