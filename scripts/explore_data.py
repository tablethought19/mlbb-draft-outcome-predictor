import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()
df_clean = pd.read_csv("data/raw/mlbb_drafts_clean.csv")  # same row order as X/y

# Recreate the exact same split used in train_models.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# X_test.index gives back the original row positions — cross-reference against
# the tournament column to find which MPL PH S17 matches are genuinely unseen
test_indices = X_test.index
mpl_ph_s17_test = df_clean.loc[
    df_clean.index.isin(test_indices) & (df_clean['tournament'] == "MPL Philippines Season 17")
]

print(f"MPL PH S17 matches available for case study (test set only): {len(mpl_ph_s17_test)}")