import shap
import pickle
import pandas as pd
import numpy as np

# Load model and data
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

X = pd.read_csv("data/processed/X.csv")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Test on first row
sample = X.iloc[[0]]
shap_values = explainer.shap_values(sample, check_additivity=False)

# Print shape to see what we're working with
print("SHAP values type:", type(shap_values))
print("SHAP values shape:", shap_values.shape)

# Get values for class 1 (team1 wins)
if len(shap_values.shape) == 3:
    vals = shap_values[0, :, 1]  # new format
else:
    vals = shap_values[0]  # old format

feature_names = X.columns.tolist()

top_positive = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)[:5]
top_negative = sorted(zip(feature_names, vals), key=lambda x: x[1])[:5]

print("\nTop 5 factors favouring Team 1:")
for name, val in top_positive:
    print(f"  {name}: {val:.4f}")

print("\nTop 5 factors favouring Team 2:")
for name, val in top_negative:
    print(f"  {name}: {val:.4f}")