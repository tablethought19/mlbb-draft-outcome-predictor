# MLBB Draft Outcome Predictor

An AI-based system that predicts the win probability of professional Mobile Legends: Bang Bang (MLBB) matches using only pre-match draft composition; hero picks, bans, side, and patch version — with no in-game performance data required.

Built as a Final Year Project for BSc (Hons) Computer Science (Artificial Intelligence) at Asia Pacific University of Technology and Innovation (APU).

## Features

- Draft Predictor: select a full draft (5 picks + 5 bans per team, side, patch) and get a win probability, with SHAP-based explainability showing which picks/bans influenced the prediction and why
- Meta Dashboard:  6 analytical views: Most Picked Heroes, Most Banned Heroes, Hero Synergy, Patch Meta Shift, Counter Picks, and Hero Win Rate

## Data

Match data (3,479 professional matches, Nov 2023 – Jun 2026) was collected from [Liquipedia](https://liquipedia.net/mobilelegends/) via the LiquipediaDB API, covering MPL PH/ID/MY/SG/MENA, Türkiye, Cambodia, and the M5–M7 World Championships. Licensed under CC-BY-SA 3.0.

## Model

Three models were trained and compared: Logistic Regression, Random Forest, and XGBoost. Random Forest was selected as the deployed model, achieving a macro-averaged F1-score of 0.5718 (substantially outperforming a naive baseline of 0.3501), and 55.17% accuracy in a case study against 29 held-out MPL Philippines Season 17 matches.

**Limitation:** predictions are based on draft composition only, in-game execution, player skill, and strategy are not accounted for.

## Project Structure

```
FYP MLBB AI/
├── app/main.py                  # Streamlit application
├── data/raw/                    # Raw and cleaned match data
├── data/processed/               # Feature matrix (X, y)
├── models/                      # Trained model files (.pkl)
└── scripts/
    ├── fetch_matches.py          # Collects match data from Liquipedia API
    ├── clean_data.py             # Cleans and filters raw data
    ├── feature_engineering.py    # Encodes draft data into features
    ├── train_models.py           # Trains and compares models
    ├── predict.py                # Standalone prediction script
    ├── case_study_validation.py  # Validates model against held-out matches
    ├── baseline_comparison.py    # Compares model against naive baseline
    ├── explore_data.py           # Data exploration
    └── test_shap.py              # SHAP explainability testing
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your own Liquipedia API key:
   ```
   LIQUIPEDIA_API_KEY=your_key_here
   ```
   Request an API key from [Liquipedia](https://liquipedia.net/api-terms-of-service/).
4. Run the app:
   ```
   streamlit run app/main.py
   ```

## Rebuilding the Pipeline

To re-fetch and re-process the data from scratch:
```
python scripts/fetch_matches.py
python scripts/clean_data.py
python scripts/feature_engineering.py
python scripts/train_models.py
```

## Acknowledgements

Match data sourced from Liquipedia, with thanks to Radoslav Kolev, Liquipedia's Product Owner, for approving free-tier API access for this project.

## Author

Kobalen Khaniska Arasen — BSc (Hons) Computer Science (Artificial Intelligence), APU
Supervisor: Dr. Maythem Kamal Abbas


