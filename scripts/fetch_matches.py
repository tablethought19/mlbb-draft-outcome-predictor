import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["LIQUIPEDIA_API_KEY"]
BASE_URL = "https://api.liquipedia.net/api/v3/match"
HEADERS = {
    "Authorization": f"Apikey {API_KEY}",
    "Accept-Encoding": "gzip"
}

def parse_games(matches):
    rows = []
    skipped = 0
    for match in matches:
        tournament = match.get("tournament")
        date = match.get("date")
        match_id = match.get("match2id")
        match_patch = match.get("patch")

        for game in match.get("match2games", []):
            extra = game.get("extradata", {})
            patch = game.get("patch") or extra.get("patch") or match_patch

            row = {
                "match_id":   match_id,
                "tournament": tournament,
                "date":       date,
                "game_id":    game.get("match2gameid"),
                "winner":     game.get("winner"),
                "patch":      patch,
                "team1side":  extra.get("team1side"),
                "team2side":  extra.get("team2side"),
                "team1pick1": extra.get("team1champion1"),
                "team1pick2": extra.get("team1champion2"),
                "team1pick3": extra.get("team1champion3"),
                "team1pick4": extra.get("team1champion4"),
                "team1pick5": extra.get("team1champion5"),
                "team2pick1": extra.get("team2champion1"),
                "team2pick2": extra.get("team2champion2"),
                "team2pick3": extra.get("team2champion3"),
                "team2pick4": extra.get("team2champion4"),
                "team2pick5": extra.get("team2champion5"),
                "team1ban1":  extra.get("team1ban1"),
                "team1ban2":  extra.get("team1ban2"),
                "team1ban3":  extra.get("team1ban3"),
                "team1ban4":  extra.get("team1ban4"),
                "team1ban5":  extra.get("team1ban5"),
                "team2ban1":  extra.get("team2ban1"),
                "team2ban2":  extra.get("team2ban2"),
                "team2ban3":  extra.get("team2ban3"),
                "team2ban4":  extra.get("team2ban4"),
                "team2ban5":  extra.get("team2ban5"),
            }

            picks = [row[f"team{t}pick{n}"] for t in [1,2] for n in range(1,6)]
            has_all = all(picks)
            no_defaults = not any(str(p).lower() in ("default", "none", "") for p in picks)
            valid_winner = str(row.get("winner", "")) in ("1", "2")

            if has_all and no_defaults and valid_winner:
                rows.append(row)
            else:
                skipped += 1

    return rows, skipped

def fetch_all(max_pages=20):
    all_matches = []
    offset = 0

    for page in range(max_pages):
        params = {
            "wiki":       "mobilelegends",
            "limit":      1000,
            "offset":     offset,
            "conditions": "[[liquipediatier::1]] OR [[liquipediatier::2]]",
        }
        print(f"Fetching page {page+1} (offset {offset})...")
        r = requests.get(BASE_URL, headers=HEADERS, params=params)

        if r.status_code == 429:
            print("Rate limited, waiting 60s...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            print(f"Error {r.status_code}")
            break

        matches = r.json().get("result", [])
        print(f"  Got {len(matches)} matches")

        if not matches:
            print("No more results.")
            break

        all_matches.extend(matches)

        if len(matches) < 1000:
            print("Last page reached.")
            break

        offset += 1000
        time.sleep(3)

    return all_matches

if __name__ == "__main__":
    matches = fetch_all(max_pages=20)
    print(f"\nTotal matches fetched: {len(matches)}")

    rows, skipped = parse_games(matches)
    print(f"Valid rows: {len(rows)} | Skipped: {skipped}")

    df = pd.DataFrame(rows)
    print(f"\nTournament breakdown:\n{df['tournament'].value_counts().to_string()}")
    print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")

    df.to_csv("data/raw/mlbb_drafts_raw.csv", index=False)
    print("\nSaved to data/raw/mlbb_drafts_raw.csv")