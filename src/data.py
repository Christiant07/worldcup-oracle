"""Layer 1 — fetch + cache match data."""

import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_CACHE = DATA_DIR / "results.csv"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# football-data.org uses different names for some nations
_FD_NAME_MAP = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Iran": "IR Iran",
    "Türkiye": "Turkey",
    "North Macedonia": "North Macedonia",
}


def load_results() -> pd.DataFrame:
    """Download international results CSV once; cache to data/results.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    if not RESULTS_CACHE.exists():
        print("[data] Downloading international results dataset…")
        r = requests.get(RESULTS_URL, timeout=60)
        r.raise_for_status()
        RESULTS_CACHE.write_bytes(r.content)
        print(f"[data] Saved to {RESULTS_CACHE}")
    df = pd.read_csv(RESULTS_CACHE, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.lower() == "true"
    return df


def upcoming_fixtures(competition: str = "WC") -> list[dict]:
    """Return upcoming fixtures from football-data.org; [] if unavailable.

    Each fixture: {home_team, away_team, date, neutral}.
    Results are cached per competition so we survive rate limits.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"fixtures_{competition}.json"
    key = os.getenv("FOOTBALL_DATA_API_KEY")

    if key:
        try:
            resp = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/{competition}/matches",
                headers={"X-Auth-Token": key},
                params={"status": "SCHEDULED"},
                timeout=10,
            )
            resp.raise_for_status()
            matches = resp.json().get("matches", [])
            fixtures = [
                {
                    "home_team": _FD_NAME_MAP.get(
                        m["homeTeam"]["name"], m["homeTeam"]["name"]
                    ),
                    "away_team": _FD_NAME_MAP.get(
                        m["awayTeam"]["name"], m["awayTeam"]["name"]
                    ),
                    "date": (m.get("utcDate") or "")[:10],
                    "neutral": True,  # WC matches are on neutral ground
                }
                for m in matches
            ]
            cache_path.write_text(json.dumps(fixtures))
            return fixtures
        except Exception as e:
            print(f"[data] football-data.org fetch failed: {e}")

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    print("[data] No FOOTBALL_DATA_API_KEY and no cached fixtures — returning []")
    return []
