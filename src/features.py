"""Layer 1 — identity-agnostic features: rolling form + incremental Elo."""

from __future__ import annotations

from collections import defaultdict, deque
from difflib import get_close_matches

import numpy as np
import pandas as pd

N_RECENT = 10  # rolling form window
ELO_K = 20     # standard Elo K-factor
ELO_DEFAULT = 1500.0

# Map every spelling we might see (football-data.org fixtures, STT output, common
# aliases) onto the spelling the historical dataset — and therefore the model —
# actually uses. WITHOUT this, e.g. football-data's "Côte d'Ivoire" / "IR Iran" /
# "Cape Verde Islands" miss the Elo table and silently get a default 1500 rating,
# which is exactly the "weird percentages" symptom. Keys are lowercased.
ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "america": "United States",
    "united states of america": "United States",
    "ir iran": "Iran",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cape verde islands": "Cape Verde",
    "korea republic": "South Korea",
    "korea": "South Korea",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "czechia": "Czech Republic",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "holland": "Netherlands",
    "curacao": "Curaçao",
}


def resolve_team(name: str | None, elo: dict[str, float]) -> str | None:
    """Snap a free-form team name onto the model's known Elo key.

    Tries exact, alias, case-insensitive, then a tight fuzzy match. Returns the
    input unchanged if nothing resolves (so callers can still fall back to a
    default Elo), or None for an empty name.
    """
    if not name:
        return name
    if name in elo:
        return name
    low = name.strip().lower()
    if low in ALIASES and ALIASES[low] in elo:
        return ALIASES[low]
    lower_map = {k.lower(): k for k in elo}
    if low in lower_map:
        return lower_map[low]
    match = get_close_matches(name, list(elo), n=1, cutoff=0.86)
    return match[0] if match else name

FEATURE_COLS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "home_win_rate",
    "home_avg_gd",
    "away_win_rate",
    "away_avg_gd",
    "neutral",
]


def _form_stats(buf: deque) -> tuple[float, float]:
    if not buf:
        return 0.5, 0.0
    return float(np.mean([m["win"] for m in buf])), float(np.mean([m["gd"] for m in buf]))


def build_training_data(
    df: pd.DataFrame,
    n_recent: int = N_RECENT,
    K: int = ELO_K,
) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    """Iterate matches chronologically; record features BEFORE updating state.

    Returns (X, y, final_elo, final_form) where:
      X          — feature DataFrame (FEATURE_COLS)
      y          — Series of 'W' / 'D' / 'L' labels (home perspective)
      final_elo  — dict[team -> float] after all matches
      final_form — dict[team -> deque] after all matches
    """
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: ELO_DEFAULT)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=n_recent))

    rows: list[dict] = []
    labels: list[str] = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_h, r_a = elo[home], elo[away]

        h_win_rate, h_avg_gd = _form_stats(form[home])
        a_win_rate, a_avg_gd = _form_stats(form[away])

        rows.append(
            {
                "elo_home": r_h,
                "elo_away": r_a,
                "elo_diff": r_h - r_a,
                "home_win_rate": h_win_rate,
                "home_avg_gd": h_avg_gd,
                "away_win_rate": a_win_rate,
                "away_avg_gd": a_avg_gd,
                "neutral": int(bool(row["neutral"])),
            }
        )

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        label = "W" if hs > as_ else ("D" if hs == as_ else "L")
        labels.append(label)

        # Elo update
        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
        s_h = 1.0 if label == "W" else (0.5 if label == "D" else 0.0)
        elo[home] = r_h + K * (s_h - e_h)
        elo[away] = r_a + K * ((1.0 - s_h) - (1.0 - e_h))

        # Form update (from each team's own perspective)
        form[home].append({"win": int(hs > as_), "gd": hs - as_})
        form[away].append({"win": int(as_ > hs), "gd": as_ - hs})

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(labels, name="label")
    return X, y, dict(elo), dict(form)


def features_for_matchup(
    home: str,
    away: str,
    neutral: bool,
    elo: dict[str, float],
    form: dict[str, deque],
) -> pd.DataFrame:
    """Single-row feature DataFrame for inference."""
    home = resolve_team(home, elo)
    away = resolve_team(away, elo)
    r_h = elo.get(home, ELO_DEFAULT)
    r_a = elo.get(away, ELO_DEFAULT)
    h_win_rate, h_avg_gd = _form_stats(form.get(home, deque()))
    a_win_rate, a_avg_gd = _form_stats(form.get(away, deque()))
    return pd.DataFrame(
        [
            {
                "elo_home": r_h,
                "elo_away": r_a,
                "elo_diff": r_h - r_a,
                "home_win_rate": h_win_rate,
                "home_avg_gd": h_avg_gd,
                "away_win_rate": a_win_rate,
                "away_avg_gd": a_avg_gd,
                "neutral": int(neutral),
            }
        ],
        columns=FEATURE_COLS,
    )
