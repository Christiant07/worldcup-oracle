"""Layer 1 — StandardScaler + LogisticRegression -> win/draw/loss probabilities."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_results, upcoming_fixtures
from src.features import build_training_data, features_for_matchup

MODEL_PATH = Path(__file__).parent.parent / "data" / "model.pkl"


def train(save: bool = True) -> tuple:
    """Train on all historical data; print held-out accuracy.

    Returns (scaler, clf, elo, form).
    """
    df = load_results()
    print(f"[model] Training on {len(df):,} matches…")
    X, y, elo, form = build_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X_tr, y_train)

    acc = accuracy_score(y_test, clf.predict(X_te))
    print(f"[model] Held-out accuracy: {acc:.3f}  ({len(y_test):,} test matches)")
    print(f"[model] Classes: {clf.classes_.tolist()}")

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((scaler, clf, elo, form), f)
        print(f"[model] Saved to {MODEL_PATH}")

    return scaler, clf, elo, form


_cache: tuple | None = None


def _load_or_train() -> tuple:
    global _cache
    if _cache is not None:
        return _cache
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _cache = pickle.load(f)
    else:
        _cache = train()
    return _cache


def predict(home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
    """Return win/draw/loss probabilities for home_team vs away_team.

    Keys: 'W' (home win), 'D' (draw), 'L' (away win).
    """
    scaler, clf, elo, form = _load_or_train()
    X = features_for_matchup(home_team, away_team, neutral, elo, form)
    X_s = scaler.transform(X)
    probs = clf.predict_proba(X_s)[0]
    return dict(zip(clf.classes_.tolist(), probs.tolist()))


def predict_score(home_team: str, away_team: str, neutral: bool = True) -> dict:
    """Derive an expected-goals scoreline from the W/D/L probabilities.

    We don't train on shot data, so this is a probability-weighted approximation:
    each outcome is mapped to an average international scoring rate and weighted by
    its probability. Directionally honest, good enough to display as "expected goals".

    Returns: {"home_xg": float, "away_xg": float, "home_team": str, "away_team": str}.
    """
    probs = predict(home_team, away_team, neutral)
    w, d, l = probs.get("W", 0.0), probs.get("D", 0.0), probs.get("L", 0.0)

    home_xg = round(w * 2.1 + d * 1.1, 1)
    away_xg = round(l * 1.8 + d * 1.0, 1)

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_team": home_team,
        "away_team": away_team,
    }


def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    return math.exp(-lam) * lam**k / math.factorial(k)


def predict_live(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    minute: int,
    neutral: bool = True,
    max_extra: int = 8,
) -> dict:
    """Recompute win/draw/loss probabilities GIVEN a live (or hypothetical) scoreline.

    This is what lets the bars move as a game unfolds. The pre-match model only sees
    form + Elo; once goals are on the board and time has elapsed, the real chances are
    dominated by "who's ahead and how long is left". We model that honestly:

      • Convert the pre-match expectation into per-team scoring rates (full-match xG).
      • Scale them by the fraction of the match still to play.
      • Treat remaining goals for each side as independent Poisson draws.
      • Sum the joint distribution to get P(home win) / P(draw) / P(away win) from the
        CURRENT score forward.

    So 5–5 in the 80th minute correctly collapses toward a draw (little time left),
    instead of the persona guessing numbers out of thin air.

    Returns the same W/D/L keys as predict(), plus the live context that produced them.
    """
    base = predict_score(home_team, away_team, neutral)
    lam_home = max(base.get("home_xg") or 0.0, 0.35)
    lam_away = max(base.get("away_xg") or 0.0, 0.35)

    minute = max(0, min(int(minute), 120))
    # Fraction of a 90-minute match still to play. Keep a small floor so even at/after
    # 90' there's a sliver of stoppage-time chance for the trailing side.
    if minute >= 90:
        frac = 0.04
    else:
        frac = max((90 - minute) / 90.0, 0.04)

    rem_home = lam_home * frac
    rem_away = lam_away * frac

    pw = pd = pl = 0.0
    for xh in range(max_extra + 1):
        ph = _poisson_pmf(xh, rem_home)
        for xa in range(max_extra + 1):
            p = ph * _poisson_pmf(xa, rem_away)
            final_h = home_goals + xh
            final_a = away_goals + xa
            if final_h > final_a:
                pw += p
            elif final_h == final_a:
                pd += p
            else:
                pl += p

    total = pw + pd + pl or 1.0
    return {
        "W": pw / total,
        "D": pd / total,
        "L": pl / total,
        "minute": minute,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "home_team": home_team,
        "away_team": away_team,
    }


def forecast_world_cup(competition: str = "WC") -> list[dict]:
    """Predict all upcoming World Cup fixtures and return enriched fixture list."""
    fixtures = upcoming_fixtures(competition)
    results = []
    for fix in fixtures:
        probs = predict(fix["home_team"], fix["away_team"], neutral=fix.get("neutral", True))
        results.append({**fix, "probs": probs})
    return results


if __name__ == "__main__":
    train()

    print("\n--- Sample predictions ---")
    samples = [
        ("Brazil", "Argentina", True),
        ("France", "England", True),
        ("Spain", "Germany", True),
        ("United States", "Mexico", True),
    ]
    for home, away, neutral in samples:
        p = predict(home, away, neutral)
        print(
            f"{home:20s} vs {away:20s} | "
            f"W:{p.get('W', 0):.1%}  D:{p.get('D', 0):.1%}  L:{p.get('L', 0):.1%}"
        )

    print("\n--- WC 2026 upcoming fixtures ---")
    forecasts = forecast_world_cup()
    if not forecasts:
        print("  (no fixtures fetched — set FOOTBALL_DATA_API_KEY or fixtures are cached)")
    for f in forecasts[:15]:
        p = f["probs"]
        print(
            f"{f['date']}  {f['home_team']:20s} vs {f['away_team']:20s} | "
            f"W:{p.get('W', 0):.1%}  D:{p.get('D', 0):.1%}  L:{p.get('L', 0):.1%}"
        )
