"""Layer 1 — StandardScaler + LogisticRegression -> win/draw/loss probabilities."""

from __future__ import annotations

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
