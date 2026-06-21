"""Edge-case test suite for the World Cup Oracle prediction + parsing layers.

Standalone (no pytest, no network, no Anthropic calls) so it runs anywhere:

    python -m tests.test_oracle      # or: python tests/test_oracle.py

Covers the bugs found in the live demo (phantom teams, "80" → 8-0, "losing"
orientation, default-Elo fixture names) plus probability sanity + monotonicity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyst import (  # noqa: E402
    canonical_team,
    extract_matchup,
    parse_live_score,
    parse_minute_only,
    parse_score_only,
)
from src.features import resolve_team  # noqa: E402
from src.model import _load_or_train, predict, predict_live, predict_score  # noqa: E402
from src.server import _orient_score  # noqa: E402

_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    global _PASS, _FAIL
    if got == want:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"{label}\n      got:  {got!r}\n      want: {want!r}")


def check_true(label: str, cond: bool) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"{label}  (expected True)")


# ── 1. Score / minute parsing ──────────────────────────────────────────────────

def test_score_parsing() -> None:
    # The "80" → 8-0 demo bug: a bare two-digit minute is NOT a scoreline.
    check("'80' is not a score", parse_score_only("80"), None)
    check("'eightieth minute' is not a score", parse_score_only("eightieth minute"), None)
    check("'80' is a minute", parse_minute_only("80"), 80)
    check("'eightieth minute' is a minute", parse_minute_only("eightieth minute"), 80)

    # Real scorelines, digits and words.
    check("'2-0'", parse_score_only("it's 2-0"), (2, 0))
    check("'two nil'", parse_score_only("two nil"), (2, 0))
    check("'one to nothing'", parse_score_only("one to nothing"), (1, 0))
    check("'five five'", parse_score_only("five five"), (5, 5))
    check("'5 all'", parse_score_only("5 all"), (5, 5))
    check("'two each'", parse_score_only("two each"), (2, 2))
    check("'three apiece'", parse_score_only("three apiece"), (3, 3))

    # Full live lines (score + minute together).
    check("'1-0 at 80'", parse_live_score("Spain losing 1-0 at the 80th minute"), (1, 0, 80))
    check("'five five eightieth'", parse_live_score("five five in the eightieth minute"), (5, 5, 80))
    check("'two nil 67th'", parse_live_score("two nil at the sixty seventh minute"), (2, 0, 67))
    check("'minute 90'", parse_live_score("it's 2 1 at minute 90"), (2, 1, 90))

    # Non-scores must stay None (no false positives).
    check("'bet $1,000'", parse_live_score("I'd bet a thousand on this"), None)
    check("'2026 World Cup'", parse_live_score("the 2026 World Cup final"), None)
    check("plain question", parse_live_score("who do you think wins this one"), None)
    check("score w/o minute -> live None", parse_live_score("it's two nil"), None)


# ── 2. Matchup extraction: phantom rejection + conservative switching ───────────

def test_matchup_extraction() -> None:
    cur = ("Spain", "Saudi Arabia")
    # The demo phantoms: mid-conversation, a misheard word must NOT switch matchup.
    check("phantom 'That Spain'", extract_matchup("That Spain is losing one nil", [], current=cur), None)
    check("phantom 'Oh'", extract_matchup("Oh, I said Spain is losing", [], current=cur), None)
    check("bare minute keeps matchup", extract_matchup("80", [], current=cur), None)
    check("single team keeps matchup", extract_matchup("Spain is losing one nil", [], current=cur), None)

    # Two explicitly-named known teams always wins (opener or mid-convo switch).
    check("two teams override", extract_matchup("what about France versus Brazil", [], current=cur),
          ("France", "Brazil"))
    check("two teams from scratch", extract_matchup("France or Brazil", [], current=None),
          ("France", "Brazil"))


def test_canonical_team() -> None:
    check_true("strict rejects 'Oh'", canonical_team("Oh", strict=True) is None)
    check_true("strict rejects 'That Spain'", canonical_team("That Spain", strict=True) is None)
    check_true("strict rejects gibberish", canonical_team("Asdf Qwer", strict=True) is None)
    check("USA alias", canonical_team("USA", strict=True), "United States")
    check("IR Iran -> Iran", canonical_team("IR Iran", strict=True), "Iran")
    check("Côte d'Ivoire", canonical_team("Côte d'Ivoire", strict=True), "Ivory Coast")
    check("case-insensitive", canonical_team("spain", strict=True), "Spain")


# ── 3. The default-Elo resolution bug (5 fixture teams) ─────────────────────────

def test_team_resolution() -> None:
    _, _, elo, _ = _load_or_train()
    # These football-data spellings must map to a REAL Elo, not the 1500 default.
    for fixture_name, model_key in [
        ("Côte d'Ivoire", "Ivory Coast"),
        ("IR Iran", "Iran"),
        ("Cape Verde Islands", "Cape Verde"),
        ("Bosnia-Herzegovina", "Bosnia and Herzegovina"),
        ("Czechia", "Czech Republic"),
    ]:
        resolved = resolve_team(fixture_name, elo)
        check(f"resolve {fixture_name!r}", resolved, model_key)
        check_true(f"{fixture_name!r} has non-default Elo", abs(elo.get(resolved, 1500.0) - 1500.0) > 1.0)


# ── 4. Pre-match probability sanity ─────────────────────────────────────────────

def test_predict_sanity() -> None:
    for h, a in [("Brazil", "Canada"), ("Spain", "Saudi Arabia"), ("France", "Argentina")]:
        p = predict(h, a, True)
        check_true(f"{h} v {a} sums to 1", abs(sum(p.values()) - 1.0) < 1e-6)
        check_true(f"{h} v {a} keys", set(p) == {"W", "D", "L"})
        check_true(f"{h} v {a} in [0,1]", all(0.0 <= v <= 1.0 for v in p.values()))

    # Blowout: strong favourite clearly ahead (draws keep the gap < 0.4 even here).
    blow = predict("Brazil", "Canada", True)
    check_true("Brazil clear favourite", blow["W"] > 0.45 and blow["W"] > blow["L"] + 0.3)
    # Resolved fixture team should produce a decisive (non-coinflip) call.
    cv = predict("Spain", "Cape Verde Islands", True)
    check_true("Spain >> Cape Verde", cv["W"] > 0.6)


# ── 5. Live (in-play) probability behaviour ─────────────────────────────────────

def test_predict_live() -> None:
    def wdl(d):
        return {k: round(d[k], 3) for k in "WDL"}

    # 5-5 at the 80th collapses toward a draw (little time left).
    p = predict_live("Brazil", "Argentina", 5, 5, 80, True)
    check_true("5-5@80 -> draw heavy", p["D"] > 0.7)

    # 2-0 up at 70' is nearly sealed.
    p = predict_live("Brazil", "Argentina", 2, 0, 70, True)
    check_true("2-0@70 -> home wins", p["W"] > 0.85)

    # 0-1 down at 80' is nearly lost (this is the Spain/Saudi demo case).
    p = predict_live("Spain", "Saudi Arabia", 0, 1, 80, True)
    check_true("Spain 0-1@80 -> away wins", p["L"] > 0.75)

    # Monotonicity: a one-goal lead is safer at 85' than at 20'.
    early = predict_live("Spain", "Saudi Arabia", 1, 0, 20, True)["W"]
    late = predict_live("Spain", "Saudi Arabia", 1, 0, 85, True)["W"]
    check_true("lead safer later (85 > 20)", late > early)

    # Every live distribution is a valid probability simplex.
    for hg, ag, m in [(0, 0, 1), (3, 3, 119), (1, 0, 45), (0, 2, 90)]:
        p = predict_live("France", "Spain", hg, ag, m, True)
        check_true(f"live {hg}-{ag}@{m} sums to 1", abs(p["W"] + p["D"] + p["L"] - 1.0) < 1e-6)


# ── 6. Score orientation (who's actually ahead) ─────────────────────────────────

def test_orientation() -> None:
    # "Spain are losing one-nil" → Spain has the 0, opponent the 1.
    check("Spain losing 1-0", _orient_score((1, 0, 80), "Spain are losing one nil", "Spain", "Saudi Arabia"),
          (0, 1, 80))
    # "Spain two-nil up" → Spain ahead.
    check("Spain two-nil up", _orient_score((2, 0, 60), "Spain are two nil up", "Spain", "Saudi Arabia"),
          (2, 0, 60))
    # Away team named first before the score ("Iran 1-0") → away leads.
    check("away named first", _orient_score((1, 0, 20), "Iran 1-0", "Spain", "Iran"), (0, 1, 20))
    # Level scores never flip.
    check("level never flips", _orient_score((2, 2, 70), "it's two each", "Spain", "Iran"), (2, 2, 70))


# ── 7. Multi-turn: score in one turn, minute in the next ─────────────────────────

def test_multiturn_merge() -> None:
    # Turn 1: "Spain are losing one-nil" → score-only, oriented.
    score = parse_score_only("Spain are losing one nil")
    check("turn1 score parsed", score, (1, 0))
    oriented = _orient_score((score[0], score[1], 90), "Spain are losing one nil", "Spain", "Saudi Arabia")
    check("turn1 oriented", (oriented[0], oriented[1]), (0, 1))
    # Turn 2: "80" → minute only, merges with stored score.
    minute = parse_minute_only("80")
    check("turn2 minute parsed", minute, 80)
    live = predict_live("Spain", "Saudi Arabia", oriented[0], oriented[1], minute, True)
    check_true("merged live -> Saudi favoured", live["L"] > 0.75)


def main() -> int:
    tests = [
        test_score_parsing,
        test_matchup_extraction,
        test_canonical_team,
        test_team_resolution,
        test_predict_sanity,
        test_predict_live,
        test_orientation,
        test_multiturn_merge,
    ]
    print("=" * 68)
    print("WORLD CUP ORACLE — EDGE-CASE TEST SUITE")
    print("=" * 68)
    for t in tests:
        t()
        print(f"  ✓ ran {t.__name__}")

    print("-" * 68)
    print(f"PASSED: {_PASS}   FAILED: {_FAIL}")
    if _FAILURES:
        print("\nFAILURES:")
        for f in _FAILURES:
            print(f"  ✗ {f}")
        return 1
    print("ALL GREEN ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
