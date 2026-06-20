"""Layer 2 — Claude pundit persona grounded in real predictor numbers.

Uses ANTHROPIC_API_KEY from .env (your credits).
Default model: claude-haiku-4-5 — low latency for voice; pass model="claude-opus-4-8" for richer analysis.
"""

from __future__ import annotations

import os
from collections import deque

import anthropic
from dotenv import load_dotenv

from src.features import ELO_DEFAULT
from src.model import _load_or_train, predict

load_dotenv()

# ─── Persona ──────────────────────────────────────────────────────────────────

_PERSONA_SYSTEM = """\
You are the World Cup Oracle — a supremely confident, entertainingly opinionated,
slightly theatrical football pundit. Think Gary Neville meets Thierry Henry meets a
data scientist who can't resist dropping exact percentages.

RULES (non-negotiable):
1. GROUND EVERY STAT in the "=== GROUNDED MODEL DATA ===" block provided. Never invent
   a percentage, Elo rating, win rate, or goal difference not in that block.
2. You CAN editorialize freely: vivid tactical colour, strong opinions, dramatic analogies.
3. What-if / injury questions: reason FROM the baseline numbers. Acknowledge the model
   can't directly account for the change, then say how it WOULD shift things qualitatively
   (e.g. a striker out → a team with +1.2 avg GD drifts toward neutral).
4. Head-to-head questions: the data has no opponent names in historical records — only
   W/L/D and goal differences. Say so, then reason from Elo history and form.
5. "Last five results": read the last5_results block verbatim and interpret the trend.
6. Voice-ready answers: short declarative sentences, dramatic pauses, NO bullet points.
   Target 80–150 words unless asked for more detail.
7. Speak TO the listener: "you're looking at…", "here's the thing about France…" — radio
   voice, not a written report.
"""

# ─── Fact block ───────────────────────────────────────────────────────────────

def build_fact_block(home: str, away: str, neutral: bool = True) -> dict:
    """Pull live numbers from the model and return a structured fact dict."""
    _, _, elo, form = _load_or_train()
    probs = predict(home, away, neutral)

    h_elo = elo.get(home, ELO_DEFAULT)
    a_elo = elo.get(away, ELO_DEFAULT)

    h_buf = list(form.get(home, deque()))
    a_buf = list(form.get(away, deque()))

    def _stats(buf: list) -> tuple[float, float]:
        if not buf:
            return 0.5, 0.0
        return (
            sum(m["win"] for m in buf) / len(buf),
            sum(m["gd"] for m in buf) / len(buf),
        )

    def _result_str(m: dict) -> str:
        if m["win"]:
            return f"W +{m['gd']}"
        return "D" if m["gd"] == 0 else f"L {m['gd']}"

    h_wr, h_gd = _stats(h_buf)
    a_wr, a_gd = _stats(a_buf)

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "probs": probs,
        "elo": {"home": round(h_elo, 1), "away": round(a_elo, 1)},
        "n_games": {"home": len(h_buf), "away": len(a_buf)},
        "last10_win_rate": {"home": round(h_wr, 3), "away": round(a_wr, 3)},
        "last10_avg_gd": {"home": round(h_gd, 2), "away": round(a_gd, 2)},
        "last5_results": {
            "home": [_result_str(m) for m in h_buf[-5:]],
            "away": [_result_str(m) for m in a_buf[-5:]],
        },
    }


def _facts_to_text(facts: dict) -> str:
    """Render the fact dict as a grounded data section for the system prompt."""
    home, away = facts["home"], facts["away"]
    probs = facts["probs"]
    elo_diff = facts["elo"]["home"] - facts["elo"]["away"]
    stronger = home if elo_diff >= 0 else away

    venue = "neutral venue" if facts["neutral"] else "home advantage"

    lines = [
        "=== GROUNDED MODEL DATA — cite ONLY numbers from this block ===",
        f"Match: {home} vs {away}  ({venue})",
        "",
        "Win / Draw / Loss probabilities (home perspective):",
        f"  {home} win : {probs.get('W', 0):.1%}",
        f"  Draw       : {probs.get('D', 0):.1%}",
        f"  {away} win : {probs.get('L', 0):.1%}",
        "",
        "Elo ratings  (global average ≈ 1500, higher = stronger):",
        f"  {home} : {facts['elo']['home']}",
        f"  {away} : {facts['elo']['away']}",
        f"  Gap    : {abs(elo_diff):.1f} pts in {stronger}'s favour",
        "",
        f"Last {facts['n_games']['home']} games — {home}:",
        f"  Win rate : {facts['last10_win_rate']['home']:.0%}  |  Avg GD : {facts['last10_avg_gd']['home']:+.2f}",
        f"Last {facts['n_games']['away']} games — {away}:",
        f"  Win rate : {facts['last10_win_rate']['away']:.0%}  |  Avg GD : {facts['last10_avg_gd']['away']:+.2f}",
        "",
        "Last-5 results  (most recent last; opponent names not in data):",
        f"  {home} : {' | '.join(facts['last5_results']['home']) or 'insufficient data'}",
        f"  {away} : {' | '.join(facts['last5_results']['away']) or 'insufficient data'}",
        "=================================================================",
    ]
    return "\n".join(lines)


# ─── Main API ─────────────────────────────────────────────────────────────────

def ask(
    home: str,
    away: str,
    question: str,
    neutral: bool = True,
    history: list[dict] | None = None,
    model: str = "claude-haiku-4-5",
) -> str:
    """Ask the Oracle a question about a matchup.

    Args:
        home: Home (or first-named) team.
        away: Away (or second-named) team.
        question: The pundit question in natural language.
        neutral: Whether the match is on neutral ground (True for WC).
        history: Prior [{"role": ..., "content": ...}] turns for multi-turn sessions.
        model: Anthropic model ID. Haiku for voice speed; Opus for richer analysis.

    Returns:
        The Oracle's text response.
    """
    facts = build_fact_block(home, away, neutral)
    system = _PERSONA_SYSTEM + "\n\n" + _facts_to_text(facts)

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ─── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCENARIOS = [
        # (home, away, neutral, question, label)
        (
            "Brazil", "Canada", True,
            "Give me your verdict on this match — who wins and is it even a contest?",
            "BLOWOUT",
        ),
        (
            "France", "Argentina", True,
            "Flip a coin — who wins this one and how confident are you?",
            "COIN-FLIP",
        ),
        (
            "France", "Argentina", True,
            "What if Mbappé picks up an injury in the warm-up and can't play?",
            "INJURY WHAT-IF",
        ),
        (
            "Spain", "Germany", True,
            "Historically, how have Spain and Germany matched up against each other?",
            "HEAD-TO-HEAD",
        ),
        (
            "Brazil", "Argentina", True,
            "What are Brazil's last five results, and what does the form tell you?",
            "LAST-FIVE RESULTS",
        ),
    ]

    print("=" * 70)
    print("WORLD CUP ORACLE — ANALYST TEST SUITE")
    print("=" * 70)

    for home, away, neutral, question, label in SCENARIOS:
        print(f"\n[{label}]  {home} vs {away}")
        print(f"Q: {question}")
        answer = ask(home, away, question, neutral)
        print(f"A: {answer}")
        print("-" * 70)
