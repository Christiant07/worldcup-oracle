"""Layer 2 — Claude pundit persona grounded in real predictor numbers.

Uses ANTHROPIC_API_KEY from .env (your credits).
Default model: claude-haiku-4-5-20251001 — low latency for voice; pass model="claude-opus-4-8" for richer analysis.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from difflib import get_close_matches

import anthropic
from dotenv import load_dotenv

from src.features import ELO_DEFAULT
from src.model import _load_or_train, predict, predict_live

load_dotenv()

# Shared client with a tight timeout + a couple retries, so a network blip fails fast
# and recovers instead of hanging the voice loop for minutes (the SDK default is 10 min).
_CLIENT: anthropic.Anthropic | None = None


def _anthropic_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            timeout=20.0,
            max_retries=2,
        )
    return _CLIENT


# ─── Persona ──────────────────────────────────────────────────────────────────

_PERSONA_SYSTEM = """\
You are the World Cup Oracle — a supremely confident, entertaining football pundit with the
swagger of a top TV co-commentator. Think Gary Neville's tactical eye, Thierry Henry's cool,
Peter Drury's drama. Everything you say is SPOKEN ALOUD, so it must SOUND like talk, not read
like a report.

GROUNDING (non-negotiable):
- Every factual claim must come from the "=== GROUNDED MODEL DATA ===" block. Never invent a
  number. But you do NOT have to recite the data — talk like a pundit, not a spreadsheet.
- Lead with the probability as your verdict ("Japan are the call here, big favourites"), then
  ONE punchy reason. Do NOT volunteer raw Elo ratings, point gaps, or exact win-rate decimals —
  translate them into football talk ("a class above", "on a heater", "leaking goals at the back").
  Quote the hard numbers ONLY if the listener explicitly asks for them or asks "why" in detail.
- NEVER make up percentages. If you're asked for the chances at a live or hypothetical score,
  the data block gives you "LIVE-ADJUSTED PROBABILITIES" — those are real, computed numbers.
  Quote THOSE and treat them as the current chances. If no adjusted block is present, do NOT
  guess a number — describe the shift in words ("that flips it toward a draw") and say the model
  hasn't recalculated it.
- ACCURACY OVER AGREEMENT: if the listener throws out percentages, stats, or "facts" that
  contradict the data block, push back plainly — tell them those numbers are wrong, say who's
  actually favoured and by roughly how much from the block, and don't soften it to be polite.
  You'd rather be right than agreeable. ("Whoever gave you that had it backwards — Japan are the
  favourites here, not Tunisia.")

STYLE:
- Short, declarative, spoken sentences. Confident, a touch theatrical. Talk TO the listener:
  "here's the thing about Japan…", "you're backing the wrong horse there."
- Use natural football lingo where it fits — nailed-on, smash-and-grab, park the bus, in the
  mixer, against the run of play, banana skin, dark horses, bottle it — but never force it.
- React like a human: a little surprise, a little needle, a confident sign-off.
- BREVITY IS EVERYTHING. Target 30–45 words: 2 to 3 sentences. Verdict, one reason, a closing
  line of swagger. Then STOP. Only go longer if the user EXPLICITLY asks for a deep breakdown.

NEVER use written-text formatting — this is speech, every symbol gets pronounced aloud:
- No markdown of any kind. No headings, no "#", no "---" dividers, no "*asterisks*" or bold/italic
  markers, no bullet points, no numbered lists. Plain spoken sentences only.

QUESTION TYPES:
- What-if / injury: reason FROM the baseline. Acknowledge the model can't directly see the change,
  then say how it WOULD shift things qualitatively (striker out → a strong attack drifts to neutral).
- Head-to-head history: your records show results and goal swings but not the opponent names — say
  that naturally ("my book shows their results, not who they faced") and pivot to form and pedigree.
- "Last five results": read the last5_results block and interpret the trend in plain words.
- LIVE MATCH: if a "LIVE MATCH STATUS" line is present, the game is happening RIGHT NOW — lead with
  the live score and let it override the pre-match probabilities.
"""

# ─── Fact block ───────────────────────────────────────────────────────────────

def build_fact_block(
    home: str,
    away: str,
    neutral: bool = True,
    live_data: dict | None = None,
    live_score: tuple[int, int, int] | None = None,
    score_only: tuple[int, int] | None = None,
) -> dict:
    """Pull live numbers from the model and return a structured fact dict.

    If `live_data` (from src.scraper.get_live_data) is provided and the match is live,
    it is attached so the persona can incorporate the in-progress score.

    If `live_score` (home_goals, away_goals, minute) is provided — a real or hypothetical
    in-play scoreline — the model recomputes the win/draw/loss probabilities from that
    point forward (predict_live) and attaches them as `live_probs`, so the persona quotes
    real adjusted numbers instead of inventing them.
    """
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

    facts = {
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
    if live_data and live_data.get("live"):
        facts["live_data"] = live_data
    if live_score is not None:
        hg, ag, minute = live_score
        try:
            facts["live_probs"] = predict_live(home, away, hg, ag, minute, neutral)
        except Exception:
            facts["live_probs"] = None
    elif score_only is not None:
        facts["score_only"] = score_only
    return facts


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
        "Last-5 results (most recent last):",
        f"  {home} : {' | '.join(facts['last5_results']['home']) or 'insufficient data'}",
        f"  {away} : {' | '.join(facts['last5_results']['away']) or 'insufficient data'}",
    ]

    so = facts.get("score_only")
    if so:
        sh, sa = so
        lines += [
            "",
            f"*** CURRENT SCORE (no minute provided): {home} {sh} – {sa} {away} ***",
            "The model CANNOT safely recompute live win probabilities without the minute.",
            "Acknowledge the score in your reply, describe the situation qualitatively,",
            "and ask the listener for the minute so you can give precise updated chances.",
            "Do NOT invent a new percentage — the pre-match numbers above are still the",
            "only ones you can quote until you get the clock.",
        ]

    lp = facts.get("live_probs")
    if lp:
        gh, ga = lp["home_goals"], lp["away_goals"]
        lines += [
            "",
            "*** LIVE-ADJUSTED PROBABILITIES — the score is "
            f"{home} {gh} – {ga} {away} at {lp['minute']}' (these are the CURRENT chances, "
            "they OVERRIDE the pre-match numbers above) ***",
            "Recomputed Win / Draw / Loss (home perspective):",
            f"  {home} win : {lp.get('W', 0):.1%}",
            f"  Draw       : {lp.get('D', 0):.1%}",
            f"  {away} win : {lp.get('L', 0):.1%}",
            "Quote THESE numbers as the live chances — do not invent your own.",
        ]

    live = facts.get("live_data")
    if live and live.get("live"):
        score = live.get("score", "?")
        minute = live.get("minute")
        rc = live.get("red_cards") or {}
        poss = live.get("possession") or {}
        lines += [
            "",
            "*** LIVE — match in progress RIGHT NOW (overrides pre-match probs) ***",
            f"LIVE MATCH STATUS: {home} {score} {away}"
            + (f"  ({minute}')" if minute is not None else ""),
        ]
        if rc:
            lines.append(
                f"Red cards: {home} {rc.get('home', 0)}, {away} {rc.get('away', 0)}"
            )
        if poss:
            lines.append(
                f"Possession: {home} {poss.get('home', '?')}%, {away} {poss.get('away', '?')}%"
            )

    lines.append("=================================================================")
    return "\n".join(lines)


# ─── Live / hypothetical scoreline parsing ──────────────────────────────────────
# So "say it's five-five in the eightieth minute" becomes (5, 5, 80) and the model
# can recompute the real in-play probabilities instead of the persona guessing.

_NUM_WORDS = {
    "nil": 0, "zero": 0, "nought": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINAL_WORDS = {
    "zeroth": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_TENS = {20, 30, 40, 50, 60, 70, 80, 90}


def _normalize_numbers(text: str) -> str:
    """Replace spoken number words with digits, merging tens+ones.

    "five five in the eightieth minute" → "5 5 in the 80 minute"
    "one nil at sixty-seventh minute"   → "1 0 at 67 minute"
    """
    toks = re.findall(r"[a-zA-Z]+|\d+", text.lower())
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit():
            val: int | None = int(t)
        elif t in _NUM_WORDS:
            val = _NUM_WORDS[t]
        elif t in _ORDINAL_WORDS:
            val = _ORDINAL_WORDS[t]
        else:
            val = None

        if val is None:
            out.append(t)
            i += 1
            continue

        # Merge "sixty" + "seven"/"seventh" → 67.
        if val in _TENS and i + 1 < len(toks):
            nxt = toks[i + 1]
            ones = None
            if nxt in _NUM_WORDS and 0 < _NUM_WORDS[nxt] < 10:
                ones = _NUM_WORDS[nxt]
            elif nxt in _ORDINAL_WORDS and 0 < _ORDINAL_WORDS[nxt] < 10:
                ones = _ORDINAL_WORDS[nxt]
            if ones is not None:
                val += ones
                i += 1
        out.append(str(val))
        i += 1
    return " ".join(out)


def _strip_ordinals(text: str) -> str:
    """Remove ordinal suffixes so '67th minute' → '67 minute' before digit normalization."""
    return re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


def parse_score_only(text: str) -> tuple[int, int] | None:
    """Extract just a scoreline (no minute) from free-form speech, or None.

    Used when the user gives a score but no clock — so we can note it in the
    fact block without computing a time-dependent probability from a guessed minute.
    """
    norm = _normalize_numbers(_strip_ordinals(text))
    level = re.search(r"\b(\d{1,2})\s*(?:all|each|apiece)\b", norm)
    if level:
        g = int(level.group(1))
        return g, g
    for mm in re.finditer(r"\b(\d{1,2})\s*(?:-|–|to|nil|:)?\s*(\d{1,2})\b", norm):
        x, y = int(mm.group(1)), int(mm.group(2))
        if x <= 19 and y <= 19:
            return x, y
    return None


def parse_live_score(text: str) -> tuple[int, int, int] | None:
    """Extract (home_goals, away_goals, minute) from free-form speech, else None.

    Requires BOTH a scoreline and a minute so ordinary questions ("bet $1,000",
    "the 2026 World Cup") never trip it. Handles "5-5", "five five", "two nil",
    "5 all", and minutes like "80th minute", "at 67 min", "minute 80".
    """
    norm = _normalize_numbers(_strip_ordinals(text))

    minute: int | None = None
    m = re.search(r"\b(\d{1,3})\s*(?:’|’|min\b|mins\b|minute|minutes)", norm)
    if not m:
        m = re.search(r"\bminute\s+(\d{1,3})\b", norm)
    if m:
        mv = int(m.group(1))
        if 1 <= mv <= 130:
            minute = mv
    if minute is None:
        return None

    hg = ag = None
    # "5 all" / "two each" → level score.
    level = re.search(r"\b(\d{1,2})\s*(?:all|each|apiece)\b", norm)
    if level:
        hg = ag = int(level.group(1))
    else:
        for mm in re.finditer(r"\b(\d{1,2})\s*(?:-|–|to|nil|:)?\s*(\d{1,2})\b", norm):
            x, y = int(mm.group(1)), int(mm.group(2))
            if x <= 19 and y <= 19 and minute not in (x, y):
                hg, ag = x, y
                break
    if hg is None:
        return None
    return hg, ag, minute


# ─── Natural-language matchup detection ─────────────────────────────────────────

def _known_teams() -> list[str]:
    """All team names the model has Elo state for (canonical dataset spellings)."""
    _, _, elo, _ = _load_or_train()
    return list(elo.keys())


def canonical_team(name: str | None) -> str | None:
    """Snap a free-form team name to the model's known spelling.

    Handles case differences and near-misses (e.g. 'USA' → 'United States',
    'Ivory Coast' → 'Côte d'Ivoire') so predict() doesn't silently fall back
    to a default Elo for an unrecognised string.
    """
    if not name:
        return None
    known = _known_teams()
    if name in known:
        return name
    lower = {k.lower(): k for k in known}
    if name.lower() in lower:
        return lower[name.lower()]
    match = get_close_matches(name, known, n=1, cutoff=0.82)
    return match[0] if match else name


_MATCHUP_SYSTEM = """\
You extract the two national football teams for a World Cup match prediction from a
user's spoken question. You will be given the upcoming fixtures list.

The Oracle predicts ANY matchup the user proposes — including hypothetical ones that
are NOT on the real schedule. Do not reason about who actually plays whom tonight.

Output ONLY a single JSON object. No prose, no markdown fences, no explanation.
- Two teams named (e.g. "France or Brazil", "France versus Brazil", "compare Spain and Italy"):
    ALWAYS pair them → {"home": "<first team>", "away": "<second team>"}
- One team named: look it up in the fixtures and use its scheduled opponent.
    If that team is not in the fixtures, return {"home": "<team>", "away": null}.
- No team named (a follow-up like "what if their striker is hurt"): {"home": null, "away": null}
Use canonical country names: "United States", "Côte d'Ivoire", "South Korea", "IR Iran".
"""


def _extract_matchup_regex(
    user_text: str,
    fixtures: list[dict] | None = None,
) -> tuple[str, str] | None:
    """Regex/fuzzy fallback — no API call. Works for 95% of voice queries."""
    fixtures = fixtures or []

    fixture_teams: list[str] = []
    for f in fixtures:
        for key in ("home_team", "away_team"):
            t = f.get(key)
            if t and t not in fixture_teams:
                fixture_teams.append(t)

    all_teams = fixture_teams + [t for t in _known_teams() if t not in fixture_teams]

    # Exact word-boundary scan (preserves mention order for home/away assignment)
    found_ordered: list[tuple[int, str]] = []
    for team in all_teams:
        m = re.search(r"\b" + re.escape(team) + r"\b", user_text, re.IGNORECASE)
        if m:
            c = canonical_team(team)
            if c and not any(c == name for _, name in found_ordered):
                found_ordered.append((m.start(), c))

    found_ordered.sort()
    found = [name for _, name in found_ordered]

    # Fuzzy pass: capitalized 1-3 word phrases not yet matched
    if len(found) < 2:
        for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", user_text):
            c = canonical_team(phrase)
            if c and c not in found:
                found.append(c)
            if len(found) == 2:
                break

    if len(found) >= 2:
        return found[0], found[1]

    if len(found) == 1:
        team = found[0]
        for f in fixtures:
            ht = canonical_team(f.get("home_team"))
            at = canonical_team(f.get("away_team"))
            if ht == team and at:
                return ht, at
            if at == team and ht:
                return ht, at

    return None


def extract_matchup(
    user_text: str,
    fixtures: list[dict] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[str, str] | None:
    """Parse free-form speech into a (home, away) matchup, or None if no teams found.

    Tries Claude first for robustness; falls back to regex if the API is unavailable.
    """
    fixtures = fixtures or []
    fixture_lines = "\n".join(
        f"- {f['home_team']} vs {f['away_team']} ({f.get('date', '')})"
        for f in fixtures[:50]
    ) or "(no upcoming fixtures available)"

    system = f"{_MATCHUP_SYSTEM}\nUpcoming fixtures:\n{fixture_lines}"

    try:
        client = _anthropic_client()
        resp = client.messages.create(
            model=model,
            max_tokens=128,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        raw = resp.content[0].text
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            data = json.loads(match.group(0))
            home = canonical_team(data.get("home"))
            away = canonical_team(data.get("away"))
            if home and away:
                return home, away
    except Exception:
        pass

    return _extract_matchup_regex(user_text, fixtures)


# ─── Rule-based fallback (no LLM) ─────────────────────────────────────────────

def _rule_based_verdict(facts: dict, question: str | None = None) -> str:
    """Pundit-style verdict from the fact block only — no LLM needed.

    Called when the Claude API is unavailable or out of credits.
    All numbers come from the grounded fact block; nothing is invented.
    """
    home = facts["home"]
    away = facts["away"]
    probs = facts["probs"]
    w = probs.get("W", 0.0)
    d = probs.get("D", 0.0)
    l = probs.get("L", 0.0)

    q = (question or "").lower()

    # xG / expected goals question
    if re.search(r"\b(xg|expected.goal|expect.*goal|score.*expect)\b", q):
        from src.model import predict_score
        try:
            xg = predict_score(home, away, facts.get("neutral", True))
            return (
                f"The model puts expected goals at {home} {xg['home_xg']:.1f} "
                f"– {xg['away_xg']:.1f} {away}. "
                f"That lines up with {home} winning {w:.0%} of the time."
            )
        except Exception:
            pass

    # Score given but no minute — acknowledge and ask for the clock
    so = facts.get("score_only")
    if so:
        sh, sa = so
        leader = home if sh > sa else (away if sa > sh else None)
        if leader:
            trailer = away if leader == home else home
            gap = abs(sh - sa)
            lead_str = f"{leader} are {gap}-{'nil' if min(sh,sa)==0 else str(min(sh,sa))} up"
        else:
            lead_str = f"it's level at {sh}-{sa}"
        return (
            f"{lead_str} — that changes things. "
            f"Pre-match I had {home} at {w:.0%}, draw {d:.0%}, {away} at {l:.0%}. "
            f"Tell me the minute and I'll give you the exact updated odds."
        )

    # Live-adjusted probs (score + minute known) — check BEFORE generic keyword branch
    # so "live score is 2-0 at 67'" doesn't fall through to "I can't pull the feed."
    lp = facts.get("live_probs")
    if lp:
        gh, ga = lp["home_goals"], lp["away_goals"]
        minute = lp["minute"]
        lw, ld, ll = lp.get("W", 0), lp.get("D", 0), lp.get("L", 0)
        if lw > ll + 0.1:
            return (
                f"At {gh}-{ga} in the {minute}' minute, {home} are the ones to back "
                f"— they're sitting at {lw:.0%} to see this out. Game's theirs to lose."
            )
        elif ll > lw + 0.1:
            return (
                f"At {gh}-{ga} in the {minute}' minute, {away} are in the driving seat "
                f"at {ll:.0%}. {home} need something special now."
            )
        else:
            return (
                f"Level at {gh}-{ga} in the {minute}' minute — it's delicately poised. "
                f"{home} win {lw:.0%}, draw {ld:.0%}, {away} {ll:.0%}. Anyone's game."
            )

    # Live score / update question (no live_probs available)
    if re.search(r"\b(live|score|right now|current|update|check|happening|going on|minute|half)\b", q):
        live = facts.get("live_data")
        if live and live.get("live"):
            score = live.get("score", "?")
            minute = live.get("minute")
            min_str = f" at {minute}'" if minute else ""
            return (
                f"Live: {home} {score} {away}{min_str}. "
                f"Pre-match the model had {home} at {w:.0%} — "
                f"ask me with the score and minute to get updated odds."
            )
        return (
            f"I can't pull the live feed right now. Pre-match: {home} {w:.0%}, "
            f"draw {d:.0%}, {away} {l:.0%}. Tell me the score and minute "
            f"and I'll give you updated chances."
        )

    # Form / last five question
    if re.search(r"\b(form|last.five|recent|result)\b", q):
        h5 = facts["last5_results"]["home"]
        a5 = facts["last5_results"]["away"]
        h_str = " | ".join(h5) if h5 else "no data"
        a_str = " | ".join(a5) if a5 else "no data"
        h_wr = facts["last10_win_rate"]["home"]
        a_wr = facts["last10_win_rate"]["away"]
        better = home if h_wr >= a_wr else away
        return (
            f"{home} last five: {h_str}. {away}: {a_str}. "
            f"{better} have the better recent form heading into this one."
        )

    h_wr = facts["last10_win_rate"]["home"]
    a_wr = facts["last10_win_rate"]["away"]

    if w > l + 0.12:
        fav, underdog, fav_prob = home, away, w
    elif l > w + 0.12:
        fav, underdog, fav_prob = away, home, l
    else:
        fav = None

    if fav:
        form_note = ""
        if fav == home and h_wr >= a_wr + 0.2:
            form_note = " and in red-hot form"
        elif fav == away and a_wr >= h_wr + 0.2:
            form_note = " and in red-hot form"
        edge = "big" if fav_prob > 0.60 else "narrow"
        return (
            f"{fav} are the {edge} favourites here at {fav_prob:.0%}{form_note}. "
            f"Draw's possible at {d:.0%}, but I'm not backing {underdog} to nick this one."
        )

    # Near coin-flip.
    if h_wr > a_wr + 0.15:
        lean = f"{home} on form"
    elif a_wr > h_wr + 0.15:
        lean = f"{away} on form"
    else:
        lean = "neither side"
    return (
        f"Genuine coin-flip — {home} {w:.0%}, draw {d:.0%}, {away} {l:.0%}. "
        f"Form leans {lean}, but on the day this could go anywhere."
    )


# ─── Main API ─────────────────────────────────────────────────────────────────

def ask(
    home: str,
    away: str,
    question: str,
    neutral: bool = True,
    history: list[dict] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    live_data: dict | None = None,
    live_score: tuple[int, int, int] | None = None,
) -> str:
    """Ask the Oracle a question about a matchup.

    Args:
        home: Home (or first-named) team.
        away: Away (or second-named) team.
        question: The pundit question in natural language.
        neutral: Whether the match is on neutral ground (True for WC).
        history: Prior [{"role": ..., "content": ...}] turns for multi-turn sessions.
        model: Anthropic model ID. Haiku for voice speed; Opus for richer analysis.
        live_data: Optional live-match dict from src.scraper.get_live_data.

    Returns:
        The Oracle's text response.
    """
    facts = build_fact_block(home, away, neutral, live_data, live_score)
    system = _PERSONA_SYSTEM + "\n\n" + _facts_to_text(facts)

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    try:
        client = _anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=220,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        pass
    except anthropic.BadRequestError as exc:
        if "credit balance is too low" not in str(exc):
            raise
    except Exception:
        pass

    return _rule_based_verdict(build_fact_block(home, away, neutral, live_data, live_score), question)


def build_system_prompt(
    home: str,
    away: str,
    neutral: bool = True,
    live_data: dict | None = None,
    live_score: tuple[int, int, int] | None = None,
    score_only: tuple[int, int] | None = None,
) -> str:
    """Return the full system prompt for a matchup (persona + grounded facts)."""
    facts = build_fact_block(home, away, neutral, live_data, live_score, score_only)
    reminder = (
        "\n\nFINAL REMINDER: You are SPOKEN ALOUD. 30–45 words, 2–3 short sentences. "
        "Lead with the verdict, one reason, a line of swagger, then STOP. No markdown or "
        "symbols of any kind (#, *, ---). Don't quote Elo or exact decimals unless asked."
    )
    return _PERSONA_SYSTEM + "\n\n" + _facts_to_text(facts) + reminder


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
