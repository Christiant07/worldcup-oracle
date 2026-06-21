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

from src.features import ELO_DEFAULT, ALIASES as _ALIAS_TO_CANON, resolve_team
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
    "nil": 0, "zero": 0, "nought": 0, "oh": 0, "nothing": 0, "none": 0,
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


# Tokens that may sit *between* the two score digits ("one to nil" → 1 0).
_SCORE_CONNECTORS = {"to", "nil", "v", "vs", "versus", "against"}


def _extract_minute(norm: str) -> int | None:
    """Find a match minute in normalized text ("80 minute", "minute 80", "80'")."""
    m = re.search(r"\b(\d{1,3})\s*(?:’|'|min\b|mins\b|minute|minutes)", norm)
    if not m:
        m = re.search(r"\bminute\s+(\d{1,3})\b", norm)
    if m:
        mv = int(m.group(1))
        if 1 <= mv <= 130:
            return mv
    return None


def _extract_scoreline(norm: str, exclude: int | None = None) -> tuple[int, int] | None:
    """Find a scoreline from normalized, space-separated tokens.

    Operates on TOKENS, not raw digits, so a single two-digit number like "80"
    (a minute) is never split into the score "8 – 0". A score is two numeric
    tokens (each 0–19) that are adjacent, optionally joined by one connector
    token ("one to nil" → 1 0). Also handles "5 all" / "two each" level scores.
    """
    # Level score: "5 all", "two each", "three apiece".
    level = re.search(r"\b(\d{1,2})\s*(?:all|each|apiece)\b", norm)
    if level:
        g = int(level.group(1))
        if g <= 19:
            return g, g

    toks = norm.split()
    nums = [(i, int(t)) for i, t in enumerate(toks) if t.isdigit()]
    for k in range(len(nums) - 1):
        (i, x), (j, y) = nums[k], nums[k + 1]
        if x > 19 or y > 19:
            continue
        gap = j - i
        # Adjacent tokens, or one connector word between them.
        if gap == 1 or (gap == 2 and toks[i + 1].lower() in _SCORE_CONNECTORS):
            if exclude is not None and exclude in (x, y):
                continue
            return x, y
    return None


def parse_score_only(text: str) -> tuple[int, int] | None:
    """Extract just a scoreline (no minute) from free-form speech, or None.

    Used when the user gives a score but no clock — so we can note it in the
    fact block without computing a time-dependent probability from a guessed minute.
    """
    return _extract_scoreline(_normalize_numbers(_strip_ordinals(text)))


def parse_minute_only(text: str) -> int | None:
    """Extract a standalone match minute ("80", "in the 80th", "minute 80"), else None.

    Used for the multi-turn case where the score was given in an earlier turn
    ("Spain are losing one-nil") and the clock arrives separately ("eighty").
    Returns None unless the utterance is essentially just a minute, so it never
    swallows the second number of a scoreline.
    """
    norm = _normalize_numbers(_strip_ordinals(text))
    m = _extract_minute(norm)
    if m is not None:
        return m
    # Bare lone number with no scoreline present → treat as the minute.
    nums = [int(t) for t in norm.split() if t.isdigit()]
    if len(nums) == 1 and 1 <= nums[0] <= 130 and _extract_scoreline(norm) is None:
        return nums[0]
    return None


def parse_live_score(text: str) -> tuple[int, int, int] | None:
    """Extract (home_goals, away_goals, minute) from free-form speech, else None.

    Requires BOTH a scoreline and a minute so ordinary questions ("bet $1,000",
    "the 2026 World Cup") never trip it. Handles "5-5", "five five", "two nil",
    "5 all", and minutes like "80th minute", "at 67 min", "minute 80".
    """
    norm = _normalize_numbers(_strip_ordinals(text))
    minute = _extract_minute(norm)
    if minute is None:
        return None
    score = _extract_scoreline(norm, exclude=minute)
    if score is None:
        return None
    return score[0], score[1], minute


# ─── Natural-language matchup detection ─────────────────────────────────────────

def _known_teams() -> list[str]:
    """All team names the model has Elo state for (canonical dataset spellings)."""
    _, _, elo, _ = _load_or_train()
    return list(elo.keys())


# Alias → model-key map is shared with the feature layer (src.features.ALIASES) so
# the voice path and the predictor resolve names identically.

# Common English words that fuzzy-match a country and create phantom "teams"
# (e.g. STT mis-hearing turns "that Spain" → a fake opponent). Never a team.
_STOPWORD_NAMES = {
    "oh", "that", "this", "the", "them", "they", "their", "there", "than",
    "chad",  # an actual country but never in WC fixtures; avoid the "I had"→Chad misfire
}


def canonical_team(name: str | None, strict: bool = False) -> str | None:
    """Snap a free-form team name to the model's known spelling.

    Handles case differences, common aliases ('USA' → 'United States',
    'Ivory Coast' → 'Côte d'Ivoire') and near-misses so predict() doesn't
    silently fall back to a default Elo for an unrecognised string.

    With strict=True, returns None when the name does not confidently resolve to
    a known team (instead of echoing the input back). Use strict for matchup
    detection so a mishearing like "That Spain" or "Oh" never becomes an opponent.
    """
    if not name:
        return None
    cleaned = name.strip()
    if cleaned.lower() in _STOPWORD_NAMES:
        return None
    known = _known_teams()
    if cleaned in known:
        return cleaned
    lower = {k.lower(): k for k in known}
    if cleaned.lower() in lower:
        return lower[cleaned.lower()]
    if cleaned.lower() in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[cleaned.lower()]
    match = get_close_matches(cleaned, known, n=1, cutoff=0.86)
    if match:
        return match[0]
    return None if strict else cleaned


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

    # Fuzzy pass: capitalized 1-3 word phrases not yet matched. STRICT — a phrase
    # only counts if it confidently resolves to a known team, so a misheard
    # "That Spain" / "Oh" never becomes a phantom opponent.
    if len(found) < 2:
        for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", user_text):
            c = canonical_team(phrase, strict=True)
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


def _explicit_known_teams(
    user_text: str,
    fixtures: list[dict] | None = None,
) -> list[str]:
    """Confidently-known teams literally named in the text, in mention order.

    Exact word-boundary scan over known team names + aliases (no fuzzy guessing),
    so this never invents a team. Used to decide when an utterance names a *new*
    matchup vs. when it's a follow-up about the current one.
    """
    fixtures = fixtures or []
    names = set(_known_teams())
    for f in fixtures:
        for k in ("home_team", "away_team"):
            if f.get(k):
                names.add(f[k])

    found: list[tuple[int, str]] = []

    def _add(pos: int, canon: str | None) -> None:
        if canon and not any(canon == n for _, n in found):
            found.append((pos, canon))

    for team in names:
        m = re.search(r"\b" + re.escape(team) + r"\b", user_text, re.IGNORECASE)
        if m:
            _add(m.start(), canonical_team(team, strict=True))
    for alias, canon in _ALIAS_TO_CANON.items():
        m = re.search(r"\b" + re.escape(alias) + r"\b", user_text, re.IGNORECASE)
        if m:
            _add(m.start(), canon)

    found.sort()
    return [n for _, n in found]


def _fixture_opponent(team: str, fixtures: list[dict] | None) -> str | None:
    """Scheduled opponent of `team` from the fixtures list, else None."""
    for f in fixtures or []:
        ht = canonical_team(f.get("home_team"), strict=True)
        at = canonical_team(f.get("away_team"), strict=True)
        if ht == team and at:
            return at
        if at == team and ht:
            return ht
    return None


def extract_matchup(
    user_text: str,
    fixtures: list[dict] | None = None,
    model: str = "claude-haiku-4-5-20251001",
    current: tuple[str | None, str | None] | None = None,
) -> tuple[str, str] | None:
    """Parse free-form speech into a (home, away) matchup, or None if no teams found.

    `current` is the matchup already in play. When it is set, switching is
    CONSERVATIVE: we only change the matchup if the utterance explicitly names
    two known teams. A single-team mention or a misheard word is treated as a
    follow-up (returns None → caller keeps the current matchup) instead of
    silently spawning a phantom opponent like "That Spain" or "Oh".
    """
    fixtures = fixtures or []
    have_current = bool(current and current[0] and current[1])

    # Two confidently-known teams named outright → unambiguous pairing. This is
    # the only way to switch matchups mid-conversation.
    explicit = _explicit_known_teams(user_text, fixtures)
    if len(explicit) >= 2:
        return explicit[0], explicit[1]

    # Mid-conversation with fewer than two named teams → it's a follow-up. Keep
    # the current matchup rather than re-resolving (which mis-switched before).
    if have_current:
        return None

    # Opening utterance: lean on Claude (best with aliases / typos), then regex.
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
            home = canonical_team(data.get("home"), strict=True)
            away = canonical_team(data.get("away"), strict=True)
            if home and away:
                return home, away
            if home and not away:
                opp = _fixture_opponent(home, fixtures)
                if opp:
                    return home, opp
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


# ─── Web-search consensus (Claude searches the web like Google) ─────────────────

# Web search needs a longer timeout than the voice client — the server-side tool
# loop can take several seconds. Kept separate so it never slows the voice path.
_WEB_CLIENT: anthropic.Anthropic | None = None


def _web_client() -> anthropic.Anthropic:
    global _WEB_CLIENT
    if _WEB_CLIENT is None:
        _WEB_CLIENT = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], timeout=60.0, max_retries=1
        )
    return _WEB_CLIENT


_WEB_PROMPT = """\
Search the web for the latest win/draw/loss probabilities or betting odds for the \
upcoming match {home} vs {away} (treat it as a neutral-venue World Cup fixture). Look at \
how sites like Google, bookmakers, and prediction models price it.

Then convert what you find into three probabilities that sum to 1 and reply with ONLY a \
single JSON object, no prose, no markdown fences:
{{"home_prob": <0-1>, "draw_prob": <0-1>, "away_prob": <0-1>, "summary": "<one short \
sentence on the web consensus>"}}

home_prob is {home} winning, away_prob is {away} winning. If the web has no real data for \
this exact matchup, return {{"home_prob": null, "draw_prob": null, "away_prob": null, \
"summary": "no reliable web data for this matchup"}}."""


def web_consensus(home: str, away: str, model: str = "claude-opus-4-8") -> dict | None:
    """Use Claude's web-search tool to fetch a web/bookmaker consensus for the matchup.

    Returns {"source": "Web (Claude search)", "home_prob", "draw_prob", "away_prob",
    "summary", "citations": [...]} — probs may be None when the web has no data.
    Returns None on any error (API unavailable, no credits, etc.). Never raises.
    Slower than the model (server-side search loop), so callers should gate it behind
    an explicit request rather than the live voice turn.
    """
    try:
        client = _web_client()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}],
            messages=[{"role": "user", "content": _WEB_PROMPT.format(home=home, away=away)}],
        )
    except Exception:
        return None

    text_parts: list[str] = []
    citations: list[dict] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "web_search_tool_result":
            results = getattr(block, "content", None) or []
            if isinstance(results, list):
                for r in results[:5]:
                    title = getattr(r, "title", None)
                    url = getattr(r, "url", None)
                    if title or url:
                        citations.append({"title": title, "url": url})

    text = "\n".join(text_parts)
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None

    def _p(v) -> float | None:
        try:
            f = float(v)
            return f if 0.0 <= f <= 1.0 else None
        except (TypeError, ValueError):
            return None

    return {
        "source": "Web (Claude search)",
        "home_prob": _p(data.get("home_prob")),
        "draw_prob": _p(data.get("draw_prob")),
        "away_prob": _p(data.get("away_prob")),
        "summary": data.get("summary"),
        "citations": citations,
    }


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
