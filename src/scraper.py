"""Browserbase-powered scrapers — live match state + prediction-market odds.

These are STRICTLY best-effort. Every public function wraps its whole body in a
try/except and returns a safe fallback ({"live": False} / None) so a flaky site,
a missing key, or a Browserbase hiccup can NEVER crash the voice loop.

Flow: create a Browserbase session over REST (httpx), then drive the *remote*
browser with Playwright over CDP — so no local Chromium / `playwright install`
is needed, only the playwright client library.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from contextlib import contextmanager
from typing import Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()

_BB_SESSIONS_URL = "https://api.browserbase.com/v1/sessions"
_NAV_TIMEOUT_MS = 30_000   # hard cap so the demo never hangs
_SEL_TIMEOUT_MS = 5_000

# Market-odds source order. The public Polymarket Gamma API is fast (~1s) and exact,
# so it leads; Browserbase is the fallback that scrapes when the API has no market for
# a pairing. (Browserbase remains PRIMARY for live match data — see get_live_data.)
# Swap to ["browserbase", "api"] to force the remote-browser scrape first.
ODDS_SOURCE_ORDER = ["api", "browserbase"]

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Small alias map so our canonical team names match Polymarket's spellings.
_TEAM_ALIASES = {
    "côte d'ivoire": "ivory coast",
    "ir iran": "iran",
    "south korea": "korea",
    "united states": "usa",
    "north macedonia": "macedonia",
}


def _log(msg: str) -> None:
    print(f"[scraper] {msg}", file=sys.stderr)


def _browserbase_enabled() -> bool:
    """True only if both Browserbase env vars are present and non-empty."""
    return bool(os.getenv("BROWSERBASE_API_KEY") and os.getenv("BROWSERBASE_PROJECT_ID"))


def _create_session() -> str:
    """Create a Browserbase session and return its CDP connect URL.

    Prefers the `connectUrl` the API returns; otherwise builds the standard one.
    Never logs the API key.
    """
    api_key = os.environ["BROWSERBASE_API_KEY"]
    project_id = os.environ["BROWSERBASE_PROJECT_ID"]

    resp = httpx.post(
        _BB_SESSIONS_URL,
        headers={"X-BB-API-Key": api_key, "Content-Type": "application/json"},
        json={"projectId": project_id},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    connect_url = data.get("connectUrl")
    if connect_url:
        return connect_url

    session_id = data["id"]
    return f"wss://connect.browserbase.com?apiKey={api_key}&sessionId={session_id}"


@contextmanager
def _browserbase_page(url: str) -> Iterator["object"]:
    """Yield a Playwright page connected to a fresh Browserbase browser at `url`.

    Always tears the remote browser down in a finally block.
    """
    from playwright.sync_api import sync_playwright

    connect_url = _create_session()
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(connect_url)
        try:
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            yield page
        finally:
            try:
                browser.close()
            except Exception:
                pass


# ─── Live match data ──────────────────────────────────────────────────────────

def get_live_data(home: str, away: str) -> dict:
    """Find a currently-live match between `home` and `away` and scrape its state.

    Returns {"live": True, "score": "2-1", "minute": 67,
             "red_cards": {...}, "possession": {...}} on success,
    else {"live": False}. Never raises.
    """
    if not _browserbase_enabled():
        return {"live": False}

    try:
        with _browserbase_page("https://www.bbc.com/sport/football/scores-fixtures") as page:
            try:
                page.wait_for_timeout(2_000)  # let the SPA hydrate
            except Exception:
                pass
            body = page.content()

        # The fixtures page lists EVERY match, so both team names and stray minute/score
        # markers from OTHER games appear all over it. To avoid declaring an unrelated match
        # "live 0-5", we scope detection to a tight window AROUND this fixture's two teams:
        # require them close together (same row) and look for the live marker + score only
        # inside that window. A future/non-live fixture has no minute marker beside it, so
        # this correctly falls through to {'live': False}.
        lower = body.lower()
        hi, ai = lower.find(home.lower()), lower.find(away.lower())
        if hi == -1 or ai == -1:
            return {"live": False}
        if abs(hi - ai) > 400:  # teams far apart → not the same fixture row → not our match
            return {"live": False}

        start = max(0, min(hi, ai) - 60)
        end = max(hi, ai) + len(away) + 160
        window = body[start:end]

        minute_match = re.search(r"\b(\d{1,3})\s*['′]", window)
        is_live = minute_match is not None or "LIVE" in window.upper()
        if not is_live:
            return {"live": False}

        score_match = re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", window)
        score = (
            f"{score_match.group(1)}-{score_match.group(2)}"
            if score_match else "0-0"
        )
        minute = int(minute_match.group(1)) if minute_match else None

        return {
            "live": True,
            "score": score,
            "minute": minute,
            "red_cards": {"home": 0, "away": 0},
            "possession": {},
        }
    except Exception as e:
        _log(f"get_live_data fell back: {type(e).__name__}")
        return {"live": False}


# ─── Prediction-market odds ─────────────────────────────────────────────────────

def get_market_odds(home: str, away: str) -> dict | None:
    """Best prediction-market price for `home` vs `away`.

    Tries each source in ODDS_SOURCE_ORDER and returns the first hit:
      - "browserbase": scrape Polymarket via the remote browser (headline integration).
      - "api":         the public Polymarket Gamma API (accurate, exact numbers).
    Returns {"source", "home_prob", "draw_prob", "away_prob"} (probs 0–1), or None.
    Never raises.
    """
    for src in ODDS_SOURCE_ORDER:
        try:
            odds = _scrape_odds_browserbase(home, away) if src == "browserbase" \
                else _api_market_odds(home, away)
            if odds:
                return odds
        except Exception as e:
            _log(f"odds source '{src}' fell back: {type(e).__name__}")
    return None


def _scrape_odds_browserbase(home: str, away: str) -> dict | None:
    """Scrape Polymarket via Browserbase. Returns a result ONLY when confident.

    Polymarket is a heavy bot-protected SPA, so this usually finds nothing and
    cleanly returns None — at which point get_market_odds falls through to the API.
    """
    if not _browserbase_enabled():
        return None

    query = f"{home} vs {away}".replace(" ", "%20")
    with _browserbase_page(f"https://polymarket.com/markets?_q={query}") as page:
        try:
            page.wait_for_timeout(2_500)
        except Exception:
            pass
        body = page.content()

    # Confident-only: require BOTH teams each immediately followed by a percentage,
    # e.g. "Japan 67%". Anything less → None (let the API answer accurately).
    def _pct_after(team: str) -> float | None:
        m = re.search(re.escape(team) + r"[^%0-9]{0,12}?(\d{1,3})\s*%", body, re.I)
        return int(m.group(1)) / 100.0 if m else None

    home_p, away_p = _pct_after(home), _pct_after(away)
    if home_p is None or away_p is None:
        return None
    draw_p = round(max(0.0, 1 - home_p - away_p), 2)
    return {
        "source": "Polymarket",
        "home_prob": round(home_p, 2),
        "draw_prob": draw_p,
        "away_prob": round(away_p, 2),
    }


# ─── Public prediction-market APIs (accurate fallback, no Browserbase) ────────────

def _norm(s: str) -> str:
    """Lowercase, strip accents/punctuation; apply team aliases."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s).strip()
    return _TEAM_ALIASES.get(s, s)


def _team_in(team: str, text: str) -> bool:
    """True if `team` (alias-normalised) shares a meaningful token with `text`."""
    t_tokens = [w for w in _norm(team).split() if len(w) > 2]
    text_n = _norm(text)
    return any(w in text_n for w in t_tokens)


def _api_market_odds(home: str, away: str) -> dict | None:
    """Resolve real head-to-head odds from the Polymarket Gamma API.

    Head-to-head events expose per-outcome "Yes/No" markets whose Yes price is the
    outcome probability (e.g. Tunisia 0.115 / Draw 0.205 / Japan 0.675). Returns None
    if no clean H2H market exists for this pairing (e.g. group-only markets).
    """
    headers = {"User-Agent": _BROWSER_UA}
    resp = httpx.get(
        f"{_GAMMA_BASE}/public-search",
        params={"q": f"{home} {away}", "limit_per_type": 10},
        headers=headers, timeout=15.0,
    )
    resp.raise_for_status()
    events = resp.json().get("events", [])

    event = _pick_h2h_event(events, home, away)
    if not event:
        return None

    det = httpx.get(
        f"{_GAMMA_BASE}/events/{event['id']}", headers=headers, timeout=15.0
    ).json()

    home_p = draw_p = away_p = None
    for m in det.get("markets", []):
        # Skip settled/half-priced legs — a resolved market reads as 1/0.
        if m.get("closed") or str(m.get("umaResolutionStatus", "")).lower() == "resolved":
            continue
        title = m.get("groupItemTitle") or m.get("question") or ""
        yes = _yes_price(m)
        if yes is None:
            continue
        tl = title.lower()
        if "draw" in tl or "tie" in tl:
            draw_p = yes
        elif _team_in(home, title):
            home_p = yes
        elif _team_in(away, title):
            away_p = yes

    if home_p is None or away_p is None:
        return None
    if draw_p is None:
        draw_p = max(0.0, 1 - home_p - away_p)

    # Sanity gate — reject resolved/degenerate/unpriced markets so we never show
    # nonsense like a 0.9995 draw or a 1.0 favourite. Returning None is better.
    total = home_p + draw_p + away_p
    if not (0.8 <= total <= 1.2):
        return None
    if max(home_p, draw_p, away_p) >= 0.97 or draw_p > 0.5:
        return None

    return {
        "source": "Polymarket",
        "home_prob": round(home_p, 2),
        "draw_prob": round(draw_p, 2),
        "away_prob": round(away_p, 2),
    }


def _pick_h2h_event(events: list[dict], home: str, away: str) -> dict | None:
    """Choose an OPEN soccer head-to-head event mentioning BOTH teams, else None."""
    _BAD = ("presidential", "election", "rocket league", "tennis", "davis cup",
            "six nations", "nba", "nfl", "method of win", "goals scored",
            "qualifiers", "friendlies")
    best, best_score = None, 0
    for e in events:
        title = e.get("title", "")
        tl = title.lower()
        if e.get("closed"):                          # skip already-played matches
            continue
        if " vs" not in tl and " v." not in tl:      # must be a head-to-head, not a group/tournament market
            continue
        if not (_team_in(home, title) and _team_in(away, title)):
            continue
        if any(b in tl for b in _BAD):
            continue
        score = 1 + (1 if "world cup" in tl else 0)
        if score > best_score:
            best, best_score = e, score
    return best


def _yes_price(market: dict) -> float | None:
    """Extract the 'Yes' outcome price (0–1) from a Gamma market dict."""
    import json as _json
    try:
        outcomes = market.get("outcomes")
        prices = market.get("outcomePrices")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        if isinstance(prices, str):
            prices = _json.loads(prices)
        if not outcomes or not prices:
            return None
        for name, price in zip(outcomes, prices):
            if str(name).strip().lower() == "yes":
                return float(price)
    except Exception:
        return None
    return None


if __name__ == "__main__":
    print("browserbase enabled:", _browserbase_enabled())
    print("odds (Tunisia/Japan):", get_market_odds("Tunisia", "Japan"))   # API path → fast + exact
    print("odds (no market)    :", get_market_odds("Germany", "Brazil"))  # honest None
    print("live (Germany/Brazil):", get_live_data("Germany", "Brazil"))   # Browserbase → {'live': False}
