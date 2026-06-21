"""Web UI backend — FastAPI app wrapping the Oracle for the browser.

Run:
    python -m src.server          # or: uvicorn src.server:app --reload

Endpoints:
    GET  /            → static/index.html (the mic + bracket UI)
    GET  /fixtures    → forecast_world_cup() with W/D/L probabilities
    POST /predict     → {home, away} → probs + grounded fact block
    POST /ask         → multipart {session_id, audio?|text?}
                        Deepgram STT → analyst.ask() → Deepgram TTS.
                        Returns {transcript, home, away, text, probs, audio (wav b64)}.

Voice path keeps Deepgram STT (Nova 3) + TTS (Aura 2) as the core experience.
"""

from __future__ import annotations

import asyncio
import base64
import json as _json
import os
import re
from pathlib import Path

import anthropic as _anthropic
from deepgram import DeepgramClient
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.analyst import (
    ask,
    build_fact_block,
    build_system_prompt,
    extract_matchup,
    parse_live_score,
    parse_minute_only,
    parse_score_only,
    _rule_based_verdict,
)
from src.data import upcoming_fixtures
from src.model import predict, predict_live, predict_score

load_dotenv()

_async_anthropic: _anthropic.AsyncAnthropic | None = None


def _get_async_anthropic() -> _anthropic.AsyncAnthropic | None:
    global _async_anthropic
    if _async_anthropic is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return None
        _async_anthropic = _anthropic.AsyncAnthropic(
            api_key=key, timeout=20.0, max_retries=2
        )
    return _async_anthropic

STATIC_DIR = Path(__file__).parent.parent / "static"

# Aura 2 voice — authoritative male, fits the pundit persona.
TTS_VOICE = "aura-2-hermes-en"
TTS_RATE = 24_000

app = FastAPI(title="World Cup Oracle")

_dg = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])

# In-memory per-browser session state: {session_id: {home, away, history}}.
# Plenty for a single-table hackathon demo; no persistence needed.
_SESSIONS: dict[str, dict] = {}

# Cache the fixture forecast. Short TTL so in-play games show up within 2 minutes.
import time as _time
_FIXTURES_CACHE: list[dict] | None = None
_FIXTURES_CACHE_AT: float = 0.0
_FIXTURES_TTL = 90  # seconds — short enough to catch a game going IN_PLAY

# Cache the STT keyterm list (team names) so we build it once.
_KEYTERMS_CACHE: list[str] | None = None

# Spoken aliases STT is likely to emit for awkward country names — boosting these
# alongside the canonical spelling cuts the "Curaçao → Kuruchayo" class of misfires.
_TEAM_ALIASES = {
    "United States": ["USA", "United States"],
    "IR Iran": ["Iran"],
    "Côte d'Ivoire": ["Ivory Coast", "Cote d'Ivoire"],
    "Cape Verde Islands": ["Cape Verde"],
    "South Korea": ["Korea"],
    "Curaçao": ["Curacao"],
    "Bosnia-Herzegovina": ["Bosnia"],
}


def _fixtures() -> list[dict]:
    global _FIXTURES_CACHE, _FIXTURES_CACHE_AT
    now = _time.monotonic()
    # Bust the cache if it's stale OR if any cached game is currently in-play
    # (so the sidebar stays fresh without restarting the server).
    cache_stale = (now - _FIXTURES_CACHE_AT) > _FIXTURES_TTL
    has_live = any(f.get("status") in ("IN_PLAY", "PAUSED") for f in (_FIXTURES_CACHE or []))
    if _FIXTURES_CACHE is None or cache_stale or has_live:
        try:
            fx = upcoming_fixtures("WC")
        except Exception:
            fx = _FIXTURES_CACHE or []
        _FIXTURES_CACHE = [f for f in fx if f.get("home_team") and f.get("away_team")]
        _FIXTURES_CACHE_AT = now
    return _FIXTURES_CACHE


def _session(session_id: str) -> dict:
    return _SESSIONS.setdefault(session_id, {"home": None, "away": None, "history": [], "live_score": None})


_WIN_VERBS = r"winning|leading|ahead|in front|on top|up"
_LOSE_VERBS = r"losing|trailing|behind|chasing|lost|down"


def _team_tokens(name: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z ]", " ", name.lower()).split() if len(t) >= 3]


def _team_state(transcript: str, tokens: list[str]) -> str | None:
    """Return 'win' / 'lose' if the transcript says this team is winning/losing, else None."""
    if not tokens:
        return None
    alt = "|".join(re.escape(t) for t in tokens)
    win_re = re.compile(
        rf"\b(?:{alt})\b.{{0,40}}\b(?:{_WIN_VERBS})\b"
        rf"|\b(?:{_WIN_VERBS})\b.{{0,25}}\b(?:{alt})\b",
        re.I,
    )
    lose_re = re.compile(
        rf"\b(?:{alt})\b.{{0,40}}\b(?:{_LOSE_VERBS})\b"
        rf"|\b(?:{_LOSE_VERBS})\b.{{0,25}}\b(?:{alt})\b",
        re.I,
    )
    if win_re.search(transcript):
        return "win"
    if lose_re.search(transcript):
        return "lose"
    return None


def _orient_score(
    live_score: tuple[int, int, int],
    transcript: str,
    home: str,
    away: str,
) -> tuple[int, int, int]:
    """Orient (home_goals, away_goals) so the leader matches what the transcript says.

    Handles, in priority order:
    - "Spain are losing one-nil" / "Spain are two-nil up" → verb attached to a team
    - "Iran winning" / "Brazil leading"                  → away team in front
    - "Iran 1-0 at 20 minutes"                           → away name before the score
    The parser returns goals in spoken order; this decides which side they belong to.
    """
    hg, ag, minute = live_score
    if hg == ag or not home or not away:
        return live_score

    hi, lo = max(hg, ag), min(hg, ag)
    home_tokens, away_tokens = _team_tokens(home), _team_tokens(away)

    # 1. Explicit winning/losing verb tied to either team.
    home_state = _team_state(transcript, home_tokens)
    away_state = _team_state(transcript, away_tokens)
    home_ahead: bool | None = None
    if home_state == "win" or away_state == "lose":
        home_ahead = True
    elif home_state == "lose" or away_state == "win":
        home_ahead = False

    if home_ahead is True:
        return hi, lo, minute
    if home_ahead is False:
        return lo, hi, minute

    # 2. Fallback: away team name appears before the first score digit ("Iran 1-0").
    tl = transcript.lower()
    if away_tokens:
        score_pos = len(tl)
        for n in (hg, ag):
            p = tl.find(str(n))
            if p != -1 and p < score_pos:
                score_pos = p
        for tok in away_tokens:
            pos = tl.find(tok)
            if pos != -1 and pos < score_pos:
                return ag, hg, minute

    return live_score


def _keyterms() -> list[str]:
    """Team names (+ spoken aliases) to boost in Deepgram Nova-3 STT.

    Scoped to the ~48 nations actually in the WC fixtures so proper nouns like
    Curaçao / Cape Verde / Uzbekistan get recognised instead of mangled.
    """
    global _KEYTERMS_CACHE
    if _KEYTERMS_CACHE is None:
        kt: set[str] = set()
        for f in _fixtures():
            for t in (f.get("home_team"), f.get("away_team")):
                if t:
                    kt.add(t)
        for aliases in _TEAM_ALIASES.values():
            kt.update(aliases)
        _KEYTERMS_CACHE = sorted(kt)
    return _KEYTERMS_CACHE


# Markdown the model occasionally emits despite the persona rules. Aura 2 pronounces these
# literally ("star", "hash"), and they look wrong in the chat, so strip them before TTS/display.
_MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_RULE = re.compile(r"(?m)^\s*([-*_=])\1{2,}\s*$")
_MD_EMPH = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_MD_BLANKS = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Remove markdown artefacts (headings, rules, *emphasis*, stray # / *) for clean speech."""
    text = _MD_HEADING.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_EMPH.sub(r"\2", text)
    text = text.replace("*", "").replace("#", "")
    text = _MD_BLANKS.sub("\n\n", text)
    return text.strip()


def _tts_wav(text: str) -> str:
    """Synthesise speech with Deepgram Aura 2 and return base64-encoded WAV.

    Returns "" when the text is empty after stripping (e.g. a lone "---" line) so
    callers can skip emitting a silent clip.
    """
    text = strip_markdown(text)
    if not text:
        return ""
    chunks = _dg.speak.v1.audio.generate(
        text=text,
        model=TTS_VOICE,
        encoding="linear16",
        container="wav",
        sample_rate=TTS_RATE,
    )
    audio = b"".join(c for c in chunks if c)
    return base64.b64encode(audio).decode("ascii")


_SENTENCE_END = re.compile(r'[.!?…]+["\')\]]?(?:\s|$)')


def _split_complete_sentences(buf: str) -> tuple[str, str]:
    """Split off everything up to the last completed sentence boundary.

    Returns (ready, remainder). Used to flush TTS sentence-by-sentence so the
    Oracle starts speaking ~1 sentence in instead of after the whole answer.
    """
    matches = list(_SENTENCE_END.finditer(buf))
    if not matches:
        return "", buf
    idx = matches[-1].end()
    return buf[:idx], buf[idx:]


def _stt(audio_bytes: bytes) -> str:
    """Transcribe an uploaded audio blob with Deepgram Nova 3 (prerecorded REST).

    No encoding/sample_rate passed → Deepgram sniffs the container (webm/ogg/wav).
    """
    resp = _dg.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        smart_format=True,
        punctuate=True,
        keyterm=_keyterms(),  # boost WC nation names so STT stops mangling them
    )
    try:
        return resp.results.channels[0].alternatives[0].transcript.strip()
    except (AttributeError, IndexError):
        return ""


# ─── Browserbase scrapers (best-effort, never crash the voice loop) ───────────────

def _safe_live(home: str, away: str) -> dict:
    """Live match data via Browserbase, degrading to {'live': False} on any failure."""
    try:
        from src.scraper import get_live_data

        return get_live_data(home, away) or {"live": False}
    except Exception:
        return {"live": False}


def _safe_odds(home: str, away: str) -> dict | None:
    """Prediction-market odds via Browserbase, degrading to None on any failure."""
    try:
        from src.scraper import get_market_odds

        return get_market_odds(home, away)
    except Exception:
        return None


def _safe_web_probs(home: str, away: str) -> dict | None:
    """Google-scraped win probabilities (Browserbase), degrading to None on failure."""
    try:
        from src.scraper import get_web_probabilities

        return get_web_probabilities(home, away)
    except Exception:
        return None


def _safe_web_consensus(home: str, away: str) -> dict | None:
    """Claude web-search consensus, degrading to None on any failure."""
    try:
        from src.analyst import web_consensus

        return web_consensus(home, away)
    except Exception:
        return None


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/fixtures")
def fixtures() -> JSONResponse:
    """Upcoming WC fixtures, each enriched with W/D/L probabilities."""
    results = []
    for fix in _fixtures():
        try:
            probs = predict(fix["home_team"], fix["away_team"], neutral=fix.get("neutral", True))
        except Exception:
            continue
        results.append({**fix, "probs": probs})
    return JSONResponse(results)


class PredictRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True


@app.post("/predict")
def predict_endpoint(req: PredictRequest) -> JSONResponse:
    facts = build_fact_block(req.home, req.away, req.neutral)
    try:
        facts["xg"] = predict_score(req.home, req.away, req.neutral)
    except Exception:
        facts["xg"] = None
    return JSONResponse(facts)


@app.get("/predict_score")
def predict_score_endpoint(home: str, away: str, neutral: bool = True) -> JSONResponse:
    """Expected-goals scoreline derived from the model probabilities."""
    try:
        return JSONResponse(predict_score(home, away, neutral))
    except Exception:
        return JSONResponse({"home_xg": None, "away_xg": None,
                             "home_team": home, "away_team": away})


@app.get("/live")
async def live_endpoint(home: str, away: str) -> JSONResponse:
    """Best-effort live-match data. Always returns a dict."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _safe_live, home, away)
    return JSONResponse(data)


@app.get("/live-prob")
async def live_prob_endpoint(home: str, away: str) -> JSONResponse:
    """Live match data + live-adjusted win probabilities in one call.

    Returns {live: {...}, live_probs: {...}|null}.
    live_probs is null when no live match is found or when minute is unavailable.
    """
    loop = asyncio.get_running_loop()
    live = await loop.run_in_executor(None, _safe_live, home, away)

    live_probs = None
    if live and live.get("live"):
        score_str = live.get("score", "0-0")
        minute = live.get("minute")
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", score_str or "")
        if m and minute is not None:
            hg, ag = int(m.group(1)), int(m.group(2))
            try:
                live_probs = predict_live(home, away, hg, ag, int(minute), neutral=True)
            except Exception:
                pass

    return JSONResponse({"live": live, "live_probs": live_probs})


@app.get("/odds")
async def odds_endpoint(home: str, away: str) -> JSONResponse:
    """Best-effort prediction-market odds (Browserbase). null if not found."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _safe_odds, home, away)
    return JSONResponse(data)


@app.get("/web-odds")
async def web_odds_endpoint(home: str, away: str) -> JSONResponse:
    """Compare the Oracle's model against how the web prices the match.

    Aggregates every available source, each best-effort and independent:
      - oracle:     our model's W/D/L (always present)
      - market:     Polymarket prediction-market odds (fast API, may be null)
      - google:     Google's win-probability scrape via Browserbase (may be null)
      - web:        Claude web-search consensus with citations (slower, may be null)

    Gated behind an explicit call (a "check the web" button), NOT the voice turn,
    because the Claude web search adds a few seconds.
    """
    loop = asyncio.get_running_loop()
    try:
        oracle = predict(home, away, neutral=True)
    except Exception:
        oracle = None

    market, google, web = await asyncio.gather(
        loop.run_in_executor(None, _safe_odds, home, away),
        loop.run_in_executor(None, _safe_web_probs, home, away),
        loop.run_in_executor(None, _safe_web_consensus, home, away),
    )
    return JSONResponse(
        {"home": home, "away": away, "oracle": oracle,
         "market": market, "google": google, "web": web}
    )


@app.post("/ask")
async def ask_endpoint(
    session_id: str = Form(...),
    text: str | None = Form(None),
    audio: UploadFile | None = None,
) -> JSONResponse:
    """Core voice turn: speech (or text) in → grounded spoken verdict out."""
    sess = _session(session_id)

    # 1. Get the user utterance (Deepgram STT for audio, else text input).
    if audio is not None:
        transcript = _stt(await audio.read())
    else:
        transcript = (text or "").strip()

    if not transcript:
        return JSONResponse(
            {"transcript": "", "text": "I didn't catch that — try again.",
             "home": sess["home"], "away": sess["away"], "probs": None,
             "audio": _tts_wav("I didn't catch that. Say a matchup and try again.")}
        )

    # 2. Reset command.
    if "new match" in transcript.lower() or transcript.lower().strip(" .") == "reset":
        sess.update(home=None, away=None, history=[])
        msg = "Cleared. Give me a new matchup."
        return JSONResponse(
            {"transcript": transcript, "text": msg, "home": None, "away": None,
             "probs": None, "audio": _tts_wav(msg)}
        )

    # 3. Detect a (new) matchup from free-form speech. Falls back to the
    #    current matchup for follow-ups ("what if their striker is injured?").
    detected = extract_matchup(
        transcript, _fixtures(), current=(sess["home"], sess["away"])
    )
    if detected:
        new_home, new_away = detected
        if new_home != sess["home"] or new_away != sess["away"]:
            sess.update(home=new_home, away=new_away, history=[], live_score=None)
        else:
            sess["home"], sess["away"] = new_home, new_away

    if not sess["home"]:
        msg = "Name a matchup first — for example, France versus Brazil."
        return JSONResponse(
            {"transcript": transcript, "text": msg, "home": None, "away": None,
             "probs": None, "audio": _tts_wav(msg)}
        )

    home, away = sess["home"], sess["away"]

    # 4. Grounded pundit verdict + 5. spoken response.
    response = ask(home, away, transcript, history=sess["history"])
    sess["history"] += [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": response},
    ]

    try:
        probs = predict(home, away, neutral=True)
    except Exception:
        probs = None

    return JSONResponse(
        {
            "transcript": transcript,
            "home": home,
            "away": away,
            "text": response,
            "probs": probs,
            "audio": _tts_wav(response),
        }
    )


@app.post("/ask-stream")
async def ask_stream_endpoint(
    session_id: str = Form(...),
    text: str | None = Form(None),
    audio: UploadFile | None = None,
    use_live: str | None = Form(None),
) -> StreamingResponse:
    """Streaming SSE version of /ask — emits tokens as Claude generates them.

    Events: transcript | matchup | token | audio | done
    """
    audio_bytes = (await audio.read()) if audio is not None else None
    sess = _session(session_id)

    async def _generate():
        loop = asyncio.get_running_loop()

        # 1. Transcribe (Deepgram prerecorded is sync — run in thread).
        if audio_bytes is not None:
            transcript = await loop.run_in_executor(None, _stt, audio_bytes)
        else:
            transcript = (text or "").strip()

        yield f"data: {_json.dumps({'type': 'transcript', 'text': transcript})}\n\n"

        if not transcript:
            msg = "I didn't catch that — try again."
            b64 = await loop.run_in_executor(None, _tts_wav, msg)
            yield f"data: {_json.dumps({'type': 'done', 'text': msg, 'audio': b64})}\n\n"
            return

        # 2. Reset command.
        if "new match" in transcript.lower() or transcript.lower().strip(" .") == "reset":
            sess.update(home=None, away=None, history=[], live_score=None, score_only=None)
            msg = "Cleared. Give me a new matchup."
            b64 = await loop.run_in_executor(None, _tts_wav, msg)
            yield f"data: {_json.dumps({'type': 'done', 'text': msg, 'audio': b64, 'home': None, 'away': None, 'reset': True})}\n\n"
            return

        # 3. Detect matchup — only reset history when it's a *different* pair.
        #    Pass the current matchup so switching is conservative (a misheard word
        #    can't spawn a phantom opponent; follow-ups keep the current pair).
        _cur = (sess["home"], sess["away"])
        detected = await loop.run_in_executor(
            None, lambda: extract_matchup(transcript, _fixtures(), current=_cur)
        )
        if detected:
            new_home, new_away = detected
            if new_home != sess["home"] or new_away != sess["away"]:
                sess.update(home=new_home, away=new_away, history=[], live_score=None, score_only=None)
            else:
                sess["home"], sess["away"] = new_home, new_away

        if not sess["home"]:
            msg = "Name a matchup first — for example, France versus Brazil."
            b64 = await loop.run_in_executor(None, _tts_wav, msg)
            yield f"data: {_json.dumps({'type': 'done', 'text': msg, 'audio': b64})}\n\n"
            return

        home, away = sess["home"], sess["away"]

        try:
            probs = predict(home, away, neutral=True)
        except Exception:
            probs = None

        try:
            xg = predict_score(home, away, neutral=True)
        except Exception:
            xg = None

        # Fetch live data when explicitly requested OR when the user's question
        # mentions live context (score, current state, right now, etc.).
        _LIVE_RE = re.compile(
            r"\b(live|score|right now|currently|in.play|minute|half.time|halftime"
            r"|goal|update|check|going on|happening)\b", re.I
        )
        _want_live = (use_live or "").lower() == "true" or bool(_LIVE_RE.search(transcript))
        live_data = None
        if _want_live:
            live_data = await loop.run_in_executor(None, _safe_live, home, away)

        # Scoreline parsing — two tiers:
        #   1. score + minute → predict_live gives real adjusted probabilities
        #   2. score only (no minute) → stored as score_only; Oracle qualitatively
        #      acknowledges the score and asks for the clock (wrong minute → wrong %)
        live_score = parse_live_score(transcript)
        if live_score is not None and home and away:
            live_score = _orient_score(live_score, transcript, home, away)
        score_only: tuple[int, int] | None = None

        if live_score is not None:
            # Full score+minute: persist in session and compute live probs.
            sess["live_score"] = live_score
            sess["score_only"] = None
        else:
            # Check if utterance has a score but no minute.
            raw_score = parse_score_only(transcript)
            minute_only = parse_minute_only(transcript)
            if raw_score is not None:
                # Orient NOW using this turn's team mentions — a later turn that
                # only supplies the minute ("eighty") won't name the teams.
                oriented = _orient_score((raw_score[0], raw_score[1], 90), transcript, home, away)
                raw_score = (oriented[0], oriented[1])
                if minute_only is not None:
                    # Score and minute both landed this turn → go live immediately.
                    live_score = (raw_score[0], raw_score[1], minute_only)
                    sess["live_score"] = live_score
                    sess["score_only"] = None
                else:
                    # Store score-only; don't overwrite a previously known minute.
                    sess["score_only"] = raw_score
                    score_only = raw_score
            elif minute_only is not None and sess.get("score_only"):
                # The clock arrives in a later turn than the score → combine them.
                sh, sa = sess["score_only"]
                live_score = (sh, sa, minute_only)
                sess["live_score"] = live_score
                sess["score_only"] = None
            elif sess.get("live_score"):
                # Follow-up with no new score → reuse the last full score.
                live_score = sess["live_score"]
                # "What if [team] scores" → increment the appropriate goal.
                _WHATIF_RE = re.compile(r"\bif\b.{0,30}\bscores?\b|\bscores?\b.{0,20}\bwhat\b", re.I)
                if _WHATIF_RE.search(transcript) and live_score is not None:
                    hg, ag, min_ = live_score
                    home_re = re.compile(r"\b" + re.escape(home) + r"\b", re.I)
                    away_re = re.compile(r"\b" + re.escape(away) + r"\b", re.I)
                    if home_re.search(transcript):
                        live_score = (hg + 1, ag, min_)
                    elif away_re.search(transcript):
                        live_score = (hg, ag + 1, min_)
            elif sess.get("score_only"):
                score_only = sess["score_only"]

        live_probs = None
        if live_score is not None:
            hg, ag, minute = live_score
            try:
                live_probs = predict_live(home, away, hg, ag, minute, neutral=True)
            except Exception:
                live_probs = None

        yield f"data: {_json.dumps({'type': 'matchup', 'home': home, 'away': away, 'probs': probs, 'xg': xg, 'live': live_data, 'live_probs': live_probs})}\n\n"

        # 4. Stream Claude tokens (or fall back to rule-based if Claude is unavailable).
        system = build_system_prompt(
            home, away, neutral=True, live_data=live_data, live_score=live_score,
            score_only=score_only,
        )
        messages = list(sess["history"]) + [{"role": "user", "content": transcript}]

        full_text = ""
        tts_buf = ""
        first_flush_done = False
        _use_fallback = False

        async_client = _get_async_anthropic()
        if async_client is None:
            _use_fallback = True

        if not _use_fallback:
            try:
                async with async_client.messages.stream(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    system=system,
                    messages=messages,
                ) as stream:
                    async for chunk in stream.text_stream:
                        full_text += chunk
                        tts_buf += chunk
                        yield f"data: {_json.dumps({'type': 'token', 'text': chunk})}\n\n"

                        ready, rest = _split_complete_sentences(tts_buf)
                        if ready and (not first_flush_done or len(ready) >= 130):
                            tts_buf = rest
                            first_flush_done = True
                            b64 = await loop.run_in_executor(None, _tts_wav, ready)
                            if b64:
                                yield f"data: {_json.dumps({'type': 'audio', 'data': b64})}\n\n"
            except (_anthropic.AuthenticationError, _anthropic.PermissionDeniedError):
                _use_fallback = True
            except _anthropic.BadRequestError as exc:
                if "credit balance is too low" in str(exc):
                    _use_fallback = True
                else:
                    yield f"data: {_json.dumps({'type': 'error', 'text': f'Anthropic request error: {exc}'})}\n\n"
                    return
            except _anthropic.APIError as exc:
                yield f"data: {_json.dumps({'type': 'error', 'text': f'Anthropic API error: {exc}'})}\n\n"
                return

        if _use_fallback and not full_text:
            facts = build_fact_block(home, away, live_score=live_score, score_only=score_only)
            full_text = _rule_based_verdict(facts, transcript)
            yield f"data: {_json.dumps({'type': 'token', 'text': full_text})}\n\n"
            tts_buf = full_text

        # 5. Flush any trailing text that never hit a sentence boundary.
        if tts_buf.strip():
            b64 = await loop.run_in_executor(None, _tts_wav, tts_buf)
            if b64:
                yield f"data: {_json.dumps({'type': 'audio', 'data': b64})}\n\n"

        sess["history"] += [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": full_text},
        ]

        yield f"data: {_json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the rest of static/ (favicon, etc.) if present.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)
