# World Cup Oracle — Project Summary

**Status:** Phase 1–9 COMPLETE | Live Demo Ready  
**Started:** Jun 20, 2026 | Hackathon: UC Berkeley AI (Jun 20–21)  
**Tracks:** Playground (main) + Deepgram (voice) + Anthropic (eligible)

---

## What We Built

A **voice-powered World Cup match predictor** with an opinionated AI pundit oracle. Speak a matchup, get back live probability bars and a witty analysis in seconds.

### Core Features (Production Ready)

#### 🎤 Voice I/O (Deepgram)
- **STT:** Nova 3 with team-name keyterms to catch mishearings (e.g., "Cape Verde" vs. "Cape Verde Islands")
- **TTS:** Aura 2 Hermes (authoritative male pundit voice) with sentence-level streaming (oracle starts speaking in ~1s)
- **Hands-free mode:** VAD-based continuous listening with "stop" keyword detection
- **Mic controls:** Push-to-talk + hands-free toggle, mute, pause/replay for audio clips

#### 🧠 Prediction Engine
- **Model:** Multinomial LogisticRegression on Elo ratings + form features
- **Data:** 49,437 national-team matches (held-out test accuracy: 57.7% / 3-class)
- **Calibration:** Verified well-calibrated at production scale (predicted 40% → actual 39%)
- **Output:** Win / Draw / Loss probabilities for any matchup (including hypotheticals)

#### ⚽ Live Match Intelligence
- **Live scoring:** Parse spoken goals + minutes ("Spain losing 1-0 at 67 min")
- **In-play model:** Poisson-based goal prediction scaled by remaining time
- **Example:** 5-5@80' → Draw 82% | 2-0@70' → Home 96% | 0-1@80' → Away 95%
- **Grounding:** All spoken percentages derived from live_score adjustments, never invented

#### 📊 Web Integration
- **Fixtures view:** Upcoming World Cup matches sorted by kickoff (with PT timezone)
- **Odds consensus:** Polymarket (API) + Google search (Browserbase scrape) + Claude web search
- **Comparison:** Oracle vs. Market vs. Google vs. Web consensus side-by-side
- **Live refresh:** Auto-updates when a match is in-play (30s cadence)

#### 🎯 Claude Analyst
- **Model:** Claude Haiku 4.5 (low latency for voice) + Opus 4.8 (richer web search)
- **Grounding:** Fact block injected into system prompt (forbidden from inventing stats)
- **Persona:** Opinionated commentator, 30–45 word answers, natural football lingo
- **Multi-turn:** Conversation history per session; reset for a new matchup

#### 🧪 Test Coverage
- **69 offline edge-case assertions** covering:
  - Score parsing (ordinals, words, normalization: "eightieth"→80, "five nil"→5-0)
  - Minute-only parsing (splits multi-turn score + minute utterances)
  - Phantom-team rejection (strict matching, stopword filtering)
  - Alias resolution (Côte d'Ivoire→Ivory Coast, IR Iran→Iran)
  - Probability monotonicity (a lead at 80' > same lead at 70')
  - Orientation both ways (Spain losing 1-0 = Saudi winning 1-0)
  - Multi-turn merge (score in turn 1, minute in turn 2)

---

## Architecture

```
Football-data.org (WC fixtures + live data)
                    ↓
        [Browserbase scraper (CDP)]  ← Polymarket/Google odds
                    ↓
      [src/data.py] — cached fixtures + live status
                    ↓
[src/features.py] — Elo + form + aliases → resolve_team()
                    ↓
[src/model.py] — predict() + predict_live() + predict_score()
                    ↓
      [src/analyst.py] — Claude + web_search + ask()
                    ↓
[src/server.py] — FastAPI + SSE streaming + Deepgram I/O
      ↓                                           ↓
   /ask-stream (voice loop)              /web-odds (odds consensus)
      ↓                                           ↓
[static/index.html] — Web UI + mic + fixtures + prob bars
```

### Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `server.py` | 820 | FastAPI: /ask-stream (SSE), /fixtures, /predict, /web-odds, /live, /odds |
| `analyst.py` | 1100+ | Claude + web_search; parse_live_score, canonical_team, extract_matchup, ask |
| `model.py` | 220 | predict(), predict_live(), predict_score() |
| `features.py` | 180 | resolve_team(), compute features (Elo, form, rolling stats) |
| `scraper.py` | 550+ | Browserbase CDP sessions; get_live_data(), get_market_odds(), get_web_probabilities() |
| `index.html` | 1600+ | Web UI: mic, fixtures cards, SSE wiring, audio controls, hands-free VAD |
| `test_oracle.py` | 350+ | 69 offline edge-case assertions |

---

## What's Working Right Now

### ✅ Ready for Demo
1. **Web app:** `python -m src.server` → http://127.0.0.1:8000
2. **Voice:** Speak "Spain vs. Saudi Arabia" or "Spain leading 1-0 at 45 minutes"
3. **Fixtures:** Scrollable list with kickoff times (PT) + next-up badge
4. **Odds:** Click "🌐 Check the web" → Oracle vs. Market comparison
5. **Tests:** `python -m tests.test_oracle` → all 69 green
6. **Git:** Clean main branch; last commit = live-odds fallback (Jun 21)

### 🚀 Hackathon Targets Met
- **Playground:** Multimodal (voice + web UI) World Cup predictor ✓
- **Deepgram:** STT→Claude→TTS (voice is core experience) ✓
- **Anthropic:** Built with Claude Code + uses Haiku + Opus ✓

---

## What Could Be Done (Stretch Goals)

### 🎮 Visualization
- **Bracket simulator:** Run tournament from pre-match probabilities (16 → 8 → 4 → 2 → 1)
- **Dynamic bracket:** Color-code teams by win probability; animate matches as scores come in
- **Group stage heat map:** Win rates across all group matchups in a grid
- *Impact:* Best UI/UX prize target (Kodak cameras)

### 📈 Analytics
- **Head-to-head deep dive:** Scrape past matchup stats (goals, patterns, injuries)
- **Player-level influence:** Use team sheet + player Elo to adjust predictions
- **Injury impact:** "If Mbappe is out, draw goes from 12% to 15%"
- **Home-field advantage:** Factor in altitude, weather, crowd size

### 💰 Betting Integration
- **Live-odds arbitrage:** Show oracle-vs-market discrepancies (mispriced props)
- **Bet-sizing:** Kelly criterion → recommended bet amounts
- **Bet tracking:** Record user's bets → compare to actual outcomes
- *Note:* Would require compliance vetting for gambling (out of scope for hackathon)

### 🌍 Expansion
- **Multi-language:** TTS in Spanish, Portuguese, French (Aura supports these)
- **Club football:** Premier League, LaLiga, Champions League predictions (retrain model)
- **Mobile app:** React Native or Flutter wrapper around the FastAPI server
- **API for external consumers:** POST /predict → JSON (team B2B/media feeds)

### 🔧 Model Improvements
- **Ensemble:** LR + XGBoost + neural net majority vote
- **Dynamic Elo:** Weight recent matches higher (decay older games)
- **Soft-label training:** Use betting odds as a signal for edge cases
- **Counterfactual reasoning:** "If home Elo +100, win % goes to..."

### 🎯 Persona + UX
- **Regional commentary:** Tailored jokes for regional rivalries (UK vs. France, USA vs. Mexico)
- **Real-time stat callouts:** Mid-reply interrupt if a goal is scored (live-data polling)
- **Confidence scoring:** "I'm very sure" vs. "toss-up" based on prediction entropy
- **Replay analysis:** "Here's what changed after that goal"

### 🧪 Testing
- **CI/CD pipeline:** GitHub Actions → pytest, linting, security scan
- **Stress test:** Concurrent voice requests (simulated multi-user)
- **Deepgram fallback:** Mock STT/TTS for offline demo robustness
- **Odds scraper coverage:** Automated checks for Polymarket/Browserbase accuracy

---

## What Still Needs to be Done (Post-Hackathon)

### 📝 Devpost & Submission
- [ ] Record 60–90s demo video (open web UI, speak matchups, show Oracle talking + fixtures)
- [ ] Fill DEVPOST.md with team names + video link
- [ ] Submit before **Sun 12:00 PM** (edits close)

### 🎥 Demo Prep
- [ ] Test voice + fixtures with cached data (no live API calls)
- [ ] Fallback: screenshot static fixtures if Browserbase times out
- [ ] Dry-run with venue wifi (limited bandwidth)
- [ ] Pre-load 3–4 fixture cards so scrolling is instant

### 🔍 Known Gotchas / Edge Cases
| Issue | Status | Notes |
|-------|--------|-------|
| Browserbase scraper (live data) | ⚠️ Best-effort | Bot protection on Google; may return None on some searches |
| Multi-word team names | ✅ Fixed | "Cape Verde Islands" now caught via keyterms + canonical_team |
| Accented names (Curaçao) | ⚠️ Partial | Caught by difflib; accent-stripping in alias map would be cleaner |
| Rate limiting (football-data.org) | ✅ Cached | data/ directory persists; refresh on fresh branch |
| Deepgram sample_rate type | ✅ Fixed | Must be int, not float (Rust parser quirk) |
| Hands-free VAD (quiet starts) | ✅ Fixed | Added PEAK_FLOOR + 2-frame confirm; works now |

### 🎓 Future Hardening
- [ ] End-to-end test suite (full voice loop with mock Deepgram)
- [ ] Analyst refusal handling ("That's not a real matchup") → graceful feedback
- [ ] Rate-limiting headers for API endpoints (/fixtures, /odds)
- [ ] Telemetry: log request latencies, error rates, popular queries

### 📦 Deployment
- [ ] Docker image + docker-compose for reproducibility
- [ ] Heroku / Railway deploy template
- [ ] Environment docs (.env.example with all required keys)
- [ ] Setup script (venv + pip install + data cache warm-up)

---

## How to Run

### Prerequisites
```bash
# Python 3.12, secrets in .env
FOOTBALL_DATA_API_KEY=<key>        # football-data.org
ANTHROPIC_API_KEY=<key>             # Claude API
DEEPGRAM_API_KEY=<key>              # Deepgram STT/TTS
BROWSERBASE_API_KEY=<key>           # (optional, for live data)
BROWSERBASE_PROJECT_ID=<id>         # (optional)
```

### Start the Server
```bash
python -m src.server
# Opens http://127.0.0.1:8000 in your browser
```

### Run Tests
```bash
python -m tests.test_oracle
# 69 assertions, all green
```

### Example: Predict a Matchup (CLI)
```python
from src.analyst import ask
from src.data import load_fixtures

result = ask(
    home="Spain",
    away="Saudi Arabia",
    fixtures=load_fixtures(),
    model="claude-haiku-4-5-20251001"
)
print(result)
# Output: "Spain are heavy favorites..."
```

### Example: Live Score
```python
from src.model import predict_live

probs = predict_live(home="Spain", away="Saudi Arabia", hg=1, ag=0, minute=45, neutral=False)
# {'home': 0.81, 'draw': 0.15, 'away': 0.04}
```

---

## Tech Stack

| Layer | Tech |
|-------|------|
| **Backend** | Python 3.12 + FastAPI + Uvicorn |
| **ML** | scikit-learn (LogisticRegression) + pandas + numpy |
| **Voice** | Deepgram SDK (STT: Nova 3, TTS: Aura 2) |
| **LLM** | Claude Haiku 4.5 + Opus 4.8 (Anthropic SDK) |
| **Scraping** | Browserbase (CDP) + Playwright (browser automation) |
| **Frontend** | Vanilla JS (no framework) + Web Audio API (VAD) + SSE (streaming) |
| **Data** | football-data.org (live fixtures) + martj42/international_results (model training) |

---

## Key Metrics

- **Model accuracy:** 57.7% (3-class) on held-out test set
- **Prediction latency:** ~200ms (Haiku) + streaming TTS (first word ~1s)
- **Calibration:** Predicted 40% → Actual 39% (well-calibrated)
- **Test coverage:** 69 edge-case assertions (offline)
- **Fixture coverage:** 64 WC matches (full group stage + knockouts)
- **Voice latency fix:** Streaming TTS → Oracle speaks in 1s (was 3–5s)
- **Hands-free UX:** VAD-based with <100ms latency; works reliably above 0.016 RMS
- **Web odds:** Polymarket (API ~600ms) + Google (Browserbase fallback ~3s)

---

## Key Decisions Made

1. **Voice over bracket** → Deepgram prize + lower risk, high impact
2. **Haiku for voice, Opus for web** → Latency vs. richness trade-off
3. **Streaming SSE** → User sees Oracle talking while we're still generating
4. **Cached fixtures** → Rate-limit safety; real football-data.org data, no fabrication
5. **Grounded persona** → Fact block injected so Claude can't invent statistics
6. **Three-class output** → Win/Draw/Loss (not binary home-win only)
7. **Browserbase + API fallback** → Best-effort odds (API fast, scraper robust)

---

## Lessons Learned

### 🎯 What Worked
- Voice as the hero experience (judges love it)
- Streaming TTS (makes the Oracle feel responsive)
- Persona grounding (prevents hallucination)
- Comprehensive edge-case tests (caught 4 demo-killing bugs in Phase 9)
- Hands-free mode (freed the user to watch the screen)

### ⚠️ What Was Tricky
- STT mishearings (fixed with team-name keyterms + strict canonical matching)
- Score parsing ("80" → 8-0 bug; fixed with token-based normalization)
- Browserbase reliability (Google bot protection; fallback to API for safety)
- Deepgram gotchas (sample_rate int, interim_results affects utterance_end_ms)
- Web consensus latency (Claude web_search is slow; gated behind a button)

### 🔄 Iteration Loop
- Recorded demo → found bugs → fixed + tested → rinse repeat
- Phase 9 alone found + fixed 4 real issues (phantom teams, score parsing, orientation, Elo aliases)

---

## Files & Structure

```
worldcup-oracle/
├── CLAUDE.md                 # Project guide (this session's rulebook)
├── PLAN.md                   # Task checklist (phases 1–9)
├── MEMORY.md                 # Running log (decisions + gotchas)
├── PROJECT_SUMMARY.md        # This file
├── DEVPOST.md                # Hackathon submission (video + pitch)
├── README.md                 # Public-facing overview
├── .env                       # Secrets (never committed)
├── .env.example               # Template for .env
├── requirements.txt          # Python dependencies
├── data/
│   ├── fixtures_WC.json      # Cached World Cup 2026 matches (football-data.org)
│   ├── model.pkl             # Trained LogisticRegression + scaler (0.0 Elo drift)
│   └── ...                   # Training artifacts (international_results.csv cache)
├── src/
│   ├── data.py               # Fetch/cache fixtures + live data
│   ├── features.py           # Compute Elo + form; alias resolution
│   ├── model.py              # Predict + predict_live (Poisson) + predict_score
│   ├── analyst.py            # Claude + web_search; parse_live_score, ask()
│   ├── scraper.py            # Browserbase + Polymarket API + Google scraper
│   ├── server.py             # FastAPI: /ask-stream, /fixtures, /web-odds, etc.
│   └── app.py                # (legacy) Terminal voice loop (replaced by web UI)
├── static/
│   └── index.html            # Web UI: fixtures, mic, SSE wiring, hands-free VAD
├── tests/
│   ├── test_oracle.py        # 69 offline edge-case assertions
│   └── ...
└── .git/                      # Git history (6 commits, clean)
```

---

## Next Steps (Today)

1. **Record demo video** (60–90s) → upload to Devpost
2. **Fill DEVPOST.md** with team names + video link
3. **Submit before Sun 12 PM** (edits close)
4. **Dry-run demo** with cached data (no live API) at venue
5. **Show up at judges' table** (Sun 1–3 PM with full team)

---

## Contact & Attribution

- **Team:** Christian + ... (fill in)
- **Hackathon:** UC Berkeley AI (Jun 20–21, 2026)
- **Built with:** Claude Code + Claude API + Deepgram + football-data.org
- **Code:** https://github.com/... (update once repo is public)

---

**Last Updated:** Jun 21, 2026 (Phase 9 complete)
