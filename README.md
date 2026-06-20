# World Cup Oracle

A talking football pundit powered by ML + Claude + Deepgram voice.

Speak a World Cup matchup. The Oracle listens, runs the numbers, and talks back — opinionated, grounded, theatrical.

Built at the **UC Berkeley AI Hackathon 2026** (June 20–21).

---

## What it does

1. **You speak** — _"France versus Argentina, who wins?"_
2. **Nova 3 transcribes** — Deepgram STT captures your question in real time
3. **The model predicts** — scikit-learn LogisticRegression trained on 49,000 international matches gives win/draw/loss probabilities, Elo ratings, and rolling form
4. **Claude analyzes** — the pundit persona gets the real numbers injected into its system prompt and delivers a grounded, entertaining verdict
5. **Aura 2 speaks** — Deepgram TTS reads the response back in a confident pundit voice
6. **You follow up** — ask what-ifs, injury questions, form queries — the Oracle remembers the matchup

---

## Demo

```
[Oracle] Ready. Say a matchup — e.g. 'France versus Argentina, who wins?'

  [you] France versus Argentina who wins

[Oracle] Thinking about France vs Argentina…

[Oracle] Right, here's the thing. Argentina's your favourites —
         the model backs that up entirely at 42.9% to win.
         France? 24.9%. That Elo gap — 48 points — is real and tangible.
         Argentina's been absolutely electric: 90% win rate, plus 2.5
         goals per game over their last ten. But this 32.2% draw
         probability? That tells you France can suffocate anyone.
         My call: Argentina, narrow. One goal, probably in extra time.
         [spoken aloud via Deepgram Aura 2]
```

---

## Setup

**Prerequisites:** Python 3.12, `portaudio` (for mic/speaker), API keys for Anthropic, Deepgram, football-data.org.

```bash
# macOS
brew install portaudio

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in: ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, FOOTBALL_DATA_API_KEY
```

## Run

```bash
# Full voice experience (HERO)
python -m src.app

# Text-only analyst test (5 scenarios: blowout, coin-flip, injury what-if, h2h, form)
python -m src.analyst

# Retrain the model (cached in data/model.pkl after first run)
python -m src.model
```

---

## How it's built

```
src/
  data.py      — fetches & caches international results + WC 2026 fixtures
  features.py  — incremental Elo (K=20) + rolling 10-match form per team
  model.py     — StandardScaler + LogisticRegression (3-class: W/D/L)
  analyst.py   — build_fact_block() → Claude system prompt → pundit response
  app.py       — Deepgram STT WebSocket → analyst.ask() → Deepgram TTS playback
```

**Key design:** `build_fact_block()` pulls live probabilities, Elo ratings, and form stats from the pickled model and injects them into Claude's system prompt as a `=== GROUNDED MODEL DATA ===` block. The persona is explicitly forbidden from citing any number not in that block — no hallucinated stats.

**Voice loop:** A background thread streams mic audio to Deepgram via WebSocket. A `threading.Event` mutes the mic during TTS playback to prevent echo. Utterance boundaries (`utterance_end_ms=1200`) trigger the Claude call.

---

## Stack

| Layer | Tech |
|---|---|
| Data | martj42/international-results (49k matches, 1872–present) |
| Prediction | Python 3.12, pandas, scikit-learn |
| Analysis | Anthropic Claude (claude-haiku-4-5, grounded system prompt) |
| Voice in | Deepgram Nova 3 (streaming STT) |
| Voice out | Deepgram Aura 2 Hermes (TTS) |

## Prizes targeted

- **Playground** (main track) — conversational AI experience
- **Deepgram** — most creative voice experience (voice is the core, not a bolt-on)
