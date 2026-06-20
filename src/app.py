"""Layer 3 — Voice loop (HERO): mic → Deepgram STT → analyst.ask() → Deepgram TTS → speaker.

Usage:
    python -m src.app

Say a matchup (e.g. "France versus Argentina, who wins?") to kick off.
Ask follow-up questions in plain English — the Oracle remembers the matchup.
Say "new match" to reset. Ctrl-C to exit.
"""

from __future__ import annotations

import os
import re
import threading
import time

import pyaudio
from deepgram import DeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted
from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd
from dotenv import load_dotenv

from src.analyst import ask

load_dotenv()

# ── Audio constants ───────────────────────────────────────────────────────────
MIC_RATE = 16_000
MIC_CHUNK = 1_024
TTS_RATE = 24_000
# Aura 2 male voices — swap at workshop: hermes (fast/crisp), draco (deep/dramatic)
TTS_VOICE = "aura-2-hermes-en"

# ── Team name extraction ──────────────────────────────────────────────────────
_TEAM_RE = re.compile(
    r"(?P<home>[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\s+"
    r"(?:vs?\.?\s+|versus\s+|against\s+)"
    r"(?P<away>[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})"
)


def _extract_teams(text: str) -> tuple[str, str] | None:
    m = _TEAM_RE.search(text)
    if m:
        home = m.group("home").title()
        away = m.group("away").title()
        if len(home) > 2 and len(away) > 2:
            return home, away
    return None


# ── TTS playback ──────────────────────────────────────────────────────────────
def _speak(dg: DeepgramClient, text: str) -> None:
    """Stream raw PCM bytes from Deepgram TTS and play through system speaker."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=TTS_RATE, output=True)
    try:
        for chunk in dg.speak.v1.audio.generate(
            text=text,
            model=TTS_VOICE,
            encoding="linear16",
            container="none",
            sample_rate=TTS_RATE,
        ):
            if chunk:
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Main voice loop ───────────────────────────────────────────────────────────
def run() -> None:
    dg = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
    pa = pyaudio.PyAudio()

    home: str | None = None
    away: str | None = None
    history: list[dict] = []
    buf: list[str] = []

    done = threading.Event()
    tts_active = threading.Event()  # mic sender pauses while Oracle is speaking

    print("\n[Oracle] Ready.")
    print("[Oracle] Say a matchup — e.g. 'France versus Argentina, who wins?'")
    print("[Oracle] Say 'new match' to reset.  Ctrl-C to exit.\n")

    with dg.listen.v1.connect(
        model="nova-3",
        encoding="linear16",
        sample_rate=MIC_RATE,
        punctuate=True,
        smart_format=True,
        utterance_end_ms=1200,
        vad_events=True,
        interim_results=False,
    ) as connection:
        mic = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )

        def _send_mic() -> None:
            while not done.is_set():
                if tts_active.is_set():
                    time.sleep(0.02)
                    continue
                try:
                    data = mic.read(MIC_CHUNK, exception_on_overflow=False)
                    connection.send_media(data)
                except Exception:
                    break

        threading.Thread(target=_send_mic, daemon=True).start()

        try:
            for msg in connection:
                if isinstance(msg, ListenV1SpeechStarted):
                    print("[Oracle] (listening…)", end="\r")

                elif isinstance(msg, ListenV1Results):
                    alts = msg.channel.alternatives if msg.channel else []
                    if alts and alts[0].transcript:
                        buf.append(alts[0].transcript)
                        print(f"  [you] {alts[0].transcript}")

                elif isinstance(msg, ListenV1UtteranceEnd):
                    if not buf:
                        continue

                    text = " ".join(buf).strip()
                    buf.clear()

                    if "new match" in text.lower():
                        home = away = None
                        history.clear()
                        print("[Oracle] Resetting. Say a new matchup.\n")
                        continue

                    detected = _extract_teams(text)
                    if detected:
                        home, away = detected
                        history.clear()
                        print(f"[Oracle] Matchup locked: {home} vs {away}")

                    if not home:
                        print("[Oracle] Hmm, say a matchup first — e.g. 'Brazil vs France'.\n")
                        continue

                    print(f"[Oracle] Thinking about {home} vs {away}…")
                    try:
                        response = ask(home, away, text, history=history)
                    except Exception as exc:
                        print(f"[Oracle] Error: {exc}")
                        continue

                    history += [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": response},
                    ]

                    print(f"\n[Oracle] {response}\n")
                    tts_active.set()
                    try:
                        _speak(dg, response)
                    finally:
                        tts_active.clear()

        except KeyboardInterrupt:
            print("\n[Oracle] Goodbye.")
        finally:
            done.set()
            mic.stop_stream()
            mic.close()
            pa.terminate()


if __name__ == "__main__":
    run()
