# Cooking Assistant

A hands-free, real-time voice agent that cooks with you. Ask for a recipe, then
walk through it by voice: next step, go back, what's in it, scale it for six,
swap an ingredient, run several timers at once. You can talk over it to
interrupt, and timers announce themselves even mid-sentence.

```
mic ─► MicGate ─► STT (streaming) ──final──► LLM (streaming + tools) ─► sentence
        ▲  (echo gate, barge-in)                    │                    segmenter
        │                                     cooking session,               │
     speaker ◄── TTS (streaming) ◄──────────── timers, recipe  ◄─────────────┘
```

Everything runs on one asyncio event loop. The LLM and the TTS run
concurrently, so the first sentence is already being spoken while the model is
still writing the rest.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt && pip install -e .
```

```bash
cp .env.example .env   # then add GEMINI_API_KEY and (recommended) DEEPGRAM_API_KEY
```

Try it without keys or a microphone first:

```bash
cooking-agent demo
```

Then check your setup and talk to it:

```bash
cooking-agent diagnose
```

```bash
cooking-agent run
```

| Command | What it does |
|---|---|
| `cooking-agent run` | Live voice agent (mic + speakers). `--wake-word chef` to require a wake word, `--no-barge-in` to disable interrupting. |
| `cooking-agent chat` | Type instead of talk, with the real LLM and tools. `--speak` also speaks replies. |
| `cooking-agent demo` | Scripted offline session with fake providers: tools, a timer firing, a barge-in. |
| `cooking-agent bench` | End-to-end latency benchmark over WAV fixtures (see below). |
| `cooking-agent diagnose` | Lists audio devices, measures mic level, checks keys and LLM connectivity. |

## Providers

The agent talks to three small interfaces (`providers/base.py`), so each part
can be swapped by writing one adapter.

| Role | Default | Offline fallback |
|---|---|---|
| Speech-to-text | Deepgram Nova-3, streaming WebSocket with endpointing | Vosk (`VOSK_MODEL_PATH`) |
| LLM | Gemini (`google-genai`), streaming with function calling | none: an LLM key is required |
| Text-to-speech | Deepgram Aura-2, streaming WebSocket, kept open across turns | pyttsx3 (OS voice) |

With `STT_PROVIDER=auto` / `TTS_PROVIDER=auto` (the default) Deepgram is used
when `DEEPGRAM_API_KEY` is set and the offline providers otherwise. For the
offline ones: `pip install -e ".[offline]"`. All settings are listed in
[`.env.example`](.env.example).

## What it can do (LLM tools)

| Tool | Example |
|---|---|
| `get_recipe` | "Find me a recipe for jeera rice, vegan, for four." |
| `next_step` / `previous_step` / `repeat_step` | "Next." "Go back one." "Say that again." |
| `list_ingredients` | "What do I need?" |
| `scale_servings` | "Make it for six people." |
| `substitute_ingredient` | "I don't have ghee, what can I use?" |
| `set_timer` / `cancel_timer` / `list_timers` | "Set a rice timer for 12 minutes." "How long is left?" "Stop the pasta timer." |
| `end_session` | "That's all, thanks, bye." |

Recipes are generated with a pydantic schema as the LLM's structured-output
schema and validated on return (quantities, units, per-step timer hints), so
scaling and substitution work on real numbers rather than on free text.

## Latency

Each turn is instrumented (`metrics.py`); set `METRICS_LOG=metrics.jsonl` to
keep the records. The main number is **end of your speech → first audio out**,
which is the sum of:

* **STT endpointing**: how long the STT waits in silence before deciding you
  finished (`DEEPGRAM_ENDPOINTING_MS`, default 300 ms).
* **LLM first token**: lower with a smaller/faster model or
  `GEMINI_THINKING_LEVEL=low`.
* **TTS first chunk**: the first segment is cut early at a clause boundary so
  synthesis starts sooner.

For slow tools (`get_recipe` takes a few seconds), the agent speaks a short
filler line first so you never sit in silence.

### Benchmark

`bench/fixtures/` holds ten short WAV utterances (`manifest.json` lists the
text and the tool each one should trigger). The benchmark streams each one
through the full pipeline in real time, exactly as a mic would, against a
loaded recipe on step 2 with a timer running:

```bash
cooking-agent bench --runs 3 --out bench/results.json
```

It prints per-utterance lines and p50 / p90 / max for STT endpointing, LLM
first token, first audio and end-to-end, plus **tool routing accuracy** (did
the LLM call the right tool for what was said). Exit code is non-zero if any
utterance produced no turn.

`cooking-agent bench --fake --no-realtime` runs the same harness with offline
fake providers in under a second; it is what the test suite uses.

The fixtures are synthetic (espeak-ng, regenerate with `bench/make_fixtures.sh`).
Real recordings in your own kitchen are a better test: add 16-bit mono WAVs to
the folder and entries to `manifest.json`.

## Barge-in and echo

While the assistant is speaking, the mic is gated (replaced by silence before
it reaches the STT) so the assistant cannot hear and answer itself. If the mic
is clearly louder than what is being played for 200 ms, that is treated as you
talking over it: playback stops at once, the last 400 ms of your speech is
released to the STT so your first words are not lost, and the model is told on
the next turn that its previous reply was cut off.

This is an energy heuristic, not acoustic echo cancellation. It works well
with headphones or a speaker a little away from the mic. With loud speakers
right next to the mic, raise the threshold or use `--no-barge-in`.

## Development

```bash
pip install -r requirements-dev.txt && pip install -e .
```

```bash
pytest
```

The tests use deterministic fake STT/LLM/TTS providers and a simulated speaker,
so they need no keys, network or audio hardware.

```
src/cooking_agent/
  agent.py        turn loop: STT finals -> LLM rounds + tools -> segments -> TTS -> speaker; barge-in
  state.py        cooking-session and turn state machines
  tools.py        tool schemas (pydantic) and dispatcher
  recipe.py       recipe schema, quantity formatting
  timers.py       concurrent named asyncio timers
  segmenter.py    split streaming text into speakable sentences
  audio.py        mic / WAV sources, speaker sinks, MicGate
  metrics.py      per-turn latency records
  bench.py        latency benchmark
  config.py       env-driven provider selection
  providers/      gemini, deepgram, local (vosk, pyttsx3), fake
```
