# Cooking Assistant

A hands-free, real-time voice agent that cooks with you. Ask for a recipe, then
walk through it by voice at your own pace: after every step it offers a timer
for the time the step states and asks if you're ready to move on. Scale it for
six, swap an ingredient, run several timers at once. Keep your own recipe book
(import from a web page, a file, or just dictate a recipe), and it learns your
preferences (vegetarian, allergies, what you usually have at home) and tailors
new recipes to them. You can talk over it to interrupt, and timers announce
themselves even mid-sentence.

```
mic ─► MicGate ─► STT (streaming) ──final──► LLM (streaming + tools) ─► sentence
        ▲  (echo gate, barge-in)                    │                    segmenter
        │                                     cooking session,               │
     speaker ◄── TTS (streaming) ◄──────────── timers, recipe  ◄─────────────┘
```

Everything runs on one asyncio event loop. The LLM and the TTS run
concurrently, so the first sentence is already being spoken while the model is
still writing the rest.

## Setup

Requires Python 3.11 or newer (`python --version`; on Windows `py -0` lists
what is installed). No other system tools are needed.

### 1. Create a virtual environment and install

**Windows (PowerShell)**, in the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

When the environment is active your prompt starts with `(.venv)`. Activate it
again in every new terminal before using `cooking-agent`.

### 2. Add your API keys

Copy the template and fill it in (`copy .env.example .env` on Windows,
`cp .env.example .env` elsewhere):

* No key is needed for the AI itself: by default it runs on your computer
  through Ollama (next section).
* `GEMINI_API_KEY` (**optional, recommended for importing recipes**): used
  to read recipe photos, PDFs and recordings, and to transcribe videos when
  there is no Deepgram key. Free key at <https://aistudio.google.com/apikey>.
  To use Gemini for the conversation too, set `LLM_PROVIDER=gemini`.
* `DEEPGRAM_API_KEY` (**optional**): fast, accurate speech in and out.
  Sign up at <https://console.deepgram.com>; new accounts get free credit.

`.env` is git-ignored; never commit it.

### Local AI with Ollama

The assistant's AI runs locally with [Ollama](https://ollama.com): free, no
rate limits, and it works without internet once the model is downloaded.
A graphics card with 8 GB of memory runs a 7–9B model comfortably.

```powershell
winget install Ollama.Ollama      # Windows; on macOS/Linux see ollama.com/download
ollama pull qwen3:8b              # about 5 GB, once
```

Ollama runs in the background after install. `cooking-agent diagnose` checks
it. To use another model: `ollama pull <model>` and set `OLLAMA_MODEL` in
`.env`.

While the assistant is running it keeps the model loaded, so the answer
after a long simmer isn't a cold start: it pings the model every 15 minutes
of quiet and again about a minute before each timer ends. The model's graphics
memory (about 6 GB) stays in use until you quit the assistant; Ollama frees
it 30 minutes later.

### 3. Speech without Deepgram (optional, offline)

Without a Deepgram key the voice mode uses free offline speech: Vosk to hear
you and your computer's own voice (pyttsx3) to talk. Slower and less accurate,
but no account needed.

```bash
pip install -e ".[offline]"
```

Then download and unzip a Vosk model, e.g.
[vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
(about 40 MB), and point `.env` at the unzipped folder:

```
VOSK_MODEL_PATH=C:\Users\you\.cooking-agent\models\vosk-model-small-en-us-0.15
```

## Running it

Go from simplest to full voice:

| Step | Command | Needs | What you should see |
|---|---|---|---|
| 1 | `cooking-agent demo` | nothing | A scripted cooking session printed as `[you]` / `[assistant]` lines. Proves the install works. |
| 2 | `cooking-agent diagnose` | `.env` | `[ok]` / `[FAIL]` for audio devices, mic level (speak when asked), speech in, speech out and the AI. Fix any `FAIL` before step 4. `--mic-seconds 0` skips the mic test. |
| 3 | `cooking-agent chat` | Ollama + a model (or a Gemini key with `LLM_PROVIDER=gemini`) | Type to the real assistant. Try "find me a recipe for jeera rice for two", then "ready". Empty line quits. |
| 4 | `cooking-agent run` | the AI as in step 3, plus a Deepgram key *or* the offline speech setup | Talk to it. Say "find me a recipe for fried rice". Ctrl+C quits. |

The first time you use `chat` or `run`, it asks about allergies, health
conditions and diet before anything else. Answer, or say "none" or "skip".

Use headphones for `run` if you can: the assistant mutes the mic while it is
speaking, and headphones keep it from hearing itself.

### Troubleshooting

* **`No matching distribution found for numpy...`**: your Python is older than
  3.11. Install 3.11 or newer and recreate `.venv`.
* **`cooking-agent` is not recognised**: the virtual environment is not active
  (step 1), or `pip install -e .` was not run.
* **`[FAIL] stt: VOSK_MODEL_PATH must point to...`**: no way to hear you yet.
  Add `DEEPGRAM_API_KEY`, or do the offline setup (step 3).
* **Replies take many seconds**: check your connection. Time to first word is
  normally around a second; set `GEMINI_THINKING_LEVEL=low` in `.env` to speed
  the AI up further.
* **It answers itself or stops talking at random**: speaker sound is reaching
  the mic. Use headphones, or start with `cooking-agent run --no-barge-in`.

## All commands

| Command | What it does |
|---|---|
| `cooking-agent run` | Live voice agent (mic + speakers). `--wake-word chef` to require a wake word, `--no-barge-in` to disable interrupting. |
| `cooking-agent chat` | Type instead of talk, with the real LLM and tools. `--speak` also speaks replies. |
| `cooking-agent demo` | Scripted offline session with fake providers: step pacing, a timer firing, a barge-in, dictating a recipe. |
| `cooking-agent recipes add <link-or-file>` | Save a recipe from a web page, YouTube, Instagram, TikTok, a photo, PDF, audio or video. Also `recipes list [words]`, `recipes show <id>`, `recipes remove <id>`. |
| `cooking-agent setup` | Answer the first-time questions (allergies, conditions, diet) by typing. The voice agent asks them on first run anyway. |
| `cooking-agent prefs` | Show what it knows about you. `prefs add diet vegetarian`, `prefs remove mushrooms`, `prefs forget-history`. |
| `cooking-agent bench` | End-to-end latency benchmark over WAV fixtures (see below). |
| `cooking-agent diagnose` | Lists audio devices, measures mic level, checks keys and LLM connectivity. |

## Providers

The agent talks to three small interfaces (`providers/base.py`), so each part
can be swapped by writing one adapter.

| Role | Default | Offline fallback |
|---|---|---|
| Speech-to-text | Deepgram Nova-3, streaming WebSocket with endpointing | Vosk (`VOSK_MODEL_PATH`) |
| LLM | Local model through Ollama (`ollama`), streaming with tool calling | Gemini (`google-genai`) with `LLM_PROVIDER=gemini` |
| Text-to-speech | Deepgram Aura-2, streaming WebSocket, kept open across turns | pyttsx3 (OS voice) |

With `STT_PROVIDER=auto` / `TTS_PROVIDER=auto` (the default) Deepgram is used
when `DEEPGRAM_API_KEY` is set and the offline providers otherwise. For the
offline ones: `pip install -e ".[offline]"`. All settings are listed in
[`.env.example`](.env.example).

## Services and packages

**Online services** (what is sent where):

| Service | Used for | Sent | Needed? |
|---|---|---|---|
| Ollama (local, `http://localhost:11434`) | Understanding what you say, choosing actions, writing recipes (default) | Nothing leaves your computer | Yes, by default (or Gemini instead) |
| Google Gemini API | Reading recipe photos/PDFs/recordings and transcribing videos when the key is set; the whole conversation with `LLM_PROVIDER=gemini` | The recipe file or transcript; with `LLM_PROVIDER=gemini` also the conversation, cooking state and your saved preferences | Optional |
| Deepgram (`wss://api.deepgram.com`) | Live speech-to-text (Nova-3) and text-to-speech (Aura-2); transcribing recipe videos | Your microphone audio while listening; the assistant's replies as text | Recommended; offline fallback exists |
| Recipe websites, YouTube, Instagram, TikTok, ... | Only when you run `recipes add <link>` | A normal page/caption request | Only for importing |

Nothing else leaves your computer. The recipe book and your preferences are a
local SQLite file (`~/.cooking-agent/recipes.db`).

**Python packages** (pinned in `requirements.txt`):

| Package | What it does here |
|---|---|
| `ollama` | Talks to the local Ollama server: streaming replies, tool calls, structured recipe output |
| `google-genai` | Google's official SDK for Gemini (optional provider and recipe import) |
| `websockets` | Talks to Deepgram's streaming speech APIs directly |
| `pydantic` | Defines and validates the recipe format and every tool's arguments, so a malformed AI reply is caught |
| `numpy` | Audio maths: loudness (for barge-in), resampling, the test tones |
| `sounddevice` | Microphone input and speaker output (PortAudio, bundled on Windows/macOS) |
| `python-dotenv` | Loads `.env` into settings |
| `yt-dlp` | Reads captions/subtitles from video sites and downloads just the audio when the recipe is only spoken |
| `vosk` *(offline extra)* | Offline speech-to-text |
| `pyttsx3` *(offline extra)* | Offline text-to-speech using your OS's voice |
| `pytest`, `pytest-asyncio` *(dev)* | Test suite |

Built-in Python modules do the rest: `asyncio` (everything runs on one event
loop), `sqlite3` (recipe book), `urllib` (fetching recipe pages), `argparse`
(the CLI).

## What it can do (LLM tools)

| Tool | Example |
|---|---|
| `get_recipe` | "Find me a recipe for jeera rice, vegan, for four." (your recipe book first) |
| `next_step` / `previous_step` / `repeat_step` | "Yes, ready." "Go back one." "Say that again." |
| `start_step_timer` | "Yes" to "Do you want me to start a timer for 12 minutes?" |
| `list_ingredients` | "What do I need?" |
| `scale_servings` | "Make it for six people." |
| `substitute_ingredient` | "I don't have ghee, what can I use?" |
| `change_recipe` | Mid-recipe: "Skip the cashews." "Add some peas." "Make it less spicy." "Double the garlic." "Fry it for 5 minutes instead." (the current step is kept; when a step mentions something you took out or swapped, the app adds "Leave out the cashews." or "Use butter instead of ghee." as it reads it; "could I leave out the garlic?" is answered, not acted on; an ingredient that matches more than one line ("the rice") gets "Which one?") |
| *(no tool)* | Questions about the dish, answered from the recipe you're cooking: "How do I know the rice is done?" "Why do we soak it?" "Can I use brown rice?" |
| `set_timer` / `cancel_timer` / `list_timers` | "Set a rice timer for 12 minutes." "How long is left?" "Stop the pasta timer." |
| `list_saved_recipes` / `save_recipe` | "What recipes do I have?" "Save this one." |
| `start_recipe_dictation` and friends | "I want to save my own recipe, Mum's dal, serves four." |
| `remember_preference` / `forget_preference` / `get_preferences` | "I'm vegetarian." "I'm allergic to peanuts." "Forget that I don't like mushrooms." |
| `end_session` | "That's all, thanks, bye." |

Recipes are generated with a pydantic schema as the LLM's structured-output
schema and validated on return (quantities, units, per-step timer hints), so
scaling and substitution work on real numbers rather than on free text.

## Cooking at your pace

The assistant never moves on by itself. After reading a step:

* if the step states a time ("simmer for 10 to 12 minutes"), it asks **"Do you
  want me to start a timer for 10 minutes?"** (a range uses the lower bound, so
  you check early). Say yes, or ask for a different length;
* then, or straight away for steps without a time, it asks **"Are you ready for
  the next step?"** and waits. "Not yet" is fine; it won't nag.

These questions are added by the app, not left to the model, so they come
every time in the same words, and the app remembers which question is open so
a plain "yes" does the right thing. Times are read from the step text when a
recipe doesn't mark them, so imported and dictated recipes get timers too.

## Your recipe book

`get_recipe` looks in your recipe book first and only asks the LLM to write a
recipe when there's no match. Saved recipes are read back exactly as stored and
scaled to however many people you're cooking for. The book is a local SQLite
file (`RECIPE_DB`, default `~/.cooking-agent/recipes.db`). Ways to fill it:

* **From a link:** `cooking-agent recipes add <link>`.
  * *Recipe web pages:* most embed the recipe in a standard machine-readable
    form (schema.org), which is taken exactly as published. Pages without it
    are read as text.
  * *YouTube (videos and Shorts), Instagram reels and posts, TikTok, Facebook,
    X, Pinterest and many more* (via [yt-dlp](https://github.com/yt-dlp/yt-dlp)):
    first the caption/description, where many creators write the full recipe;
    if it isn't there, the video's subtitles; if there are none, a small audio
    track is downloaded to a temporary folder, transcribed (Deepgram, or Gemini
    without a Deepgram key) and deleted. The video itself is never downloaded.
  * Instagram and some other sites often need you to be logged in: add
    `--cookies-from-browser chrome` (or `firefox`, `safari`, `edge`) to use the
    login from your browser, or set `YTDLP_COOKIES_FROM_BROWSER`.
* **From a file:** a photo or screenshot of a recipe (`.jpg`, `.png`,
  `.heic`...), a PDF, an audio or video recording (`.mp3`, `.m4a`, `.mp4`,
  `.mov`...), a saved web page, a pasted caption in a `.txt` file, or a recipe
  you write yourself:

  ```markdown
  # Mum's Dal
  Serves: 4

  ## Ingredients
  - 1 cup toor dal, rinsed
  - 1/2 tsp turmeric
  - Salt to taste

  ## Steps
  1. Pressure cook the dal with 3 cups of water for 15 minutes.
  2. Temper cumin in ghee and pour over.
  ```

  then `cooking-agent recipes add dal.md` (JSON in the app's recipe schema works too).
* **By voice:** "I want to save my own recipe, Mum's dal, serves four." Then
  read out ingredients and steps in any order and as many turns as you like; it
  answers "Got it." Fix mistakes ("remove step three"), ask it to read the
  draft back, and say "that's it, save it".
* **A generated recipe you liked:** "Save this recipe."

### Word for word

Imported recipes are kept and read exactly as the creator wrote or said them.

* Where a source has no fixed layout (captions, transcripts, pages without
  recipe data), Gemini finds the recipe but is only allowed to *copy*: split
  it into ingredient lines and steps, and leave out chatter (greetings,
  hashtags, "link in bio", "um"). Every line it returns is then checked in code
  against the source: the same words in the same order, nothing added or
  reworded. Reworded lines are sent back once; anything still not word for word
  is listed when you import, so you can check it. Photos and PDFs can't be
  checked this way, so you're reminded to look them over (`recipes show <id>`).
* When cooking, the app speaks each step itself ("Step 3. " followed by the
  step exactly as stored) rather than having the model read it, and ingredient
  lines are given exactly as published. Only scaling rewrites an amount
  ("1½ cups" becomes "3 cups" for double), because the original number no
  longer applies.
* If the source doesn't say how many it serves, 4 is assumed and you're told;
  use `--servings` so scaling starts from the right number.
* A recipe in another language is kept in that language; the English voice
  will not read it well.

Ingredient lines like "1½ cups basmati rice, rinsed" are parsed into quantity,
unit, name and note so scaling works; lines without an amount ("salt to
taste") are kept as written and never scaled. Search tolerates small
speech-recognition slips ("jeera rise") but not different words ("chickpea" ≠
"chicken").

## Your preferences

The very first time you start `run` or `chat`, the assistant asks: *"Do you
have any food allergies, or conditions like lactose intolerance, celiac disease
or diabetes that I should keep in mind? And do you follow a diet?"* It saves
what you say ("none" and "skip" are fine; if you don't answer, it asks again
next time). `cooking-agent setup` asks the same by typing.

The assistant keeps a small profile in the same local file and sends it to the
LLM with each turn and each new recipe:

* **What you tell it**, kept until you say otherwise: allergies, health
  conditions and intolerances, diet, dislikes, likes, what you usually have at
  home. "I'm lactose intolerant." "I'm allergic to cashews." "I'm vegetarian."
  "I always have paneer and ghee." "Forget that I don't like mushrooms." A diet
  you mention about yourself in passing ("honestly I'm vegetarian, but eggs
  are okay", "we're vegan") is saved straight away by the app itself, and the
  newest one replaces an older one.
* **What it learns**, from what you actually do: a diet you've asked for in
  most recent requests (at least 3 times), how many people you usually cook for
  (used when you don't say), ingredients that appear in at least two
  recipes you actually started cooking, and **the changes you keep making**:
  swaps ("butter instead of ghee"), things you leave out or add, more or less
  of something. Every change is saved straight away; it becomes a habit once
  you've made it in two different recipes ("onions", "onion" and "red onion"
  count as the same thing), so a one-off "no cashews today" doesn't change
  future recipes, and taking something out and putting it back in the same
  recipe cancels out. If two habits contradict, the latest wins. "Forget that
  I swap ghee for butter" drops one habit; `prefs forget-history` drops all.

When a new recipe is written, the app itself makes your usual changes to it
(swaps, things left out, more or less of something) and the assistant only
mentions the ones it actually made. For a recipe from your own book, it
offers them instead and only makes them if you say yes. A learned habit is
never applied if it clashes with an allergy, condition or stated diet, and a
learned "more" isn't applied to something a condition limits (more salt with
high blood pressure, more sugar with diabetes). A diet it has only learned (you
usually ask for vegetarian but never said so) gives a heads-up if a recipe
breaks it ("this has chicken, and you usually cook vegetarian") but never
forces a rewrite; only a diet you've stated does that.

A habit isn't applied to a dish built around that ingredient. If you usually
leave out cashews and ask for a kaju curry, the cashews stay, and the app
says so and offers better options: "Kaju Curry is built around cashews, so I
kept them in, though you usually leave them out. You could use coconut milk
instead, or make coconut curry. I'd go with coconut milk. Shall I, or keep the
cashews?" Answer "yes" (its pick), "keep them", "leave them out", "the other
one", or name an option ("coconut milk", "make the coconut curry"); anything
else, like "I don't have coconut milk" or a question about an option, goes to
the assistant, and after a question the app asks again. Taking its
suggestion isn't learned as a new habit. The options are written by the model
and checked against your allergies, conditions, diet, dislikes and other
habits. A usual swap works
the same way: for ghee rice it asks "Shall I use butter anyway, or keep the
ghee?". The dish name is checked in English and in common Indian names (kaju,
aloo, palak, gobi, jeera, murgh, and so on), and the recipe writer is told
what those words mean, because it once wrote a kaju curry with almonds. Only
the dish name counts: "pulao with cashews" isn't a cashew dish, and "more" or
"less" habits still apply.

New recipes are written to fit: allergies are always excluded, health
conditions are turned into concrete cooking rules (lactose intolerance: no
milk, cream, paneer, yogurt, lactose-free or plant versions are fine; celiac:
no wheat, maida, semolina, regular soy sauce...; diabetes: low added sugar and
refined carbs; high blood pressure: low salt), your diet is followed, dislikes
are left out, and ingredients you usually have are preferred, unless they
clash with an allergy or condition. What you ask for in the moment wins for
diet and taste ("chicken biryani, just this once"), never for allergies or
conditions. When it applied something it learned rather than something you
said, it tells you (only what it actually changed).

**Safety check.** Every recipe's ingredients are also checked in code against
your allergies and conditions, with common families (tree nuts, shellfish,
dairy, gluten...) and look-alikes that are fine (coconut milk, rice flour,
nutmeg, eggplant). A "nuts" allergy covers peanuts too (peanut oil, groundnuts,
peanut butter); "tree nuts" covers tree nuts only. Vegetarian, eggetarian, eggless, vegan and Jain diets are
checked the same way (vegetarian: no meat, fish, eggs, gelatin or meat stock;
eggetarian: vegetarian but eggs are fine, also understood as "vegetarian but I
eat eggs", "eggitarian" or "I eat eggs but no meat"; eggless: no eggs; vegan:
also no dairy, ghee or honey; Jain: vegetarian, no honey, no onion, garlic,
potato, ginger or other root vegetables), both when they're saved in your profile and when
you ask for one just this once ("vegan kheer"). For a diet, a request that
names the ingredient itself wins ("chicken biryani, just this once");
allergies and conditions are never overridden.

* a generated recipe that still contains something you can't have is sent back
  once to be rewritten; if it still conflicts, you get a spoken warning;
* recipes from your own book are never changed, but you're warned ("Heads up:
  this has milk, which conflicts with lactose intolerance");
* a substitution, or an ingredient added or increased mid-recipe, into
  something you can't have is flagged.

This is a helper, not a medical guarantee: word matching can't see hidden
ingredients in branded products, and the guidance for conditions like
diabetes is general. Always check labels.

`cooking-agent prefs` shows everything it knows; `prefs forget-history` clears
what it learned.

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

Many fixtures are everyday commands that the app answers itself without the
LLM (instant commands). To measure the **model**, add `--no-instant` (every
fixture goes to the LLM) and `--text-stt` (feed each fixture's text instead of
recognising its audio, so speech-recognition errors don't blur the result).

### Diet test

`cooking-agent diet-eval` asks the LLM for ten recipes where the rule must
hold but the dish name doesn't give it away (vegetarian pad thai, caesar salad
and ramen; vegan kheer and chocolate cake; Jain dal tadka and pav bhaji;
lactose-intolerant paneer butter masala; celiac pasta; peanut-allergy satay).
Each goes through the normal path (profile in the prompt, the code check, one
rewrite if needed) and passes if no conflict is left. `--out FILE` keeps the
recipes for reading.

### Understanding test

`cooking-agent understanding-eval` checks what the assistant does with 28
things a cook might say on step 2 of a pulao or a tomato spaghetti, right
after "Are you ready for the next step?": questions about the dish ("would
brown rice work here?", "could I leave out the garlic?"), changes ("drop the
cashews", "can you make it milder", "use half a cup less water"), loose
commands ("alright, the cashews look golden", "we're cooking for four now")
and chit-chat. A question must get a spoken answer (not silence or an error
message) with the step and recipe untouched; a change must really change the
recipe, including what the steps will say, without moving the step; a tool
that returned an error doesn't count. It exits with an error below 90%
(`--min-accuracy`). Each case runs 3 times
(`--runs`), with instant commands on as in real use, on a throwaway recipe
book and profile. `--out FILE` keeps every reply.

### Choosing the local model

Measured 2026-11-16 on an RTX 4060 Laptop GPU (8 GB), Ollama 0.40.16, one
model loaded at a time, all on the GPU, `bench --text-stt --no-instant
--runs 3`, `diet-eval` and `understanding-eval --runs 3`.

`qwen3:8b` (the default) after the Phase 2 review fixes: **tool routing 100%**
(30/30), **understanding test 95%** (80/84 over the 28 cases), **diet test
10/10** (2 rewrites), LLM first token p50 / p90 1.1 s / 2.0 s.

The model choice, before the review fixes (20-case understanding test):

| Model | Tool routing | Understanding test | LLM first token p50 / p90 | Diet test |
|---|---|---|---|---|
| **`qwen3:8b`** | **97%** | **100%** (60/60) | 1.0 s / 1.7 s | **10/10** (1 rewrite) |
| `gemma4:e4b` | 67% (answered in words instead of calling `get_recipe`, `next_step`, `substitute_ingredient`, `set_timer` in 10 of 30 turns) | 98% (one "milder" call had a bad argument) | **0.7 s** / 2.4 s | 9/10 (left a "Worcestershire sauce substitute" in a vegetarian caesar salad) |

Phase 1 (first prompt, same benchmark): `qwen3:8b` 90% routing, 10/10 diets;
`llama3.1:8b` 90%, 1.5 s first token, 10/10; `mistral:7b` 10% (almost never
calls tools).

`qwen3:8b` stays the default: Gemma 4 is a bit faster to start talking but
misses a third of the tool calls, which is what moves the recipe and timers.
Everyday commands don't depend on the model at all (instant commands).

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
  state.py        cooking-session and turn state machines, open question, dictation draft
  tools.py        tool schemas (pydantic), dispatcher, follow-up questions after steps
  recipe.py       recipe schema; parsing quantities, durations and ingredient lines
  recipe_store.py recipe book (SQLite) and search
  recipe_import.py  import from any link or file; word-for-word extraction check
  media_import.py   captions/subtitles/audio via yt-dlp; transcription
  preferences.py  stated preferences, learned habits, first-run setup flag
  dietary.py      ingredient checks against allergies and health conditions
  timers.py       concurrent named asyncio timers
  segmenter.py    split streaming text into speakable sentences
  audio.py        mic / WAV sources, speaker sinks, MicGate
  metrics.py      per-turn latency records
  bench.py        latency benchmark
  config.py       env-driven provider selection
  providers/      gemini, deepgram, local (vosk, pyttsx3), fake
```
