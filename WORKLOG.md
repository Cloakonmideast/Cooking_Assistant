# Work log

A record of every change made to this project folder, newest first. Each entry
says what was done, why, and how it was checked.

---

## 2026-11-15 — Plan for a local AI model written down

* Decided direction: **keep it simple and local** (no website/phone app,
  no containers, no paid services for now).
* New branch `local-model` (from `windows-py311-fixes`) for this work.
* Plan saved to `docs/superpowers/plans/2026-11-15-local-model.md` and kept up
  to date as it changes (Status table + Plan history at the bottom). Five
  tasks: Ollama adapter → instant offline commands → wire them into the agent
  with an offline fallback → fewer tools per turn → pick the model by
  measuring.
* Advisor model: checked that Claude Code's advisor is already set to Fable
  (`"advisorModel": "fable"` in the user-level `~/.claude/settings.json`,
  applies to every project). No setting changed.
* No code changed yet.

---

## 2026-11-15 — First manual voice test: it never replied (Gemini overloaded and rate-limited)

### What happened

`cooking-agent run` started fine, asked the first-run question out loud, and
Vosk heard the answers ("hello", "i don't have any dietary restrictions i only
have that i'm vegetarian"). But **no turn ever got a reply**:

| Turn | Heard | What happened |
|---|---|---|
| 1 | "i know i know" | Gemini returned **503 UNAVAILABLE: "This model is currently experiencing high demand"** after ~5 s. The spoken apology was cut off by the next utterance. |
| 2 | "hello" | No reply after 16 s; cancelled when the cook spoke again. |
| 3 | "i don't have any dietary restrictions…" | No reply after 23 s; session ended. |

### Cause (measured)

**Correction to the earlier entry:** the 20–47 s replies were *not* the
network. The default model `gemini-3.8-flash` is overloaded on Google's side.
Measured with the same key, two calls each:

| Model | Result |
|---|---|
| `gemini-3.8-flash` (the app's default) | 133 s, then 503 |
| `gemini-flash-latest` | 187 s, then 503 |
| `gemini-3.7-flash` | 429 (over free quota) |
| `gemini-3.5-flash` | **0.8–1.6 s** |
| `gemini-3.5-flash-lite`, `gemini-flash-lite-latest` | 0.6–1.1 s |

A second problem showed up straight away with `gemini-3.5-flash`: the
**free tier allows 5 requests per minute per model**, and one spoken turn uses
1–3 requests (a tool call and its follow-up; `get_recipe` adds one more). A
`chat` test did setup, wrote a recipe and listed ingredients correctly (first
word in 1.6–3.4 s), then the next three turns failed with
**429 RESOURCE_EXHAUSTED … limit: 5 … Please retry in 45s**.

Two app weaknesses made this worse:

* Any Gemini error gives the same "Sorry, something went wrong" line, no retry,
  no other model tried.
* While a reply is still being prepared, anything the mic hears cancels it, so
  during a slow reply the cook repeating themselves restarts the wait.

### What was changed

* `.env`: added `GEMINI_MODEL=gemini-3.5-flash` (local only, not committed).

No code changed yet.

---

## 2026-11-15 — Tested the real app; offline voice set up; README setup guide

Committed on branch `windows-py311-fixes` (README.md and this log only).

### 1. Tested every way of running it

| Check | Result |
|---|---|
| `cooking-agent demo` | Works (scripted, offline). |
| `cooking-agent diagnose --mic-seconds 0` (before) | Gemini key works; audio devices found (27, incl. both Bluetooth headsets). **FAIL** speech-to-text: no `DEEPGRAM_API_KEY` and no Vosk model. Speech output showed "ok" but pyttsx3 was not actually installed (the check only builds the provider object, it doesn't import pyttsx3). |
| `cooking-agent chat` with the real Gemini, using a throw-away recipe database so the real profile was untouched | Works: asked the first-run allergy question, wrote a new jeera rice recipe for two, read step 1 word for word, asked "ready for the next step?". Time to first word: 20–33 s. |
| `cooking-agent diagnose --mic-seconds 0` (after step 2 below) | All `[ok]`: audio, stt=vosk, tts=pyttsx3, llm=gemini. |
| `cooking-agent bench --only next_step set_timer list_timers` (recorded voice clips through Vosk → Gemini → pyttsx3, as if from the mic) | 2 of 3 completed with the right action (100 % tool routing on those). `set_timer` timed out after 90 s. |

**Why `set_timer` failed.** Ran Vosk on that clip alone: it *did* hear it,
but as "a lifetime or for twelve minutes" (the clip says "set a rice timer for
twelve minutes"). So the turn was waiting on Gemini, not stuck in speech
recognition as the benchmark's message guesses. The clips are a robotic
synthetic voice (espeak-ng), which the small offline Vosk model handles
poorly; a real voice, and Deepgram, do better.

**Slowness.** Gemini took 6–47 s to start replying in these runs (normally
about 1 s). pip downloads from PyPI also kept timing out, so this is the
network connection, not the app.

**Not tested:** `cooking-agent run` with a live microphone. It needs a person
speaking. Everything it uses was checked above.

### 2. Set up offline voice (free, no account)

* Installed the offline extras into `.venv`: `vosk==0.3.45`, `pyttsx3==2.99`.
* Downloaded and unzipped the Vosk model `vosk-model-small-en-us-0.15`
  (~40 MB) to `~/.cooking-agent/models/` (outside the project).
* Appended `VOSK_MODEL_PATH=...` pointing at it to `.env` (the existing key
  was not touched; `.env` is not committed).

For better speech, add a `DEEPGRAM_API_KEY` to `.env`; the app then uses
Deepgram automatically instead of Vosk/pyttsx3.

### 3. README: setup and running guide

* Replaced "Quick start" (bash-only) with **Setup**: Windows PowerShell and
  macOS/Linux steps, where to get each API key, and the offline Vosk option.
* Added **Running it**: demo → diagnose → chat → run, with what each needs and
  what you should see; first-run question; headphones tip.
* Added **Troubleshooting** for the problems actually hit here (numpy/Python
  version, `cooking-agent` not found, `VOSK_MODEL_PATH` FAIL, slow replies,
  echo).
* Added **Services and packages**: which online services are used, what is
  sent to each, and what every Python package is for.
* The old command table is now under **All commands** (unchanged).

---

## 2026-11-15 — Make the project install and pass its tests on Windows with Python 3.11

Committed on branch `windows-py311-fixes` (not pushed; `main` is untouched).
To bring it into `main`: `git checkout main` then `git merge windows-py311-fixes`.

### 1. Fixed a flaky test on Windows

**File:** `tests/test_agent.py` — `test_timer_announcement_is_spliced_into_active_reply`

**Problem.** The test failed about 4 runs in 10 on Windows with `assert 0 < 0`:
the timer announcement was spoken *before* the reply's first sentence instead
of in the middle of the reply.

**Cause.** A timing race. The test starts a 0.15 s timer and has the fake LLM
write a 22-word reply at 0.02 s per word. It expects the timer to go off after
the first sentence (5 words) is ready. On Linux/macOS 5 words take ~100 ms, so
that holds. On Windows the timer tick is coarse and `asyncio.sleep(0.02)`
really takes ~30 ms; measured first-sentence times were 133–167 ms, on both
sides of 150 ms, so whichever came first won.

**Fix.** Timer changed from 0.15 s to 0.3 s, with a comment explaining the
numbers. 0.3 s is after the first sentence on every platform (≤ ~170 ms) and
well before the reply ends (~440 ms on Linux, ~700 ms on Windows), so the test
still checks the same behaviour: an announcement spliced in mid-reply.

**Checked.** The test passed 30/30 runs; the full suite passed 3 runs in a row.

### 2. Fixed two tests that crashed on Windows (text encoding)

**File:** `tests/test_media_import.py`, lines 180 and 188
(`test_pasted_caption_in_a_text_file`,
`test_page_without_recipe_data_falls_back_to_its_text`)

**Problem.** Both failed with `UnicodeEncodeError: 'charmap' codec can't encode characters`.

**Cause.** The sample caption contains emoji (🔥🍚). The tests saved it with
`write_text(...)` without an encoding, so Python used the operating system's
default. On Windows that is cp1252, which has no emoji, so saving the file
crashed before the app code ran. (On Linux/macOS the default is UTF-8, which is
why they pass there.)

**Fix.** Added `encoding="utf-8"` to both `write_text(...)` calls.

**Not affected:** the app itself. `recipe_import.py` already reads files with
`encoding="utf-8"`, so importing real recipe files with emoji works on Windows.

### 3. Fixed the numpy version so it installs on Python 3.11

**Files:** `requirements.txt`, `pyproject.toml`

**Problem.** `pip install -r requirements.txt` failed with
`No matching distribution found for numpy==2.5.4`.

**Cause.** numpy 2.5.x requires Python 3.12+, but the project says it supports
3.11 (`requires-python = ">=3.11"`) and this machine has Python 3.11.

**Fix.** `numpy==2.5.4` → `numpy>=2.4.6,<2.6`. On Python 3.11 pip installs
2.4.6 (the newest numpy for 3.11); on 3.12+ it still gets 2.5.x.

**Checked.** `pip install -r requirements.txt` and `pip install -e .` succeed;
`pip check` reports no broken requirements.

### 4. Created the virtual environment

`.venv/` created with Python 3.11 and the runtime + dev dependencies installed
(`.venv` is git-ignored). Activate it in PowerShell with
`.venv\Scripts\Activate.ps1`.

**Checked.** `cooking-agent demo` runs the whole scripted session offline
(step pacing, timer offer and announcement, barge-in, dictating and saving a
recipe, allergy warning, scaling for four).

### 5. Cleaned up the folder to just the latest code

**Starting point.** Local `main` was already at the latest commit on GitHub
(`4b9f8b5`, the merge of PR #1 `realtime-upgrade`); nothing new to pull.

**Problem.** Nine older files were *staged* (about to be committed) but were
not part of the project: the old single-file assistant and scratch scripts,
plus `.env`, which holds the API key and must never be committed.

**What was done.**

* Unstaged everything (`git reset`), so nothing stray gets committed.
* Kept `.env` in place (the app needs the key); it is already in `.gitignore`.
* Moved (not deleted) these to `../python_old_local_files/`, next to this folder:

  | File | What it was |
  |---|---|
  | `cooking_assistant.py`, `demo.py` | the old version of this assistant, replaced by `src/cooking_agent/` |
  | `tes.py` | printed the full API key to the screen |
  | `test_env.py`, `test_mic.py` | scratch checks for the key and the microphone |
  | `Untitled-1.cpp`, `.vscode/tasks.json` | an unrelated C++ hello-world and its build task |
  | `callcenter.xml` | an unrelated Salesforce sample file |
  | `venv/` | an old virtual environment made at a different path, so it no longer worked |

  That folder can be deleted once you're sure nothing in it is needed.

**Result.** The folder contains exactly what is on GitHub, plus `.env`, `.venv/`
and this log.
