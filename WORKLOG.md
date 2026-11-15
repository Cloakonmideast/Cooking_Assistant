# Work log

A record of every change made to this project folder, newest first. Each entry
says what was done, why, and how it was checked.

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
