# Local AI Cooking Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The assistant runs on the laptop with a local AI model; everyday cooking commands are instant and work offline; Gemini becomes optional.

**Architecture:** A new `OllamaLLM` adapter speaks the existing `LLMProvider` interface (`providers/base.py`), so the agent, tools, safety checks, recipe book and timers are unchanged. A phrase matcher runs before any AI call and dispatches common commands straight to the existing tools, which makes them instant and offline. Each turn offers the model only the tools that fit the current phase.

**Tech Stack:** Python 3.11, `ollama` (official Python client, talks to `http://localhost:11434`), pydantic, pytest.

**Spec:** Agreed in conversation on 2026-11-15: "keep it simple and local"; local-model plan steps 1–4 and 7. Out of scope: website/phone app, containers, Piper voice, web recipe search, photo/PDF import without Gemini, other languages.

## Status

| Task | State |
|---|---|
| 1. Ollama adapter and settings | not started |
| 2. Instant command matcher | not started |
| 3. Wire matcher into agent + offline fallback | not started |
| 4. Phase-specific tool lists | not started |
| 5. Choose the model by measuring | not started |

**Execution approach:** not chosen yet (recommended: Native, since the 5 tasks are a straight line and cheap to verify).
**Branch:** `local-model` (from `windows-py311-fixes`).

## Global Constraints

- Python 3.11 on Windows; every test passes with no network, no Ollama, no audio hardware.
- No new paid services. Gemini stays available with `LLM_PROVIDER=gemini`.
- Default becomes `LLM_PROVIDER=ollama` (planner's call).
- English only.
- Every task ends with a `WORKLOG.md` entry, an update to this plan's Status table, and a commit.

## Review Focus

1. Ollama not running or model not downloaded: the cook hears `LLM_UNAVAILABLE` ("I can't reach the AI model right now…"), never a crash or the generic "sorry". *(Task 3: `test_llm_down_speaks_offline_message`)*
2. Longer sentences containing a command word ("next time make it spicier", "what goes with the rice timer") go to the AI, not to an instant action. *(Task 2: `test_longer_sentences_fall_through`)*
3. Vosk writes numbers as words: "twelve minutes", "a minute", "one and a half minutes", "thirty seconds". *(Task 2: `test_number_words`)*
4. During first-run setup or dictation, "no" and "next" mean something else; only timer commands are matched then. *(Task 2: `test_setup_and_dictation_only_timers`)*
5. "Stop" with one timer cancels it; with two or more it goes to the AI; it never ends the session. *(Task 2: `test_stop_rules`)*

---

### Task 1: Ollama adapter and settings

**Files:**
- Create: `src/cooking_agent/providers/ollama.py`
- Modify: `src/cooking_agent/config.py` (`Settings`, `make_llm`), `src/cooking_agent/media_import.py` (`make_transcriber`)
- Modify: `requirements.txt`, `pyproject.toml` (add `ollama`, pinned to the version pip installs), `.env.example`, `README.md` (Setup: "Local AI with Ollama")
- Test: `tests/test_ollama_adapter.py`

**Interfaces:**
- Consumes: `LLMProvider`, `Message`, `ToolSpec`, `ToolCall`, `TextDelta`, `ToolCallEvent`, `TurnComplete`, `MediaPart` from `providers/base.py`.
- Produces: `class OllamaLLM` with `name = "ollama"`, `__init__(self, model: str, url: str = "http://localhost:11434", client=None, temperature: float = 0.4)`; `stream(system, history, tools) -> AsyncIterator[LLMEvent]`; `generate_structured(prompt, schema, media=None) -> M`.
- Produces: `Settings.ollama_model: str = "qwen3:8b"` (env `OLLAMA_MODEL`), `Settings.ollama_url: str = "http://localhost:11434"` (env `OLLAMA_URL`); `Settings.llm` default `"ollama"`.

- [ ] **Step 1: Write the failing tests** (fake `AsyncClient` that records `chat()` kwargs and yields scripted chunks)
  - `test_history_conversion`: system first; user → `{"role": "user", "content": ...}`; assistant with calls → `{"role": "assistant", "content": ..., "tool_calls": [{"function": {"name": "next_step", "arguments": {}}}]}`; each tool result → `{"role": "tool", "tool_name": "next_step", "content": json.dumps(result)}`.
  - `test_stream_text_then_tool_call`: chunks "Sure, " / "one sec." / tool call → events `TextDelta, TextDelta, ToolCallEvent(ToolCall("set_timer", {...})), TurnComplete`; `TurnComplete.message.tool_calls` filled.
  - `test_chat_called_with_think_false_and_keep_alive`: `stream=True`, `think=False`, `keep_alive="30m"`, `tools == [{"type": "function", "function": {"name", "description", "parameters"}}]`.
  - `test_generate_structured_retries_once`: invalid JSON then valid → `Recipe`; `format == Recipe.model_json_schema()`.
  - `test_generate_structured_media_rejected`: `media=[...]` → `RuntimeError` mentioning `GEMINI_API_KEY`.
  - `test_make_llm_ollama`: `make_llm(Settings(llm="ollama"))` is an `OllamaLLM`.
  - `test_make_transcriber_none_for_ollama`: no `DEEPGRAM_API_KEY` and a non-Gemini LLM → `None`.
- [ ] **Step 2: Run** `pytest tests/test_ollama_adapter.py -v` — Expected: FAIL (no module `cooking_agent.providers.ollama`).
- [ ] **Step 3: Implement** `OllamaLLM` (`await client.chat(model=..., messages=..., tools=..., stream=True, think=False, keep_alive="30m", options={"temperature": ...})`; tool-call ids `None`); wire `make_llm`; `make_transcriber` builds `GeminiTranscriber` only when `llm.name == "gemini"`.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** README section (`winget install Ollama.Ollama`, `ollama pull <model>`), `.env.example`, WORKLOG entry, Status table; commit `feat: local LLM via Ollama`.

### Task 2: Instant command matcher

**Files:**
- Create: `src/cooking_agent/commands.py`
- Modify: `src/cooking_agent/recipe.py` (`parse_duration_minutes` learns number words, only if the test shows it doesn't already)
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: `CookingSession`, `Awaiting` (`state.py`); `RecipeStore.find` (`recipe_store.py`); `ToolCall`, `ToolResult` (`providers/base.py`); `parse_duration_minutes`, `format_minutes` (`recipe.py`).
- Produces: `@dataclass CommandMatch(calls: list[ToolCall], reply: str | None = None)`.
- Produces: `match_command(text: str, session: CookingSession, store: RecipeStore | None, active_timers: list[str]) -> CommandMatch | None` (pure: decides, never runs).
- Produces: `speak_result(results: list[ToolResult]) -> str | None` with these exact phrasings:
  - `set_timer` → "Okay, rice timer set for 12 minutes."
  - `cancel_timer` → "Rice timer cancelled."
  - `list_timers` → "Rice: 4 minutes left. Pasta: 30 seconds left." / "No timers are running."
  - `list_ingredients`, `scale_servings` → first 6 lines verbatim, then "…and 3 more. Want the rest?"
  - `get_recipe` → "Found Mum's Dal in your recipe book, for 4. Say ready to start."
  - any result with `"error"` → the error text unchanged.

**Matching rules** (lower-case, strip punctuation and fillers "please"/"okay"/"um", then the **whole** utterance must match; checked in this order):

| Says | Calls | Only when |
|---|---|---|
| next, next step, ready, done, continue | `next_step` | recipe loaded; no setup/dictation open |
| yes / yeah / sure / go ahead | `start_step_timer` or `next_step` | `session.awaiting` is TIMER / NEXT_STEP |
| no / not yet / wait | none; reply "No problem, take your time." | `session.awaiting` set |
| back, go back, previous step | `previous_step` | recipe loaded |
| repeat, say that again, what was that | `repeat_step` | recipe loaded |
| set a [name] timer for N minutes/seconds/hours | `set_timer` (name default "kitchen") | always |
| stop/cancel the [name] timer; bare "stop" with exactly 1 timer | `cancel_timer` | always |
| how long is left, what timers | `list_timers` | always |
| what do I need, ingredients | `list_ingredients` | recipe loaded |
| make it / scale it for N (people/servings) | `scale_servings` | recipe loaded |
| cook / make [my] X | `get_recipe` | only if `store.find(X)` hits |
| bye, goodbye, that's all | `end_session` | always |

During setup or dictation only the three timer rows apply.

- [ ] **Step 1: Write the failing tests** — one per table row, plus:
  - `test_number_words`: "set a rice timer for twelve minutes" → `set_timer {"name": "rice", "minutes": 12}`; "a minute" → 1; "one and a half minutes" → 1.5; "thirty seconds" → 0.5.
  - `test_longer_sentences_fall_through`: "next time make it spicier", "I don't have ghee what can I use", "what goes with the rice timer" → `None`.
  - `test_setup_and_dictation_only_timers`: `setup_pending=True`: "no" → `None`, "next" → `None`, "set a pasta timer for 5 minutes" matches; same with a draft open.
  - `test_stop_rules`: "stop" with `["rice"]` → `cancel_timer rice`; `["rice", "pasta"]` → `None`; `[]` → `None`; never `end_session`.
  - `test_cook_saved_only`: "cook my dal" with "Mum's Dal" saved → `get_recipe {"dish": "dal"}`; empty store → `None`.
  - `test_speak_result_templates`: the exact strings above.
- [ ] **Step 2: Run** `pytest tests/test_commands.py -v` — Expected: FAIL (no module `cooking_agent.commands`).
- [ ] **Step 3: Implement** with anchored regexes; durations through `parse_duration_minutes`, extended to number words one–sixty, "a", "half an hour", "X and a half".
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** WORKLOG entry, Status table; commit `feat: instant offline commands`.

### Task 3: Wire the matcher into the agent + offline fallback

**Files:**
- Modify: `src/cooking_agent/agent.py` (`_think` and its LLM error path)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `match_command`, `speak_result` (Task 2); existing `CookingAgent._step_speech`, `_ask_follow_up`, `ToolBox.dispatch`.
- Produces: `LLM_UNAVAILABLE = "I can't reach the AI model right now. I can still go to the next step, repeat it, set timers, and cook recipes from your recipe book."`

**Behaviour:**
- At the start of `_think`, call `match_command`. On a match: dispatch the calls in order; speak `_step_speech(results)` or else `speak_result(results)` or else `match.reply`; then `_ask_follow_up`; never call the LLM.
- History for an instant turn is text only: `Message("user", text)`, `Message("assistant", spoken)`, with no `tool_calls` (Gemini's newer models reject historical function calls that lack thought signatures).
- Any exception from `self.llm.stream` → speak and `on_text` `LLM_UNAVAILABLE` (replaces the generic line; also makes `chat` show it).

- [ ] **Step 1: Write the failing tests** (`make_agent` fixture; an LLM that fails the test if called):
  - `test_instant_next_step_skips_llm`: recipe loaded, `submit_text("next")` → speaks "Step 1. …" plus the follow-up; `llm.calls == []`.
  - `test_instant_timer`: "set a rice timer for twelve minutes" → speaks "Okay, rice timer set for 12 minutes."; timer exists.
  - `test_unmatched_goes_to_llm`: "what can I use instead of ghee" → LLM called.
  - `test_llm_down_speaks_offline_message`: `stream` raises `ConnectionError` → `tts.spoken[-1] == LLM_UNAVAILABLE` and `on_text` received it.
  - `test_history_after_instant_turn_is_text_only`: last two history entries are user, assistant; assistant has no `tool_calls`.
  - `test_saved_recipe_offline_end_to_end`: failing LLM + store with a recipe: "cook my dal" → "next" → "yes" → "next" all succeed.
- [ ] **Step 2: Run** `pytest tests/test_agent.py -v` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** in `CookingAgent._think`.
- [ ] **Step 4: Run** `pytest -q` and `cooking-agent demo` — Expected: all pass; demo runs to the end.
- [ ] **Step 5:** WORKLOG entry, Status table; commit `feat: instant commands in agent; offline fallback message`.

### Task 4: Phase-specific tool lists

**Files:**
- Modify: `src/cooking_agent/tools.py` (`ToolBox.specs_for`), `src/cooking_agent/agent.py` (`_think` uses it)
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `ToolBox.specs_for(session: CookingSession) -> list[ToolSpec]`, in registry order (stable prefix between turns).

**Rules:**
- Always (10): `get_recipe`, `list_saved_recipes`, `start_recipe_dictation`, `set_timer`, `cancel_timer`, `list_timers`, `remember_preference`, `forget_preference`, `get_preferences`, `end_session`.
- + setup pending: `finish_setup`.
- + dictating: `add_to_dictated_recipe`, `remove_from_dictated_recipe`, `read_dictated_recipe`, `save_dictated_recipe`, `discard_dictated_recipe`.
- + recipe loaded: `next_step`, `previous_step`, `repeat_step`, `start_step_timer`, `list_ingredients`, `scale_servings`, `substitute_ingredient`, `save_recipe`.

- [ ] **Step 1: Write the failing tests:** `test_specs_no_recipe` (exactly the 10), `test_specs_cooking` (18, has `next_step`, no `finish_setup`), `test_specs_setup` (has `finish_setup`), `test_specs_dictating` (has the 5), `test_specs_order_stable` (same state → identical lists).
- [ ] **Step 2: Run** `pytest tests/test_tools.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement**; `_think` calls `self.tools.specs_for(self.session)` instead of `self.tools.specs()`.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** WORKLOG entry, Status table; commit `perf: phase-specific tool lists`.

### Task 5: Choose the model by measuring

**Files:**
- Modify: `src/cooking_agent/bench.py`, `src/cooking_agent/cli.py` (`bench --text-stt`), `src/cooking_agent/config.py` (final `ollama_model` default), `.env.example`, `README.md`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `FixtureSTT`, `_make_providers` (`bench.py`).
- Produces: `_make_providers(fake: bool, fx: Fixture, speech_end_s: float, text_stt: bool = False)`; `text_stt=True` returns `FixtureSTT` with the configured real LLM and TTS (isolates the model from Vosk mishearing the synthetic voice).

- [ ] **Step 1: Write the failing test** `test_text_stt_flag_uses_fixture_stt`; run (FAIL), implement, run `pytest -q` (PASS), commit `feat: bench --text-stt`.
- [ ] **Step 2: Approval gate.** Ask the user before: `winget install Ollama.Ollama` and pulling 3 candidates (~5 GB each, ~15 GB total): the current tool-capable 7–9B models in Ollama's library, e.g. `qwen3:8b`, `llama3.1:8b`, plus one current Gemma or Mistral.
- [ ] **Step 3: Routing and speed:** for each model, `LLM_PROVIDER=ollama OLLAMA_MODEL=<m> cooking-agent bench --text-stt --runs 3`; record tool routing accuracy and first-word time.
- [ ] **Step 4: Recipe check:** for each model, `chat` against a throwaway `RECIPE_DB` with profile vegetarian + lactose intolerant, ask "paneer butter masala for two". Pass: validates as `Recipe`, no dairy conflict after the safety check, sensible steps.
- [ ] **Step 5: Pick** highest routing accuracy with first word under ~1.5 s; set the default in `config.py` and `.env.example`; `ollama rm` the others; results table in README and WORKLOG; Status table; commit `chore: choose default local model`.
- [ ] **Step 6:** Hand back to the user for a manual `cooking-agent run` test (same script as the first test).

---

## Plan history

- **2026-11-15** — First version, written from the agreed "simple and local" scope.
