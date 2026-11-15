# Local AI Cooking Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The assistant runs on the laptop with a local AI model; everyday cooking commands are instant and work offline; Gemini becomes optional.

**Architecture:** A new `OllamaLLM` adapter speaks the existing `LLMProvider` interface (`providers/base.py`), so the agent, tools, safety checks, recipe book and timers are unchanged. A phrase matcher runs before any AI call and dispatches common commands straight to the existing tools, which makes them instant and offline. Each turn offers the model only the tools that fit the current phase.

**Tech Stack:** Python 3.11, `ollama` (official Python client, talks to `http://localhost:11434`), pydantic, pytest.

**Spec:** Agreed in conversation on 2026-11-15: "keep it simple and local"; local-model plan steps 1–4 and 7. Out of scope: website/phone app, containers, Piper voice, web recipe search, other languages. Recipe import from social media: **YouTube Shorts only for now** (Instagram set aside, decided 2026-11-16).

## Status

| Task | State |
|---|---|
| 1. Ollama adapter and settings | done |
| 2. Smoke test against a real Ollama (approval gate) | not started |
| 3. Instant command matcher | done |
| 4. Wire matcher into agent + offline fallback | not started |
| 5. Phase-specific tool lists | not started |
| 6. Diet check in code (vegetarian, vegan, Jain) | not started |
| 7. Choose the model by measuring (routing, speed, diet test) | not started |

**Execution approach:** Native: Opus 5.5 builds every task in the main session, Fable (the configured `advisorModel`) advises at decision points, and a fresh Fable reviewer subagent reviews the whole branch at the end. Its findings are fixed and logged before handing back to the user.
**Branch:** `local-model` (from `windows-py311-fixes`).

## Global Constraints

- Python 3.11 on Windows; every test passes with no network, no Ollama, no audio hardware.
- No new paid services. Gemini stays available with `LLM_PROVIDER=gemini`; photo/PDF/video import keeps using Gemini when `GEMINI_API_KEY` is set.
- Default becomes `LLM_PROVIDER=ollama` (planner's call).
- Ollama context window set explicitly: `num_ctx = 8192` (prompt + tools alone are ~3,200 tokens; a smaller default window would cut the prompt silently).
- English only.
- Every task ends with a `WORKLOG.md` entry, an update to this plan's Status table, and a commit.

## Review Focus

1. **Ollama not running, model missing, or context too small:** the cook hears `LLM_UNAVAILABLE`, never a crash; `num_ctx` is always sent. *(Task 1: `test_chat_called_with_options`; Task 4: `test_llm_down_speaks_offline_message`)*
2. **Longer sentences containing a command word** ("next time make it spicier", "what goes with the rice timer") go to the AI. *(Task 3: `test_longer_sentences_fall_through`)*
3. **Vosk writes numbers as words:** "twelve minutes", "a minute", "one and a half minutes", "thirty seconds". *(Task 3: `test_number_words`)*
4. **A stale "yes":** a bare yes/no is instant only when the app's own follow-up question was the last thing spoken; otherwise it goes to the AI. *(Task 4: `test_stale_yes_goes_to_llm`)*
5. **Commands right after interrupting** ("go back" after talking over the assistant) still match, because matching uses the raw words, not the text with the "you interrupted" note. *(Task 4: `test_command_after_barge_in_is_instant`)*

---

### Task 1: Ollama adapter and settings

**Files:**
- Create: `src/cooking_agent/providers/ollama.py`
- Modify: `src/cooking_agent/config.py` (`Settings`, `make_llm`, new `make_import_llm`), `src/cooking_agent/cli.py` (`_recipes` uses `make_import_llm`), `src/cooking_agent/media_import.py` (`make_transcriber`)
- Modify: `requirements.txt`, `pyproject.toml` (add `ollama`, pinned to the version pip installs), `.env.example`, `README.md` (Setup: "Local AI with Ollama")
- Test: `tests/test_ollama_adapter.py`

**Interfaces:**
- Consumes: `LLMProvider`, `Message`, `ToolSpec`, `ToolCall`, `TextDelta`, `ToolCallEvent`, `TurnComplete`, `MediaPart` (`providers/base.py`); `RecipeImportError` (`recipe_import.py`).
- Produces: `class OllamaLLM` (`name = "ollama"`), `__init__(self, model: str, url: str = "http://localhost:11434", client=None, temperature: float = 0.4, num_ctx: int = 8192)`; `stream(system, history, tools) -> AsyncIterator[LLMEvent]`; `generate_structured(prompt, schema, media=None) -> M`.
- Produces: `Settings.ollama_model: str = "qwen3:8b"` (env `OLLAMA_MODEL`), `Settings.ollama_url` (env `OLLAMA_URL`), `Settings.llm` default `"ollama"`.
- Produces: `make_import_llm(s: Settings) -> LLMProvider | None` — `GeminiLLM` when `GEMINI_API_KEY` is set (photos, PDFs, video transcription keep working), else `make_llm(s)`, else `None`.

- [ ] **Step 1: Write the failing tests** (fake `AsyncClient` recording `chat()` kwargs and yielding scripted chunks)
  - `test_history_conversion`: system first; user → `{"role": "user", "content": ...}`; assistant with calls → `{"role": "assistant", "content": ..., "tool_calls": [{"function": {"name": "next_step", "arguments": {}}}]}`; each tool result → `{"role": "tool", "tool_name": "next_step", "content": json.dumps(result)}`.
  - `test_stream_text_then_tool_call`: chunks "Sure, " / "one sec." / tool call → `TextDelta, TextDelta, ToolCallEvent(ToolCall("set_timer", {...})), TurnComplete` with `message.tool_calls` filled.
  - `test_chat_called_with_options`: `stream=True`, `think=False`, `keep_alive="30m"`, `options["num_ctx"] == 8192`, `options["temperature"] == 0.4`, `tools == [{"type": "function", "function": {"name", "description", "parameters"}}]`.
  - `test_generate_structured_retries_once`: invalid JSON then valid → `Recipe`; `format == Recipe.model_json_schema()`; `options["num_ctx"] == 8192`.
  - `test_generate_structured_media_rejected`: `media=[...]` → `RecipeImportError` whose message mentions `GEMINI_API_KEY` (caught by `cooking-agent recipes add`, no traceback).
  - `test_make_llm_ollama`: `make_llm(Settings(llm="ollama"))` is an `OllamaLLM`.
  - `test_make_import_llm_prefers_gemini_when_key_set` / `test_make_import_llm_ollama_without_key`.
  - `test_make_transcriber_none_for_ollama`: no `DEEPGRAM_API_KEY` and a non-Gemini LLM → `None`.
- [ ] **Step 2: Run** `pytest tests/test_ollama_adapter.py -v` — Expected: FAIL (no module `cooking_agent.providers.ollama`).
- [ ] **Step 3: Implement** `OllamaLLM` (`await client.chat(model=..., messages=..., tools=..., stream=True, think=False, keep_alive="30m", options={"temperature": ..., "num_ctx": ...})`; tool-call ids `None`); wire `make_llm`, `make_import_llm`, `_recipes`; `make_transcriber` builds `GeminiTranscriber` only when `llm.name == "gemini"`.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** README section (`winget install Ollama.Ollama`, `ollama pull <model>`), `.env.example`, WORKLOG, Status; commit `feat: local LLM via Ollama`.

### Task 2: Smoke test against a real Ollama

Task 1's tests use a fake client whose message shapes come from documentation; this task checks them against the real server before anything is built on top.

**Files:** possibly `src/cooking_agent/providers/ollama.py` and `tests/test_ollama_adapter.py` (only if a shape is wrong); `WORKLOG.md`.

- [ ] **Step 1: Approval gate.** Ask the user before `winget install Ollama.Ollama` and `ollama pull qwen3:8b` (~5 GB).
- [ ] **Step 2: Hardware:** `nvidia-smi` — record real VRAM (Windows reports 4 GB; the RTX 4060 laptop GPU is normally 8 GB).
- [ ] **Step 3:** `LLM_PROVIDER=ollama cooking-agent diagnose --mic-seconds 0` — Expected: `[ok] llm: ollama replied ...`.
- [ ] **Step 4: One tool-calling turn** through `cooking-agent chat` against a throwaway `RECIPE_DB`: "set a pasta timer for five minutes" → `tools: set_timer`, then "how long is left" → `tools: list_timers` (exercises the tool-result message going back to the model).
- [ ] **Step 5: One recipe:** "find me a recipe for jeera rice for two" → `get_recipe` returns a validated `Recipe`.
- [ ] **Step 6: Verify, don't assume:** the tool-result message shape Ollama accepts; `think=False` is accepted by this model; `ollama ps` shows the model 100% on GPU and the context length 8192. Any mismatch → fix the adapter and its fake-client test first, run `pytest -q`, commit `fix: ollama adapter matches real server`.
- [ ] **Step 7:** WORKLOG (VRAM, first-word times, any shape fixes), Status; commit.

### Task 3: Instant command matcher

**Files:**
- Create: `src/cooking_agent/commands.py`
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: `CookingSession`, `Awaiting` (`state.py`); `RecipeStore.find`; `ToolCall`, `ToolResult` (`providers/base.py`); `parse_duration_minutes`, `format_minutes` (`recipe.py`, unchanged).
- Produces: `@dataclass CommandMatch(calls: list[ToolCall], reply: str | None = None)`.
- Produces: `match_command(text: str, session: CookingSession, store: RecipeStore | None, active_timers: list[str], awaiting: Awaiting | None) -> CommandMatch | None` — pure. `awaiting` is passed in by the agent (Task 4) and is `None` unless the app's own question was the last thing spoken.
- Produces: `words_to_digits(text: str) -> str` — "twelve minutes" → "12 minutes", "a minute" → "1 minute", "one and a half minutes" → "1.5 minutes", "half an hour" → "30 minutes"; covers one–sixty. Applied inside `commands.py` only, before calling `parse_duration_minutes`, so recipe step timers are unaffected.
- Produces: `speak_result(results: list[ToolResult]) -> str | None` with these exact phrasings (none ends with a question, because a following bare "yes" would have nothing to answer):
  - `set_timer` → "Okay, rice timer set for 12 minutes."
  - `cancel_timer` → "Rice timer cancelled."
  - `list_timers` → "Rice: 4 minutes left. Pasta: 30 seconds left." / "No timers are running."
  - `list_ingredients`, `scale_servings` → every line verbatim, joined with ". "
  - `get_recipe` → "Found Mum's Dal in your recipe book, for 4. Say ready to start."
  - any result with `"error"` → the error text unchanged.

**Matching rules** (lower-case, strip punctuation and fillers "please"/"okay"/"um", `words_to_digits`, then the **whole** utterance must match; checked in this order):

| Says | Calls | Only when |
|---|---|---|
| yes / yeah / sure / go ahead | `start_step_timer` or `next_step` | `awaiting` is TIMER / NEXT_STEP |
| no / not yet / wait | none; reply "No problem, take your time." | `awaiting` is set |
| next, next step, ready, done, continue | `next_step` | recipe loaded; no dictation open |
| back, go back, previous step | `previous_step` | recipe loaded; no dictation open |
| repeat, say that again, what was that | `repeat_step` | recipe loaded; no dictation open |
| set a [name] timer for N minutes/seconds/hours | `set_timer` (name default "kitchen") | always |
| stop / cancel the [name] timer; "stop the timer" with exactly 1 timer | `cancel_timer` | always (bare "stop" never matches) |
| how long is left, what timers | `list_timers` | always |
| what do I need, ingredients | `list_ingredients` | recipe loaded |
| make it / scale it for N (people/servings) | `scale_servings` | recipe loaded |
| cook / make [my] X | `get_recipe` | only if `store.find(X)` hits |
| bye, goodbye, that's all | `end_session` | always |

- [ ] **Step 1: Write the failing tests** — one per table row, plus:
  - `test_number_words`: "set a rice timer for twelve minutes" → `set_timer {"name": "rice", "minutes": 12}`; "a minute" → 1; "one and a half minutes" → 1.5; "thirty seconds" → 0.5.
  - `test_longer_sentences_fall_through`: "next time make it spicier", "I don't have ghee what can I use", "what goes with the rice timer" → `None`.
  - `test_yes_no_need_awaiting`: "yes" / "no" with `awaiting=None` → `None`.
  - `test_dictation_only_timers`: with a draft open, "next" → `None`, "set a pasta timer for 5 minutes" matches.
  - `test_stop_rules`: "stop" → `None` always; "stop the timer" with `["rice"]` → `cancel_timer rice`; with `["rice", "pasta"]` → `None`; never `end_session`.
  - `test_cook_saved_only`: "cook my dal" with "Mum's Dal" saved → `get_recipe {"dish": "dal"}`; empty store → `None`.
  - `test_speak_result_templates`: the exact strings above; none ends with "?".
- [ ] **Step 2: Run** `pytest tests/test_commands.py -v` — Expected: FAIL (no module `cooking_agent.commands`).
- [ ] **Step 3: Implement** with anchored regexes.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** WORKLOG, Status; commit `feat: instant offline commands`.

### Task 4: Wire the matcher into the agent + offline fallback

**Files:**
- Modify: `src/cooking_agent/agent.py` (`AgentConfig`, `on_final`, `_run_turn`, `_think`, `_ask_follow_up`, LLM error path)
- Test: `tests/test_agent.py` (new tests; `test_tool_turn_updates_state_and_history` updated deliberately, see below)

**Interfaces:**
- Consumes: `match_command`, `speak_result` (Task 3); existing `_step_speech`, `_ask_follow_up`, `ToolBox.dispatch`.
- Produces: `AgentConfig.instant_commands: bool = True` (Task 6 turns it off to measure the model).
- Produces: `LLM_UNAVAILABLE = "I can't reach the AI model right now. I can still go to the next step, repeat it, set timers, and cook recipes from your recipe book."`
- Produces: `CookingAgent._question_fresh: bool` — True only while the app's own follow-up question (from `_ask_follow_up`) or the first-run setup question was the last thing spoken; any other turn sets it False.

**Behaviour:**
- `on_final` passes the **raw** accepted text to the turn alongside the LLM text (which may carry `INTERRUPTED_NOTE`); the matcher sees only the raw text.
- At the start of `_think`, if `config.instant_commands`: `match_command(raw, session, store, timers.active_names, awaiting=session.awaiting if _question_fresh else None)`. On a match: dispatch calls in order; append each name to `m.tool_calls` (bench scoring and `chat`'s per-turn line read it); speak `_step_speech(results)` or else `speak_result(results)` or else `match.reply`; then `_ask_follow_up`; never call the LLM.
- First-run setup: while the setup question is fresh, bare "no"/"none"/"next" are not matched (they answer the question and go to the AI). Once any other turn happens, instant commands work even if setup is still pending. Offline with setup pending: `LLM_UNAVAILABLE` plus "You can finish setup later by typing cooking-agent setup."
- **History decision:** an instant turn is recorded as plain text (`Message("user", raw)`, `Message("assistant", spoken)`, no `tool_calls`). Reason: the same history goes to Ollama and Gemini, and plain text is valid for both; we don't depend on how each provider replays synthetic tool calls. `test_tool_turn_updates_state_and_history` currently sends "next step please", which now goes the instant way: update it to expect `["user", "assistant"]` for that turn, and add `test_llm_tool_turn_history_shape` using an unmatched phrase to keep the tool-shaped history check for the AI path.
- Any exception from `self.llm.stream` → speak and `on_text` `LLM_UNAVAILABLE` (replaces the generic line; also makes `chat` show it).

- [ ] **Step 1: Write the failing tests** (`make_agent` fixture; an LLM that fails the test if called):
  - `test_instant_next_step_skips_llm`: recipe loaded, "next" → "Step 1. …" plus the follow-up; `llm.calls == []`; `metrics.turns[-1]["tool_calls"] == ["next_step"]`.
  - `test_instant_timer`: "set a rice timer for twelve minutes" → "Okay, rice timer set for 12 minutes."; timer exists.
  - `test_unmatched_goes_to_llm`: "what can I use instead of ghee" → LLM called.
  - `test_stale_yes_goes_to_llm`: step read (awaiting set) → "what do I need" (instant) → "yes" → LLM called, step index unchanged.
  - `test_fresh_yes_is_instant`: step with a timer read → "yes" → `start_step_timer`, LLM not called.
  - `test_command_after_barge_in_is_instant`: interrupt a long reply, then "go back" → `previous_step`, LLM not called.
  - `test_llm_down_speaks_offline_message`: `stream` raises `ConnectionError` → `tts.spoken[-1] == LLM_UNAVAILABLE`; `on_text` received it.
  - `test_history_after_instant_turn_is_text_only` and `test_llm_tool_turn_history_shape`.
  - `test_saved_recipe_offline_end_to_end`: failing LLM + store with a recipe **and a `PreferenceStore` that still needs setup**: setup question spoken → "cook my dal" → "next" → "yes" → "next" all succeed without the LLM.
  - `test_instant_commands_can_be_disabled`: `AgentConfig(instant_commands=False)`, "next" → LLM called.
- [ ] **Step 2: Run** `pytest tests/test_agent.py -v` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** in `agent.py`.
- [ ] **Step 4: Run** `pytest -q` and `cooking-agent demo` — Expected: all pass; demo runs to the end.
- [ ] **Step 5:** WORKLOG, Status; commit `feat: instant commands in agent; offline fallback message`.

### Task 5: Phase-specific tool lists

**Files:**
- Modify: `src/cooking_agent/tools.py` (`ToolBox.specs_for`), `src/cooking_agent/agent.py` (`_think` uses it)
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `ToolBox.specs_for(session: CookingSession) -> list[ToolSpec]`, in registry order (stable prompt prefix between turns).

**Rules:**
- Always (10): `get_recipe`, `list_saved_recipes`, `start_recipe_dictation`, `set_timer`, `cancel_timer`, `list_timers`, `remember_preference`, `forget_preference`, `get_preferences`, `end_session`.
- + setup pending: `finish_setup`.
- + dictating: `add_to_dictated_recipe`, `remove_from_dictated_recipe`, `read_dictated_recipe`, `save_dictated_recipe`, `discard_dictated_recipe`.
- + recipe loaded: `next_step`, `previous_step`, `repeat_step`, `start_step_timer`, `list_ingredients`, `scale_servings`, `substitute_ingredient`, `save_recipe`.

- [ ] **Step 1: Write the failing tests:** `test_specs_no_recipe` (exactly the 10), `test_specs_cooking` (18, has `next_step`, no `finish_setup`), `test_specs_setup` (has `finish_setup`), `test_specs_dictating` (has the 5), `test_specs_order_stable`.
- [ ] **Step 2: Run** `pytest tests/test_tools.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement**; `_think` calls `self.tools.specs_for(self.session)`.
- [ ] **Step 4: Run** `pytest -q` — Expected: all pass.
- [ ] **Step 5:** WORKLOG, Status; commit `perf: phase-specific tool lists`.

### Task 6: Diet check in code (vegetarian, vegan, Jain)

Today only allergies and health conditions are checked in code; diets rely on the instruction to the model, which is riskier with a small local model.

**Files:**
- Modify: `src/cooking_agent/dietary.py` (`DIETS`, `diet_hint`, `find_conflicts(..., diets=())`), `src/cooking_agent/preferences.py` (`describe_for_recipe` diet line uses `diet_hint`), `src/cooking_agent/tools.py` (`_conflicts`, `_write_recipe`, `_warnings`, `_get_recipe`)
- Test: `tests/test_dietary_safety.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: `diets_in` (`preferences.py`), `_has_word` (`dietary.py`).
- Produces: `DIETS: dict[str, tuple[list[str], list[str], str]]`: diet -> (forbidden words, look-alikes that are fine, guidance) for `"vegetarian"`, `"vegan"`, `"jain"`:
  - vegetarian: meat, poultry, fish, seafood, egg, mayonnaise, meat/fish/chicken stock or broth, fish sauce, gelatin, lard. Fine: eggplant, eggless, egg-free, vegan, vegetable stock, chickpea, plant-based.
  - vegan: everything vegetarian forbids + all dairy (`_DAIRY`), ghee, honey. Fine: the vegetarian look-alikes + `_PLANT_DAIRY`.
  - jain: everything vegetarian forbids + honey + onion, garlic, potato, carrot, beetroot, radish, ginger, turnip, sweet potato, yam, leek, shallot, spring onion, scallion. Fine: the vegetarian look-alikes + dry ginger, sonth.
- Produces: `diet_hint(diet: str) -> str`, e.g. `"vegetarian (no meat, fish, eggs, gelatin, or meat or fish stock)"`.
- Produces: `find_conflicts(ingredient_names, allergies=(), conditions=(), diets=())`; a diet conflict's reason is `"<diet> diet"` (message `"chicken stock (conflicts with vegetarian diet)"`).

**Rules:**
- Diets checked = stated diets (normalised with `diets_in`) plus diets named in the request's constraints (`"vegan, for four"`). Learned habits are not code-checked (they are suggestions, not rules).
- **The request wins for diet, as the README promises** ("chicken biryani, just this once"): a diet is skipped when the dish or constraints themselves name something that diet forbids. Allergies and conditions are never skipped.
- Generated recipe: a diet conflict triggers the same single rewrite as allergies; a remaining one becomes a spoken warning. Saved recipe: warning only, never changed.

- [ ] **Step 1: Write the failing tests**
  - `test_vegetarian_flags_meat_fish_egg`: `["chicken stock", "2 eggs", "fish sauce", "gelatin"]` with `diets=["vegetarian"]` -> 4 conflicts, reason `"vegetarian diet"`.
  - `test_vegetarian_lookalikes_ok`: `["chickpeas", "eggplant", "vegetable stock", "eggless mayonnaise"]` -> none.
  - `test_vegan_flags_dairy_ghee_honey`: `["ghee", "honey", "paneer"]` -> 3; `["coconut milk", "oat milk"]` -> none.
  - `test_jain_flags_root_vegetables`: `["onion", "garlic", "potato", "ginger"]` -> 4; `["dry ginger powder", "tomato"]` -> none.
  - `test_diet_hint`: `diet_hint("vegan")` mentions dairy and honey.
  - `test_generated_meat_rewritten_for_vegetarian` (tools): vegetarian profile; fake LLM returns a recipe with chicken, then one without -> second prompt names the conflict; result has no `warnings`.
  - `test_request_overrides_diet` (tools): vegetarian profile, `get_recipe(dish="chicken biryani")` returning chicken -> one LLM call, no warnings.
  - `test_one_off_vegan_constraint_checked` (tools): no stated diet, `constraints="vegan"`, recipe with ghee -> rewritten.
  - `test_saved_recipe_diet_warning` (tools): saved "Mum's Special" containing chicken, vegetarian profile, request "mum's special" -> warning `"contains chicken (conflicts with vegetarian diet)"`, recipe unchanged.
  - `test_allergy_never_overridden_by_request` (tools): peanut allergy, `dish="peanut chikki"` -> still rewritten.
- [ ] **Step 2: Run** `pytest tests/test_dietary_safety.py tests/test_tools.py -v` - Expected: new tests FAIL.
- [ ] **Step 3: Implement** in `dietary.py`, `preferences.py`, `tools.py`.
- [ ] **Step 4: Run** `pytest -q` - Expected: all pass.
- [ ] **Step 5:** README "Safety check" paragraph mentions diets; WORKLOG, Status; commit `feat: code check for vegetarian, vegan and Jain diets`.

### Task 7: Choose the model by measuring

**Files:**
- Modify: `src/cooking_agent/bench.py`, `src/cooking_agent/cli.py` (`bench --text-stt`, `bench --no-instant`), `src/cooking_agent/config.py` (final `ollama_model` default), `.env.example`, `README.md`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `FixtureSTT`, `_make_providers`, `run_fixture` (`bench.py`); `AgentConfig.instant_commands` (Task 4).
- Produces: `_make_providers(fake: bool, fx: Fixture, speech_end_s: float, text_stt: bool = False)`; `text_stt=True` returns `FixtureSTT` with the real LLM and TTS (removes Vosk mishearing the synthetic voice).
- Produces: `run_benchmark(..., instant: bool = True)`; `--no-instant` builds agents with `AgentConfig(instant_commands=False)` so all ten fixtures reach the model (otherwise the matcher answers seven of them).
- Produces: `src/cooking_agent/diet_eval.py` with `DIET_CASES` (10 `DietCase(dish, allergy, condition, diet)`) and `async run_diet_eval(llm) -> dict` (per case: valid recipe, conflicts left after the safety check, rewrite needed; summary pass rate); CLI `cooking-agent diet-eval [--out FILE]`. Cases are chosen so the diet must hold (none names a forbidden ingredient): vegetarian pad thai, caesar salad, ramen; vegan kheer, palak paneer; Jain dal tadka, pav bhaji; lactose-intolerant paneer butter masala; celiac pasta; peanut-allergy satay.

- [ ] **Step 1: Write the failing tests:** `test_text_stt_flag_uses_fixture_stt`; `test_no_instant_sends_matched_fixture_to_llm` (fake mode, `next_step` fixture, `instant=False` → scripted LLM was called); `test_diet_eval_scores_cases` (fake LLM returning chicken first then a clean recipe -> vegetarian cases pass with a rewrite counted). Run (FAIL), implement, `pytest -q` (PASS), commit `feat: bench --text-stt --no-instant; diet-eval`.
- [ ] **Step 2: Approval gate:** pull 2 more candidates (~5 GB each): the current tool-capable 7–9B models in Ollama's library, e.g. `llama3.1:8b` plus one current Gemma or Mistral.
- [ ] **Step 3: Routing and speed:** per model, `LLM_PROVIDER=ollama OLLAMA_MODEL=<m> cooking-agent bench --text-stt --no-instant --runs 3`; record tool routing accuracy and first-word time.
- [ ] **Step 4: Diet test:** per model, `LLM_PROVIDER=ollama OLLAMA_MODEL=<m> cooking-agent diet-eval --out bench/diet-<m>.json`; record pass rate and how often a rewrite was needed. Spot-read two recipes per model for sense.
- [ ] **Step 5: Pick** highest routing accuracy with first word under ~1.5 s and a diet-test pass rate of 10/10 after the safety check (ties: fewer rewrites); set the default in `config.py` and `.env.example`; `ollama rm` the others; results table in README and WORKLOG; Status; commit `chore: choose default local model`.
- [ ] **Step 6:** Hand back to the user for a manual `cooking-agent run` test (same script as the first test).

---

## Plan history

- **2026-11-15** — First version, written from the agreed "simple and local" scope.
- **2026-11-15** — Revised after review by the advisor (Fable):
  - new Task 2, a smoke test against a real Ollama right after the adapter;
  - `num_ctx` is set explicitly;
  - the benchmark gets `--no-instant`, because the matcher would otherwise answer 7 of the 10 clips;
  - matching uses the raw words, so commands still work after an interruption;
  - a bare yes/no only counts while the app's own question is fresh; bare "stop" never matches;
  - canned replies don't end in questions;
  - number words are handled in `commands.py`, so recipe step timers are unaffected;
  - photo/PDF import goes to Gemini when its key is set, with a caught error otherwise;
  - first-run setup has defined behaviour offline;
  - recording instant turns as plain text is now a deliberate decision, and the existing test is updated.
- **2026-11-16** — Scope: social-media import limited to YouTube Shorts. **Proposed, awaiting the user's yes:**
  - a code check for diets (vegetarian/vegan/Jain) in `dietary.py`, because today only allergies and conditions are code-checked;
  - an automatic diet test (about 10 profile × dish cases) in Task 6;
  - no model training.
- **2026-11-16** — User approved the plan. Execution: Native with Opus 5.5 building, Fable advising, and a Fable reviewer for the whole branch at the end.
- **2026-11-16** — User approved the diet proposal: new Task 6 (diet check in code); measuring becomes Task 7 and adds an automatic diet test (`cooking-agent diet-eval`). User approved running the Ollama install and model downloads.
