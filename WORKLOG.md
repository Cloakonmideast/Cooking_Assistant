# Work log

A record of every change made to this project folder, newest first. Each entry
says what was done, why, and how it was checked.

---

## 2026-11-16 — Phase 2, Task 12: understanding test, and three problems it found

* New `cooking-agent understanding-eval [--runs 3] [--out FILE]`
  (`understanding_eval.py`): 20 typed utterances on step 2 of a pulao with the
  real "ready?" question pending (7 questions, 6 changes, 5 commands, 2
  chit-chat). Accepted outcomes per case: an answer (no state-changing tool,
  step and recipe unchanged) or a tool with arguments plus a check on the
  result (e.g. the cashews are really gone). Only tool calls that succeeded
  count. Fresh agent per run, in-memory recipe book and profile (nothing
  written to the cook's database). The wording differs from the examples in
  the prompt and tool descriptions (a test checks no case appears there).
* **Problems the first real runs on qwen3 found, all fixed:**
  1. **Instant turns taught the model to stop calling tools.** In the
     history, "next" -> "Step 1. ..." looked like the model reading steps
     without a tool, and qwen3 copied it: it wrote `next_step` as text, said
     "let's remove the green chillies" without removing them, or said
     nothing. Measured on the 11 change/command cases x 3, the expected tool
     was called 21/33 times with that history, 24/33 with none and 28/33
     with the instant turns written as tool calls. Instant turns are now
     saved as (tool call, real tool result, spoken text) for Ollama
     (`OllamaLLM.tool_history`); Gemini and the test fakes keep plain text,
     which Gemini needs (its function calls carry signatures).
  2. **`substitute_ingredient` failed on "2 tbsp".** The model sent the amount
     with its unit; the cook heard "the quantity should be a number" instead
     of the swap. The amount is now split into number and unit ("1 1/2 tsp",
     "to taste" also work).
  3. **Reading ahead from the recipe text.** Asked "what now", the model read
     step 3 from `recipe_text` without calling `next_step`, so the app still
     thought the cook was on step 2. The prompt now says steps are only read
     by the step tools, and "finished / what's next" means `next_step`. The
     eval case that exposed this was replaced with a fresh paraphrase so the
     score isn't flattered.
  The eval itself also needed two fixes: a failed tool call no longer counts
  as right, and step 2 no longer states a time (it made the pending question
  a timer offer, not "ready?").
* **Context check:** with a full 40-message history and every recipe tool,
  a turn's prompt is 4,510 of the 8,192 tokens (`prompt_eval_count`), so
  nothing is silently cut off.
* **Result on qwen3:8b after the fixes: 100% (60/60 runs), median turn
  0.87 s.** Before them, single runs scored 85-90%.
* README "Understanding test" section.
* Checked: 13 new tests (`tests/test_understanding_eval.py`; 5 more in
  `tests/test_mid_recipe.py` for "2 tbsp" and the history shape), each seen
  failing first except the eval module's own tests, which were written after
  the module; for those the judge was broken on purpose (step and recipe
  checks removed) and the tests caught it. Full suite 404 passed.

---

## 2026-11-16 — Phase 2, Task 11: keep the model loaded

* While `run` (voice) or `chat` is going, a background task pings the model
  (`llm.warmup()`) after **15 minutes** without it being used, and once
  about **a minute before each timer ends**, so the reply after a long simmer
  doesn't wait for Ollama to load the model again. `AgentConfig`:
  `keep_warm_every_s=900`, `keep_warm_before_timer_s=60`,
  `keep_warm_poll_s=5` (0 turns it off).
* The decision is a pure function, `keep_warm_due`, tested without sleeping.
  Timers are keyed by name and end time, so a timer set again under the same
  name gets its own ping. The loop checks every 5 seconds, so timers set after
  it last looked (including by instant commands) are noticed. Failed pings
  are logged, never spoken. LLMs without `warmup` (Gemini, the test fakes)
  skip it.
* **Bug found and fixed while checking this on the real Ollama:** the
  warm-up (also used at start-up since Phase 1) loaded the model with
  Ollama's default 4096-token context, while chat asks for 8192, so Ollama
  reloaded the model on the next reply. Measured: first reply after a ping
  **5.7 s**, then 0.32 s. The warm-up now sends the same options as chat;
  after the fix the reply right after a ping took **0.57 s** and
  `ollama ps` shows 8192 context, 100% GPU, "29 minutes from now".
* README: what keeping it loaded means for graphics memory.
* Checked: 10 new tests in `tests/test_keep_warm.py` (they failed first: the
  function didn't exist; the context-size test failed on the old warm-up);
  full suite 372 passed.

---

## 2026-11-16 — Phase 2, Task 10: learn and use what the cook changes

* **Saved every time** (`preferences.py`, new `adjustments` table): each swap
  (`substitute_ingredient`), each ingredient left out, added, or made more or
  less (`change_recipe`), with the recipe it was made in.
* **A habit after two different recipes** (`ADJUST_MIN_TIMES = 2`): the same
  change twice in one recipe doesn't count. `learned()` now also returns
  `usually_swaps`, `usually_leaves_out`, `usually_adds`, `usually_more`,
  `usually_less`.
* **Used**: the recipe writer gets "Changes they usually make, apply them
  unless the request says otherwise: usually swaps ghee for butter; usually
  leaves out cashews; likes more garlic..."; the reply mentions them
  (`tailored_to`). A recipe from the cook's own book is never changed: the
  result lists `your_usual_changes` that apply to it, and the assistant
  offers them.
* **Safety**: a learned swap, addition or "more" that clashes with an
  allergy, condition or stated diet is never passed on (e.g. "peanut oil"
  with a peanut allergy).
* **Learned diet = heads-up only**: if the cook usually asks for vegetarian
  but never stated it, a recipe with chicken gets a warning ("has chicken,
  and you usually cook vegetarian") and no rewrite; not when the dish names
  it ("chicken biryani"), and not when a stated or requested diet applies.
  An eggetarian whose requests say "veg" is not given a learned vegetarian
  habit (eggs aren't flagged).
* `cooking-agent prefs` shows the usual changes ("usually swaps: ghee to
  butter"); `prefs forget-history` clears them too.
* README "What it learns" updated.
* Checked: 14 new tests in `tests/test_learned_habits.py` (13 failed first;
  one assertion was fixed because "paneer" appears in the vegan rule text
  itself); full suite 362 passed. Benchmarks and diet-eval don't touch the
  real preferences database (bench uses none, diet-eval a temporary one).

---

## 2026-11-16 — Phase 2, Task 9: questions and changes mid-recipe

* **Questions about the dish** ("how do I know the rice is done?", "can I use
  brown rice?") are answered from the actual recipe: the state the model sees
  every turn now has `recipe_text` (title, servings, ingredient lines at the
  current servings, numbered steps with `[current step]` marked). The prompt
  says to answer without a tool and stay on the same step.
* **Changes** use a new tool, `change_recipe` (offered only while a recipe is
  loaded), with four actions:
  * `remove_ingredient` ("skip the cashews"): whole-word matching, so "ice"
    no longer finds "rice"; the last ingredient can't be removed. It returns
    the steps from the current one on that mention it (`steps_to_update`),
    and the model rewrites them with `change_step` in the same turn, because
    steps are read out word for word.
  * `add_ingredient` ("add some peas", "1 cup").
  * `change_amount` ("2 tbsp", or double / half / more / less; "make it less
    spicy" means less chilli). Amounts are taken at the servings being
    cooked now and stored at the recipe's own servings, so rescaling still
    works. An imported line's verbatim text is dropped once changed, so the
    new amount is what gets read.
  * `change_step`: new wording for a step (current step by default). Any time
    in the new wording becomes the step's timer offer.
* Added or increased ingredients are checked against allergies, conditions
  and diets. The dish the cook asked for still wins for a diet ("chicken
  biryani" then "add more chicken" isn't flagged); `substitute_ingredient`
  now uses the same rule.
* The current step never moves on a change or question; the instant command
  matcher ignores these phrases (tested).
* README tool table updated.
* Checked: 20 new tests in `tests/test_mid_recipe.py` (14 failed first; the 6
  matcher guards already passed), two existing tests updated for the new
  snapshot key and tool count; full suite 348 passed.

---

## 2026-11-16 — Phase 2, Task 8: eggetarian diet

* New diet **eggetarian** (`dietary.py`): everything a vegetarian avoids except
  eggs (no meat, poultry, fish, seafood, gelatin, meat or fish stock).
* Recognised from: "eggetarian", "eggitarian", "ovo-vegetarian", "vegetarian
  but I eat eggs", "eggs are fine/ok", "vegetarian who eats eggs", and "I eat
  eggs but no meat" (no "vegetarian" needed). "Vegetarian, I don't eat eggs" /
  "no eggs" stays plain vegetarian (`preferences.diets_in`).
* When both are named (stated eggetarian, request "veg fried rice"),
  eggetarian wins, so eggs aren't flagged (`prefer_eggetarian`, applied after
  the stated and requested diets are combined).
* **Eggless** is now a checked diet too (no eggs, mayonnaise, meringue), so an
  "eggless sponge cake" request from an eggetarian is still checked for eggs.
* The recipe writer is told "eggetarian (vegetarian but eggs are fine: ...)";
  the `remember_preference` tool description lists eggetarian so the model
  saves that word.
* README "Safety check" updated.
* Checked: 17 new tests in `tests/test_dietary_safety.py` failed first, then
  passed; full suite 328 passed.

---

## 2026-11-16 — Removed the Claude "Co-Authored-By" line from all commits

* Claude Code settings (`~/.claude/settings.json`, applies to every project):
  `"attribution": {"commit": "", "pr": ""}`, so no attribution line is added
  to new commits or pull request descriptions.
* Rewrote the 18 existing commits on `windows-py311-fixes` and `local-model`
  to drop the line (`git filter-branch --msg-filter`). Only the messages
  changed: the files in every commit are identical (checked with `git diff`
  against the old branches). Commit IDs changed, so IDs quoted earlier in
  chat no longer match; `main` is untouched.
* Backups of the old branches are kept by git under `refs/original/` until
  removed.

---

## 2026-11-16 — Phase 2 planned

New requests, added to the plan file as Tasks 8–14:
* an **eggetarian** diet (vegetarian plus eggs, no meat);
* **learn and use everything the cook prefers**: swaps, things left out or
  added, more/less of something, as well as diets, saved and applied to new
  recipes;
* **questions and changes mid-recipe** understood ("can I use brown rice?",
  "skip the cashews", "make it less spicy"), with the recipe changed in place
  and the current step kept;
* **keep the model loaded**: a ping every 15 minutes and about a minute
  before any timer ends;
* **test Gemma 4** (`gemma4:e4b`) and select the best model, now also using a
  new understanding test;
* a **Fable review** of Phase 2.

My calls in the plan: a habit counts after it happens twice (every event is
still saved); a learned diet gives a heads-up but never forces a rewrite.

---

## 2026-11-16 — Fable code review of the whole branch, and fixes

A separate Fable reviewer read all 14 commits (26 files) against the plan,
ran the tests and probed with real phrases. Verdict: **ready "with fixes"**.
No critical issues; the offline fallback, number words, longer sentences and
commands after an interruption were all confirmed working. It found nine
important issues, and I upgraded one of its minor ones (the silent first
minute). All ten are fixed, each with a test that reproduced the problem
first (`tests/test_review_fixes.py`, 36 tests):

1. **"non-veg" / "non-vegetarian" / "not vegetarian" was read as
   vegetarian** (and "with lots of veg" too), forcing veg recipes and false
   warnings on meat-eaters. Now negated words and "veg" meaning vegetables
   don't count as a diet (`preferences.diets_in`).
2. **Restating a rule switched the diet check off** ("Jain dal, no onion no
   garlic" let potato and honey through). Ruled-out words ("no onion",
   "garlic-free") no longer count as asking for them.
3. **Naming one forbidden item switched off the whole diet** (vegan "palak
   paneer" let cream and eggs through). Now only the named item is allowed;
   the rest of the diet is still checked (`dietary.requested_words`).
4. **"Hold on" to a timer offer lost the timer, and the next "yes" skipped a
   step.** "Hold on" / "not yet" / "wait" now keep the question open;
   "start it" only ever answers a timer offer.
5. **"I'm finished" or "that's all" quit the app mid-cooking** (and cancelled
   timers). "I'm finished" no longer quits; "that's all" only quits when no
   recipe, timer or setup is in progress. "Bye" still quits.
6. **A "yes" could answer a question that was cut off, overtaken by a timer
   announcement, or asked long ago.** A question now only counts once it was
   heard in full, a timer announcement clears it, and it expires after
   60 seconds.
7. **A second unnamed timer replaced the first** ("kitchen" twice). It's now
   "kitchen 2", "kitchen 3"...
8. **"Five minutes and thirty seconds" set 5 minutes.** Both parts are now
   added.
9. **"Cook the rice" mid-recipe replaced the recipe in progress** without
   asking. While cooking, that now goes to the AI instead.
10. **First reply after starting could be ~1 minute of silence** (model
    loading; measured 68 s). `run` and `chat` now load the model first and
    say so; the Ollama client has a 120 s timeout, so a hung server gives
    "I can't reach the AI model" instead of silence forever.

Checked against the real local model afterwards: model loaded at start, "hold
on" then "yes" worked, step navigation and timers were instant. Suite
308/308; demo runs to the end.

**Not fixed (minor, listed for later):** a few diet words missing or
over-matched (e.g. catfish, steak, dashi, mawa, shrikhand, mushrooms and
Hindi names like aloo/pyaz for Jain; "oyster mushrooms", "goat cheese",
"butter beans", "ginger powder" wrongly flagged); "vegetarian but I eat eggs"
is treated as plain vegetarian; the diet test can only see what the app's own
word lists see; "well done" moves to the next step; very long timers give a
technical error message; small wording gaps offline.

**Seen while checking, not from the review:** qwen3 still answers "none" to
the first-run allergy question with "No problem, take your time." without
marking setup done, so the question comes back next launch.

---

## 2026-11-16 — Task 7: measured three local models; `qwen3:8b` stays the default

* New measuring tools:
  * `cooking-agent bench --no-instant` sends every clip to the AI (instant
    commands would otherwise answer several of them without it);
  * `cooking-agent bench --text-stt` feeds each clip's correct text instead
    of recognising its audio, so mishearing doesn't blur the comparison;
  * `cooking-agent diet-eval` (new `diet_eval.py`): 10 recipes where the
    rule must hold but the dish name doesn't give it away (vegetarian pad
    thai, vegan chocolate cake, Jain pav bhaji, celiac pasta, peanut-allergy
    satay...), each scored after the app's own safety check.
* Results (RTX 4060 8 GB; 10 clips × 3 runs, every clip to the AI):

  | Model | Right action | First AI word (typical / 9 in 10) | Diet test |
  |---|---|---|---|
  | **qwen3:8b** | 90% | **1.0 s / 1.3 s** | 10/10 (1 rewrite) |
  | llama3.1:8b | 90% | 1.5 s / 2.1 s | 10/10 (1 rewrite) |
  | mistral:7b | 10% (almost never used actions) | 0.5 s / 0.6 s | 10/10 (1 rewrite) |

  qwen3's only miss: "I don't have ghee, what can I use instead" got an
  answer without the substitute action. llama3.1's: it called an action for
  small talk.
* Spot-read two qwen3 recipes: vegetarian pad thai (no fish sauce or egg)
  and Jain dal tadka (no onion, garlic or ginger) both make sense.
* **Decision: keep `qwen3:8b`** (same accuracy as llama3.1, about a third
  faster). Removed `llama3.1:8b` and `mistral:7b` with `ollama rm` (frees
  ~9 GB).
* Diet case "vegan palak paneer" from the plan was replaced by "vegan
  chocolate cake": the dish name contains paneer, so the "request wins for a
  diet" rule would rightly skip the check and the case would test nothing.
* README: benchmark switches, the diet test, and the results table.
* Tests: 5 new (failed first, then passed); suite 272/272.

---

## 2026-11-16 — Task 6: diets checked in code (vegetarian, vegan, Jain)

* `dietary.py`: new diet rules, checked exactly like allergies:
  * vegetarian: no meat, poultry, fish, seafood, eggs, gelatin, meat/fish
    stock (eggplant, eggless mayo, chickpeas, vegetable stock are fine);
  * vegan: also no dairy, ghee or honey (plant milks are fine);
  * Jain: vegetarian, no honey, no onion, garlic, potato, carrot, ginger or
    other root vegetables (dry ginger powder is fine).

  Each diet's exceptions apply only to their own part, so "rice" excuses
  "rice milk" from the dairy rule but never "chicken fried rice".
* `tools.py`: a generated recipe that breaks your diet is sent back once to
  be rewritten, then you're warned if it still does; your own saved recipes
  are never changed, only warned about. Diets checked are the ones saved in
  your profile plus any named in the request ("vegan kheer"). As the README
  always promised, the request wins for a diet when it names the ingredient
  itself ("chicken biryani"); allergies and conditions are never skipped.
* `preferences.py`: the recipe instructions now spell out what the diet
  means ("Diet: vegetarian (no meat, poultry, fish, ...)").
* Decision: vegetarian excludes eggs (Indian usage, matching this app's
  recipes). Habits the app has only learned are not code-checked.
* README "Safety check" paragraph updated.
* Tests: 14 new (failed first, then passed); suite 267/267.

---

## 2026-11-16 — Task 5: the AI is offered only the actions that fit the moment

* `tools.py`: new `ToolBox.specs_for(session)`. Instead of all 24 actions
  every turn, the AI now sees 10 when no recipe is loaded, 18 while cooking,
  plus the setup action during first-run setup and the 5 dictation actions
  while dictating. Fewer choices means a small local model is faster and
  picks the right one more often. The list keeps the same order each turn,
  so Ollama can reuse the unchanged start of the prompt.
* `agent.py` uses it for every AI round.
* Tests: 5 new in `tests/test_tools.py` (failed first, then passed); suite
  254/254.

---

## 2026-11-16 — Task 4: instant commands connected to the assistant; offline message

* `agent.py`: before asking the AI, every utterance is checked by the
  instant-command matcher (Task 3). If it's one ("next", "say that again",
  "set a rice timer for twelve minutes", "cook my dal"...), the action runs
  straight away and a fixed reply is spoken. No AI involved, so it's
  instant and works with no model and no internet.
* A bare "yes"/"no" is only answered instantly right after the app's own
  question ("start a timer?", "ready for the next step?"). After anything
  else it goes to the AI, so an old question can't make "yes" skip a step.
  "Not yet" keeps the question open.
* Commands spoken right after interrupting the assistant still work: matching
  uses your words, not the "you were interrupted" note added for the AI.
* Right after the first-run allergies/diet question, "no", "yes", "next" and
  "that's all" are treated as answers to it (they go to the AI); clear
  commands like "cook my dal" or timers still work.
* If the AI can't be reached (Ollama not running, no internet with Gemini),
  the assistant now says: "I can't reach the AI model right now. I can still
  go to the next step, repeat it, set timers, and cook recipes from your
  recipe book." (plus how to finish setup if it's still pending). It's also
  printed in `chat`, which used to show nothing.
* Instant turns are kept in the conversation history as plain text, which
  works with both Ollama and Gemini.
* `AgentConfig(instant_commands=False)` turns the feature off (the model
  benchmark in Task 7 uses this).
* Tests: 13 new agent tests and 2 more matcher tests (failed first, then
  passed). Existing tests that deliberately test the AI's own step pacing now
  run with instant commands off; the history-shape test and the benchmark
  test were updated for instant turns (the fake benchmark: 4 of 10 clips are
  now answered instantly, routing still 100%). Suite 249/249; demo runs to
  the end.

---

## 2026-11-16 — Task 2: tried the real local model (and fixed two bugs it exposed)

* Installed Ollama 0.40.16 (`winget`) and downloaded `qwen3:8b` (5.2 GB).
  Note: the Ollama app didn't start its server by itself after install, so
  the server was started with `ollama serve` for these tests. After a
  restart of Windows the Ollama app normally starts it.
* Hardware: RTX 4060 Laptop GPU has the full **8 GB** (Windows reports 4 GB).
  Ollama's own default context window on this GPU is only 4096 tokens, which
  confirms the app must send 8192 itself. `ollama ps`: model 100% on GPU,
  context 8192, 6.2 GB.
* `cooking-agent diagnose`: all `[ok]`, including `llm: ollama`. First reply
  68 s (the first-ever load from disk); after that replies start in
  **0.1–2 s**.
* **Bug found and fixed:** recipe writing failed every time. The model,
  held to the exact recipe format, writes `timer_minutes: 0` for steps with
  no wait, and the format rejected 0. Now 0 means "no timer"
  (`recipe.py`). Test `test_zero_timer_means_no_timer` (failed first, then
  passed); the old test that expected 0 to be rejected now checks a negative
  timer instead.
* **Flaky test found and fixed:** `test_cli_setup_by_typing` failed about 2
  runs in 10 on Windows. Saved preferences were sorted by the time they were
  added, and two saves within one Windows clock tick (~15 ms) got the same
  time, so their order was random. They're now sorted by insertion order as
  a tie-breaker (`preferences.py`). Test
  `test_stated_keeps_insertion_order_when_timestamps_tie` (failed first,
  then passed); the flaky test then passed 10/10.
* After the fix, a typed session wrote a real jeera rice recipe for two and
  read step 1 word for word.
* **Seen, not fixed here (small-model behaviour):** answered "none" to the
  allergy question with "No problem, take your time." without saving it;
  answered "what do I need" and "how long is left" from memory instead of
  checking; ignored a bare "ready". The instant commands (Task 4) take most
  of these away from the model. Task 5 (fewer tools) and Task 7 (measured
  model choice) target the rest.
* Suite: 235/235.

---

## 2026-11-16 — Task 3: instant commands (matcher)

* New `src/cooking_agent/commands.py`: decides whether what you said is one
  of the everyday commands ("next", "go back", "say that again", "yes"/"no"
  to the app's own question, "set a rice timer for twelve minutes", "stop
  the pasta timer", "how long is left", "what do I need", "make it for six",
  "cook my dal", "bye") and which action to run. Number words are understood
  ("twelve", "a minute", "one and a half", "half an hour").
* Strict on purpose: the whole sentence must match, so "next time make it
  spicier" still goes to the AI. A bare "stop" never cancels a timer. During
  first-run setup or dictation only timer commands count.
* Spoken replies for these commands are fixed sentences ("Okay, rice timer
  set for 12 minutes."); none ends in a question.
* Not connected to the assistant yet (that's Task 4).
* Tests: 50 new in `tests/test_commands.py` (failed first, then passed).

---

## 2026-11-16 — Task 1: local AI adapter (Ollama)

* New `src/cooking_agent/providers/ollama.py` (`OllamaLLM`): streams replies
  and tool calls from a local Ollama server, and writes recipes in the app's
  strict recipe format (one retry if the answer doesn't fit). Always sends a
  context window of 8192 tokens, so our ~3,200-token prompt is never cut off,
  and keeps the model loaded between turns.
* The AI now defaults to the local model (`LLM_PROVIDER=ollama`,
  `OLLAMA_MODEL=qwen3:8b`). Gemini still works with `LLM_PROVIDER=gemini`.
* `cooking-agent recipes add` uses Gemini whenever `GEMINI_API_KEY` is set
  (it can read photos, PDFs and recordings), otherwise the local model. A
  photo or PDF without a Gemini key now gives a clear message, not a crash.
* Video transcription only uses Gemini (or Deepgram); a local model is never
  asked to transcribe.
* Added dependency `ollama==0.6.3`; README (setup, run table, services and
  packages) and `.env.example` updated.
* Tests: 12 new in `tests/test_ollama_adapter.py` (watched them fail first,
  then pass); full suite 183/183.

---

## 2026-11-16 — Scope: YouTube Shorts only; diet rules found unchecked

* Recipe import from social media: YouTube Shorts only for now (Instagram
  often needs a login even for public accounts; set aside).
* Checked how recipes follow dietary restrictions: the profile goes into the
  recipe request as rules, the recipe must fit a strict format, and code
  (`dietary.py`) checks every ingredient against **allergies and health
  conditions**, with one rewrite and then a spoken warning. **Diets
  (vegetarian, vegan, Jain) are not code-checked**: only the instruction to
  the model enforces them, which is riskier with a small local model.
* Proposed in the plan file, not yet approved: a code check for diets and an
  automatic diet test when choosing the local model. No model training.
* No code changed.
* Plan approved. How it will be built: Opus 5.5 (this session's model)
  builds each task; Fable advises at decision points (already the configured
  advisor); at the end a separate Fable reviewer checks the whole
  `local-model` branch, and its findings are fixed and logged. No settings
  needed changing.

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
* **Plan revised after review by the advisor (Fable).** It found that the
  model benchmark would mostly measure the new instant commands rather than
  the model; that commands spoken right after an interruption wouldn't match;
  that a stale "yes" could skip a step; and that Ollama's default context
  window could silently cut off our ~3,200-token prompt. Fixes were added to
  the plan, including a new Task 2: a smoke test against a real Ollama before
  building on the adapter. The full list is under "Plan history" in the plan
  file.

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
