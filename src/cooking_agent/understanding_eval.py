"""Understanding test: mid-recipe, does the assistant do the right thing with
what the cook says?

A recipe is loaded and the cook is on step 2, with the app's real "Are you
ready for the next step?" question just asked. Each case is one typed
utterance and the outcomes that count as right:

* ``answer``: a spoken reply, no tool that changes anything, the step and the
  recipe untouched ("how can I tell when the rice is cooked?");
* a tool with some of its arguments and, where it matters, a check on the
  resulting recipe ("skip the cashews" -> change_recipe remove_ingredient,
  and the cashews are gone).

Instant commands stay on, as in real use. Every case runs ``runs`` times on a
fresh agent (with an in-memory recipe book and preferences, so nothing is
written to the cook's own database) and the score is the share of runs that
were right. The wording is deliberately different from the examples in the
system prompt and tool descriptions. Used to choose a local model
(``cooking-agent understanding-eval``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agent import AgentConfig, CookingAgent
from .audio import SimulatedSink
from .preferences import PreferenceStore
from .providers.base import ToolCall
from .providers.fake import QueueSTT, ToneTTS
from .recipe import Recipe
from .recipe_store import RecipeStore
from .state import CookingSession

KINDS = ("question", "change", "command", "chat")
# Tools that only read: allowed alongside an "answer".
READ_ONLY = {"list_ingredients", "list_timers", "get_preferences", "repeat_step", "list_saved_recipes"}

RECIPE = Recipe.model_validate({
    "title": "Vegetable Pulao",
    "servings": 2,
    "ingredients": [
        {"name": "basmati rice", "quantity": 1, "unit": "cup"},
        {"name": "ghee", "quantity": 2, "unit": "tbsp"},
        {"name": "cumin seeds", "quantity": 1, "unit": "tsp"},
        {"name": "cashews", "quantity": 10},
        {"name": "onion", "quantity": 1, "note": "sliced"},
        {"name": "garlic", "quantity": 2, "unit": "cloves"},
        {"name": "green chillies", "quantity": 2},
        {"name": "salt", "quantity": None},
        {"name": "water", "quantity": 2, "unit": "cups"},
    ],
    "steps": [
        {"instruction": "Soak the rice in water for 20 minutes, then drain."},
        {"instruction": "Heat the ghee and fry the cumin seeds and cashews until golden."},
        {"instruction": "Add the onion, garlic and green chillies and fry for 5 minutes."},
        {"instruction": "Add the rice, water and salt, cover and cook for 12 minutes."},
        {"instruction": "Turn off the heat and let it rest for 5 minutes."},
    ],
})
START_STEP = 2


@dataclass(frozen=True)
class Expect:
    tool: Optional[str] = None  # None = an answer, no state-changing tool
    args: dict = field(default_factory=dict)  # must match (case-insensitive substring for strings)
    check: Optional[Callable[[CookingSession], bool]] = None


ANSWER = Expect()


def _names(s: CookingSession) -> list[str]:
    return [i.name.lower() for i in s.recipe.ingredients]


def _gone(word: str) -> Callable[[CookingSession], bool]:
    return lambda s: not any(word in n for n in _names(s))


def _has(word: str) -> Callable[[CookingSession], bool]:
    return lambda s: any(word in n for n in _names(s))


def _qty(word: str, op: Callable[[float], bool]) -> Callable[[CookingSession], bool]:
    def check(s: CookingSession) -> bool:
        ing = next((i for i in s.recipe.ingredients if word in i.name.lower()), None)
        return ing is not None and ing.quantity is not None and op(ing.quantity)
    return check


@dataclass(frozen=True)
class Case:
    text: str
    kind: str
    accept: tuple[Expect, ...]


UNDERSTANDING_CASES: list[Case] = [
    # questions about the dish: answer, stay put
    Case("how can I tell when the rice is cooked", "question", (ANSWER,)),
    Case("would brown rice work here", "question", (ANSWER,)),
    Case("what's the point of soaking the rice first", "question", (ANSWER,)),
    Case("is this going to be very hot", "question", (ANSWER,)),
    Case("how much ghee goes in", "question", (ANSWER, Expect("list_ingredients"))),
    Case("how many people does this feed", "question", (ANSWER,)),
    Case("I've run out of ghee, what else could I use", "question",
         (ANSWER, Expect("substitute_ingredient", {"ingredient": "ghee"}))),
    # changes: change the recipe, keep the step
    Case("drop the cashews, I don't like them", "change",
         (Expect("change_recipe", {"action": "remove_ingredient"}, _gone("cashew")),)),
    Case("throw in a handful of peas too", "change",
         (Expect("change_recipe", {"action": "add_ingredient"}, _has("pea")),)),
    Case("can you make it milder, the kids are eating", "change",
         (Expect("change_recipe", {"action": "change_amount"}, _qty("chilli", lambda q: q < 2)),
          Expect("change_recipe", {"action": "remove_ingredient"}, _gone("chilli")))),
    Case("put twice as much garlic in", "change",
         (Expect("change_recipe", {"action": "change_amount"}, _qty("garlic", lambda q: q >= 4)),)),
    Case("I'll use butter in place of the ghee", "change",
         (Expect("substitute_ingredient", {"ingredient": "ghee"}, _has("butter")),)),
    Case("no onion for me today", "change",
         (Expect("change_recipe", {"action": "remove_ingredient"}, _gone("onion")),)),
    # commands said loosely
    Case("alright, the cashews look golden", "command", (Expect("next_step"),)),
    Case("hang on, what did the last step say", "command", (Expect("previous_step"), ANSWER)),
    Case("give me a two minute timer for the cashews", "command",
         (Expect("set_timer", {"minutes": 2}), Expect("start_step_timer"))),
    Case("we're actually cooking for four now", "command", (Expect("scale_servings", {"servings": 4}),)),
    Case("are any timers running", "command", (Expect("list_timers"), ANSWER)),
    # chit-chat
    Case("this smells amazing", "chat", (ANSWER,)),
    Case("my mum used to make this every sunday", "chat", (ANSWER,)),
]


async def _fresh_agent(llm) -> CookingAgent:
    prefs = PreferenceStore(":memory:")
    prefs.mark_setup_done()
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(ms_per_char=0), SimulatedSink(), AgentConfig(barge_in=False),
                         recipes=RecipeStore(":memory:"), preferences=prefs)
    agent.tools._loaded(RECIPE.model_copy(deep=True), "generated", None, "vegetable pulao")
    for _ in range(START_STEP):
        await agent.submit_text("next")  # instant command; leaves the real "ready?" question pending
    assert agent.session.step_index == START_STEP - 1
    return agent


def _args_match(want: dict, got: dict) -> bool:
    for k, v in want.items():
        g = got.get(k)
        if isinstance(v, str):
            if not isinstance(g, str) or v.lower() not in g.lower():
                return False
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                if abs(float(g) - v) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif g != v:
            return False
    return True


def judge(case: Case, calls: list[ToolCall], before: dict, session: CookingSession, step_before: int,
          succeeded: Optional[list[ToolCall]] = None) -> bool:
    """True if what happened is one of the case's accepted outcomes. A tool
    only counts if it ran without an error (``succeeded``)."""
    changing = [c for c in calls if c.name not in READ_ONLY]
    ok_calls = calls if succeeded is None else succeeded
    for exp in case.accept:
        if exp.tool is None:
            if not changing and session.step_index == step_before and session.recipe.model_dump() == before:
                return True
            continue
        match = next((c for c in ok_calls if c.name == exp.tool and _args_match(exp.args, c.args or {})), None)
        if match is None:
            continue
        moves = exp.tool in ("next_step", "previous_step")
        if not moves and session.step_index != step_before:
            continue
        if exp.check is not None and not exp.check(session):
            continue
        return True
    return False


async def _run_once(llm, case: Case) -> dict:
    agent = await _fresh_agent(llm)
    calls: list[ToolCall] = []
    succeeded: list[ToolCall] = []
    dispatch = agent.tools.dispatch

    async def recording(call: ToolCall):
        calls.append(call)
        result = await dispatch(call)
        if "error" not in result.result:
            succeeded.append(call)
        return result

    agent.tools.dispatch = recording  # type: ignore[method-assign]
    before, step_before = agent.session.recipe.model_dump(), agent.session.step_index
    t0 = time.perf_counter()
    error = None
    try:
        await agent.submit_text(case.text)
    except Exception as e:  # count it as wrong, keep going
        error = f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - t0
    ok = error is None and judge(case, calls, before, agent.session, step_before, succeeded)
    reply = " ".join(t for t in agent.tts.spoken[-3:])
    await agent.timers.aclose()
    return {"ok": ok, "tools": [f"{c.name}({c.args}){'' if c in succeeded else ' -> error'}" for c in calls],
            "reply": reply, "seconds": round(seconds, 2),
            "error": error}


async def run_understanding_eval(llm, cases: list[Case] = UNDERSTANDING_CASES, runs: int = 3,
                                 progress=print) -> dict:
    results = []
    for case in cases:
        tries = [await _run_once(llm, case) for _ in range(runs)]
        passed = sum(t["ok"] for t in tries)
        r = {"text": case.text, "kind": case.kind, "passed": passed, "runs": runs, "tries": tries}
        results.append(r)
        progress(_line(r))
    by_kind = {}
    for kind in KINDS:
        rs = [r for r in results if r["kind"] == kind]
        if rs:
            by_kind[kind] = round(sum(r["passed"] for r in rs) / sum(r["runs"] for r in rs), 3)
    total = sum(r["runs"] for r in results)
    secs = sorted(t["seconds"] for r in results for t in r["tries"])
    summary = {
        "model": getattr(llm, "model", getattr(llm, "name", "?")),
        "cases": len(results),
        "runs": runs,
        "accuracy": round(sum(r["passed"] for r in results) / total, 3) if total else 0.0,
        "by_kind": by_kind,
        "median_turn_s": secs[len(secs) // 2] if secs else None,
    }
    return {"cases": results, "summary": summary}


def _line(r: dict) -> str:
    status = "PASS" if r["passed"] == r["runs"] else ("FAIL" if r["passed"] == 0 else "SOME")
    last_bad = next((t for t in r["tries"] if not t["ok"]), None)
    extra = f"  got: {', '.join(last_bad['tools']) or 'answer'}" if last_bad else ""
    return f"  {status:<4} {r['passed']}/{r['runs']} [{r['kind']}] {r['text']}{extra}"


def format_understanding_report(report: dict) -> str:
    s = report["summary"]
    kinds = ", ".join(f"{k} {v:.0%}" for k, v in s["by_kind"].items())
    return (f"understanding test ({s['model']}): {s['accuracy']:.0%} right over {s['cases']} cases x "
            f"{s['runs']} runs ({kinds}); median turn {s['median_turn_s']} s")
