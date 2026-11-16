"""The understanding test (Phase 2, Task 12), scored with fake models."""

import pytest

from cooking_agent.agent import SYSTEM_PROMPT
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox
from cooking_agent.understanding_eval import (KINDS, UNDERSTANDING_CASES, Case, Expect, run_understanding_eval,
                                              ANSWER, _gone)

QUESTION = Case("how can I tell when the rice is cooked", "question", (ANSWER,))
CHANGE = Case("drop the cashews, I don't like them", "change",
              (Expect("change_recipe", {"action": "remove_ingredient"}, _gone("cashew")),))
NEXT = Case("okay that's finished, what now", "command", (Expect("next_step"),))


def quiet(_):
    pass


async def test_understanding_eval_scores_answer_vs_tool():
    talker = ScriptedLLM(script={"": ["It's done when the water is gone."]})
    report = await run_understanding_eval(talker, [QUESTION, CHANGE], runs=2, progress=quiet)
    s = report["summary"]
    assert [r["passed"] for r in report["cases"]] == [2, 0]
    assert s["accuracy"] == 0.5 and s["by_kind"] == {"question": 1.0, "change": 0.0}


async def test_a_change_must_really_change_the_recipe():
    right = ScriptedLLM(script={"": [ToolCall("change_recipe", {"action": "remove_ingredient",
                                                                 "ingredient": "cashews"})]})
    wrong = ScriptedLLM(script={"": [ToolCall("change_recipe", {"action": "remove_ingredient",
                                                                 "ingredient": "cumin"})]})
    assert (await run_understanding_eval(right, [CHANGE], runs=1, progress=quiet))["summary"]["accuracy"] == 1.0
    assert (await run_understanding_eval(wrong, [CHANGE], runs=1, progress=quiet))["summary"]["accuracy"] == 0.0


async def test_understanding_eval_step_must_not_move_on_questions():
    mover = ScriptedLLM(script={"": [ToolCall("next_step", {})]})
    report = await run_understanding_eval(mover, [QUESTION, NEXT], runs=1, progress=quiet)
    assert [r["passed"] for r in report["cases"]] == [0, 1]


async def test_a_change_that_also_moves_the_step_is_wrong():
    both = ScriptedLLM(script={"": [ToolCall("change_recipe", {"action": "remove_ingredient",
                                                                "ingredient": "cashews"}),
                                    ToolCall("next_step", {})]})
    assert (await run_understanding_eval(both, [CHANGE], runs=1, progress=quiet))["summary"]["accuracy"] == 0.0


async def test_eval_starts_on_step_2_with_the_ready_question_pending():
    yes = Case("yes", "command", (Expect("next_step"),))  # the instant "yes" answers the real question
    report = await run_understanding_eval(ScriptedLLM(), [yes], runs=1, progress=quiet)
    assert report["summary"]["accuracy"] == 1.0


def test_cases_are_well_formed_and_not_copied_into_the_prompt():
    assert len(UNDERSTANDING_CASES) >= 20
    assert {c.kind for c in UNDERSTANDING_CASES} == set(KINDS)
    box = ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), ScriptedLLM())
    descriptions = " ".join(t.description + str(t.parameters) for t in box.specs())
    assert "leave something out" in descriptions  # the change_recipe description is included
    prompt = SYSTEM_PROMPT.lower() + descriptions.lower()
    for c in UNDERSTANDING_CASES:
        assert c.text.lower() not in prompt, c.text


@pytest.mark.parametrize("case", UNDERSTANDING_CASES, ids=lambda c: c.text)
def test_every_case_accepts_something(case):
    assert case.accept


async def test_a_tool_call_that_failed_does_not_count():
    swap = Case("I've run out of ghee", "question", (Expect("substitute_ingredient", {"ingredient": "ghee"}),))
    failing = ScriptedLLM(script={"": [ToolCall("substitute_ingredient", {"ingredient": "ghee", "replacement": ""})]})
    working = ScriptedLLM(script={"": [ToolCall("substitute_ingredient", {"ingredient": "ghee",
                                                                         "replacement": "butter"})]})
    assert (await run_understanding_eval(failing, [swap], runs=1, progress=quiet))["summary"]["accuracy"] == 0.0
    assert (await run_understanding_eval(working, [swap], runs=1, progress=quiet))["summary"]["accuracy"] == 1.0


# ------------------------------------------------------------ review fix I10: the eval must not flatter
from cooking_agent.recipe import Recipe  # noqa: E402
from cooking_agent.state import CookingSession as _Session  # noqa: E402
from cooking_agent.understanding_eval import _gone as gone  # noqa: E402


class BrokenLLM(ScriptedLLM):
    async def stream(self, system, history, tools):
        raise ConnectionError("ollama not running")
        yield  # pragma: no cover


async def test_silence_or_an_error_is_not_an_answer():
    empty = ScriptedLLM(script={"": [""]})
    assert (await run_understanding_eval(empty, [QUESTION], runs=1, progress=quiet))["summary"]["accuracy"] == 0.0
    assert (await run_understanding_eval(BrokenLLM(), [QUESTION], runs=1, progress=quiet))["summary"]["accuracy"] == 0.0


def test_a_removal_must_also_be_right_in_the_spoken_steps():
    s = _Session()
    s.load_recipe(Recipe.model_validate({"title": "T", "servings": 1, "ingredients": [{"name": "rice"},
                  {"name": "cashews"}], "steps": [{"instruction": "Fry the cashews."}]}))
    s.recipe.ingredients.pop(1)  # gone from the list, but step 1 would still say to fry them
    assert not gone("cashew")(s)
    s.removed.append("cashews")  # the app's own removal adds "Leave out the cashews."
    assert gone("cashew")(s)


def test_new_cases_cover_the_review_findings():
    texts = {c.text for c in UNDERSTANDING_CASES}
    assert {"could I leave out the garlic", "use half a cup less water", "leave the basil out"} <= texts
    assert len({c.recipe.title for c in UNDERSTANDING_CASES}) == 2


def test_prompt_examples_are_not_the_eval_topics():
    p = SYSTEM_PROMPT.lower()
    assert "brown rice" not in p and "soak" not in p and "chillies" not in p
