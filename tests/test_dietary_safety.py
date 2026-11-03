"""Allergies and health conditions: first-run setup, recipe checks, warnings."""

import io

import pytest

from cooking_agent.agent import SETUP_QUESTION
from cooking_agent.cli import main
from cooking_agent.dietary import allergen_words, find_conflicts
from cooking_agent.preferences import PreferenceStore, describe_for_recipe
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.recipe import Recipe
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


# ------------------------------------------------------------------ checker
def flagged(names, allergies=(), conditions=()):
    return [c.ingredient for c in find_conflicts(names, allergies, conditions)]


@pytest.mark.parametrize("names,allergies,conditions,expected", [
    (["whole milk", "paneer", "butter", "ghee"], [], ["lactose intolerance"], ["whole milk", "paneer", "butter"]),
    (["coconut milk", "almond milk", "lactose-free yogurt", "peanut butter"], [], ["lactose intolerance"], []),
    (["maida", "soy sauce", "bread crumbs", "semolina"], [], ["celiac disease"],
     ["maida", "soy sauce", "bread crumbs", "semolina"]),
    (["rice flour", "gram flour", "tamari", "cornflour"], [], ["celiac disease"], []),
    (["cashews", "walnuts", "nutmeg", "coconut"], ["tree nuts"], [], ["cashews", "walnuts"]),
    (["peanut oil", "roasted groundnuts"], ["peanuts"], [], ["peanut oil", "roasted groundnuts"]),
    (["eggs", "eggplant", "mayonnaise"], ["egg"], [], ["eggs", "mayonnaise"]),
    (["prawns", "salmon"], ["shellfish"], [], ["prawns"]),
    (["coconut milk", "milk"], ["milk"], [], ["milk"]),
    (["kiwis", "apple"], ["kiwi"], [], ["kiwis"]),
    (["sugar", "white rice"], [], ["diabetes"], []),  # guidance only, no hard ingredient list
])
def test_find_conflicts(names, allergies, conditions, expected):
    assert flagged(names, allergies, conditions) == expected


def test_allergen_families():
    assert "cashew" in allergen_words("nut allergy")
    assert allergen_words("Peanuts") == ["peanut", "groundnut", "monkey nut", "peanut butter"]


def test_recipe_writer_gets_condition_rules_and_no_conflicting_pantry():
    text = describe_for_recipe({"condition": ["lactose intolerance"], "pantry": ["paneer", "ghee"]})
    assert "HEALTH CONDITIONS" in text and "no milk, cream" in text
    assert "prefer these where they fit: ghee." in text  # paneer dropped


# ------------------------------------------------------------------ tools
class SequenceLLM(ScriptedLLM):
    """Returns the given recipes in order from generate_structured."""

    def __init__(self, recipes):
        super().__init__(recipe=recipes[0])
        self.queue = list(recipes)

    async def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def dish(*ingredients):
    return Recipe.model_validate({"title": "Kheer", "servings": 2,
                                  "ingredients": [{"name": n, "quantity": 1} for n in ingredients],
                                  "steps": [{"instruction": "Cook."}]})


def make_box(tmp_path, llm):
    path = tmp_path / "book.db"
    prefs = PreferenceStore(path)
    prefs.remember("condition", "lactose intolerance")
    prefs.remember("allergy", "cashews")
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), llm,
                   store=RecipeStore(path), prefs=prefs)


async def call(tb, name, /, **args):
    return (await tb.dispatch(ToolCall(name, args))).result


async def test_conflicting_generated_recipe_is_sent_back_and_fixed(tmp_path):
    llm = SequenceLLM([dish("milk", "rice", "cashews"), dish("coconut milk", "rice", "raisins")])
    box = make_box(tmp_path, llm)
    out = await call(box, "get_recipe", dish="kheer")
    assert len(llm.prompts) == 2
    assert "must not have" in llm.prompts[1] and "milk (conflicts with lactose intolerance)" in llm.prompts[1]
    assert "warnings" not in out
    assert out["tailored_to"] == ["no cashews (allergy)", "suitable for lactose intolerance"]


async def test_recipe_still_conflicting_after_retry_comes_with_warnings(tmp_path):
    box = make_box(tmp_path, SequenceLLM([dish("milk", "rice")]))
    out = await call(box, "get_recipe", dish="kheer")
    assert out["warnings"] == ["contains milk (conflicts with lactose intolerance)"]


async def test_own_recipe_is_not_changed_but_warned_about(tmp_path):
    box = make_box(tmp_path, SequenceLLM([dish("rice")]))
    box.store.add(dish("milk", "rice", "cashews"))
    out = await call(box, "get_recipe", dish="kheer")
    assert out["source"] == "your recipe book"
    assert out["warnings"] == ["contains milk (conflicts with lactose intolerance)",
                               "contains cashews (conflicts with cashews allergy)"]
    assert [i.name for i in box.session.recipe.ingredients] == ["milk", "rice", "cashews"]


async def test_substitution_into_an_allergen_is_flagged(tmp_path):
    box = make_box(tmp_path, SequenceLLM([dish("coconut milk", "rice")]))
    await call(box, "get_recipe", dish="kheer")
    out = await call(box, "substitute_ingredient", ingredient="coconut milk", replacement="cream")
    assert out["warnings"] == ["contains cream (conflicts with lactose intolerance)"]


async def test_finish_setup(tmp_path):
    box = make_box(tmp_path, SequenceLLM([dish("rice")]))
    assert box.prefs.needs_setup()
    box.session.setup_pending = True
    assert "first_time_setup" in box.session.snapshot()
    out = await call(box, "finish_setup")
    assert out["profile"]["condition"] == ["lactose intolerance"]
    assert not box.prefs.needs_setup() and not box.session.setup_pending
    box.prefs.reset_setup()
    assert box.prefs.needs_setup()


# ------------------------------------------------------------------ agent + CLI
async def test_first_run_question_is_spoken_and_answer_saved(tmp_path, recipe):
    import asyncio

    from cooking_agent.agent import AgentConfig, CookingAgent
    from cooking_agent.audio import SimulatedSink
    from cooking_agent.providers.fake import QueueSTT, ToneTTS

    llm = ScriptedLLM(recipe=recipe, script={"celiac": [
        ToolCall("remember_preference", {"category": "condition", "value": "celiac disease"}),
        ToolCall("finish_setup", {})]})
    prefs = PreferenceStore(tmp_path / "p.db")
    tts = ToneTTS()
    agent = CookingAgent(QueueSTT(), llm, tts, SimulatedSink(), AgentConfig(), preferences=prefs)
    announcer = asyncio.create_task(agent._announcer())
    assert agent.begin_setup()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if tts.spoken:
            break
    assert tts.spoken == [SETUP_QUESTION]
    await agent.submit_text("I have celiac disease")
    assert prefs.profile() == {"condition": ["celiac disease"]}
    assert not prefs.needs_setup()
    announcer.cancel()


def test_cli_setup_by_typing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECIPE_DB", str(tmp_path / "book.db"))
    monkeypatch.setattr("sys.stdin", io.StringIO("peanuts\nlactose intolerance, diabetes\nnone\n\npaneer\n"))
    assert main(["setup"]) == 0
    prefs = PreferenceStore(tmp_path / "book.db")
    assert not prefs.needs_setup()
    assert prefs.profile() == {"allergy": ["peanuts"], "condition": ["lactose intolerance", "diabetes"],
                               "pantry": ["paneer"]}
