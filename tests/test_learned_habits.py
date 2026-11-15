"""Learning what the cook changes, and using it (Phase 2, Task 10)."""

import pytest

from cooking_agent.cli import main
from cooking_agent.preferences import PreferenceStore, describe_for_recipe
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.recipe import Recipe
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


@pytest.fixture
def prefs(tmp_path):
    p = PreferenceStore(tmp_path / "book.db")
    yield p
    p.close()


def make_box(tmp_path, prefs, recipe, store=False):
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), ScriptedLLM(recipe=recipe),
                   store=RecipeStore(tmp_path / "book.db") if store else None, prefs=prefs)


async def call(tb, tool_name, /, **args):
    return (await tb.dispatch(ToolCall(tool_name, args))).result


async def cook_twice(tb, tool_name, **args):
    """The same change in two separate recipes."""
    for _ in range(2):
        await call(tb, "get_recipe", dish="jeera rice")
        out = await call(tb, tool_name, **args)
        assert "error" not in out, out


# ------------------------------------------------------------ evidence
def test_one_swap_is_not_a_habit(prefs):
    prefs.record_adjustment("swap", "ghee", "butter", load_id=1)
    assert "usually_swaps" not in prefs.learned()


def test_same_change_twice_in_one_recipe_is_not_a_habit(prefs):
    prefs.record_adjustment("remove", "cashews", load_id=7)
    prefs.record_adjustment("remove", "cashews", load_id=7)
    assert "usually_leaves_out" not in prefs.learned()


def test_every_change_is_saved_even_before_it_is_a_habit(prefs):
    prefs.record_adjustment("add", "peas", load_id=1)
    assert prefs.adjustments() == [{"kind": "add", "original": "peas", "replacement": None, "times": 1}]


# ------------------------------------------------------------ through the tools
async def test_two_swaps_become_a_habit_and_reach_the_recipe_writer(tmp_path, prefs, recipe):
    tb = make_box(tmp_path, prefs, recipe)
    await cook_twice(tb, "substitute_ingredient", ingredient="ghee", replacement="butter")
    assert prefs.learned()["usually_swaps"] == [{"from": "ghee", "to": "butter"}]
    out = await call(tb, "get_recipe", dish="pulao")
    assert "usually swaps ghee for butter" in tb.llm.prompts[-1]
    assert "butter instead of ghee (your usual swap)" in out["tailored_to"]


async def test_removed_twice_is_left_out(tmp_path, prefs, recipe):
    tb = make_box(tmp_path, prefs, recipe)
    await cook_twice(tb, "change_recipe", action="remove_ingredient", ingredient="cumin")
    assert prefs.learned()["usually_leaves_out"] == ["cumin seeds"]
    await call(tb, "get_recipe", dish="pulao")
    assert "usually leaves out cumin seeds" in tb.llm.prompts[-1]


async def test_added_twice_is_learned(tmp_path, prefs, recipe):
    tb = make_box(tmp_path, prefs, recipe)
    await cook_twice(tb, "change_recipe", action="add_ingredient", ingredient="green peas", amount="1 cup")
    assert prefs.learned()["usually_adds"] == ["green peas"]


async def test_more_less_learned(tmp_path, prefs, recipe):
    tb = make_box(tmp_path, prefs, recipe)
    await cook_twice(tb, "change_recipe", action="change_amount", ingredient="ghee", amount="double")
    await cook_twice(tb, "change_recipe", action="change_amount", ingredient="cumin", amount="less")
    learned = prefs.learned()
    assert learned["usually_more"] == ["ghee"] and learned["usually_less"] == ["cumin seeds"]
    text = describe_for_recipe(prefs.profile())
    assert "likes more ghee" in text and "likes less cumin seeds" in text


async def test_forget_history_clears_adjustments(tmp_path, prefs, recipe):
    tb = make_box(tmp_path, prefs, recipe)
    await cook_twice(tb, "substitute_ingredient", ingredient="ghee", replacement="butter")
    prefs.forget_history()
    assert prefs.adjustments() == [] and "usually_swaps" not in prefs.learned()


def test_prefs_cli_shows_learned_changes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECIPE_DB", str(tmp_path / "book.db"))
    p = PreferenceStore(tmp_path / "book.db")
    for load in (1, 2):
        p.record_adjustment("swap", "ghee", "butter", load_id=load)
    p.close()
    assert main(["prefs"]) == 0
    assert "usually swaps: ghee to butter" in capsys.readouterr().out


# ------------------------------------------------------------ safety
def test_learned_swap_into_an_allergen_is_not_passed_on(prefs):
    prefs.remember("allergy", "peanuts")
    prefs.remember("diet", "vegan")
    for load in (1, 2):
        prefs.record_adjustment("swap", "oil", "peanut oil", load_id=load)
        prefs.record_adjustment("add", "paneer", load_id=load)
        prefs.record_adjustment("swap", "ghee", "coconut oil", load_id=load)
    habits = next(line for line in describe_for_recipe(prefs.profile()).splitlines() if "usually make" in line)
    assert "peanut oil" not in habits and "paneer" not in habits
    assert "usually swaps ghee for coconut oil" in habits


async def test_saved_recipe_offers_usual_changes_without_making_them(tmp_path, prefs, recipe):
    for load in (1, 2):
        prefs.record_adjustment("swap", "ghee", "butter", load_id=load)
        prefs.record_adjustment("remove", "cumin seeds", load_id=load)
    tb = make_box(tmp_path, prefs, recipe, store=True)
    tb.store.add(recipe.model_copy(update={"title": "Mum's Rice"}))
    out = await call(tb, "get_recipe", dish="mum's rice")
    assert out["source"] == "your recipe book"
    assert out["your_usual_changes"] == ["swap ghee for butter", "leave out cumin seeds"]
    assert [i.name for i in tb.session.recipe.ingredients] == ["basmati rice", "cumin seeds", "ghee", "salt"]


# ------------------------------------------------------------ learned diet: a heads-up, never a rewrite
def chicken_dish() -> Recipe:
    return Recipe.model_validate({"title": "Biryani", "servings": 2,
                                  "ingredients": [{"name": "chicken"}, {"name": "rice"}],
                                  "steps": [{"instruction": "Cook it."}]})


async def test_learned_diet_warns_without_rewrite(tmp_path, prefs):
    for dish in ("vegetarian pulao", "veg biryani", "vegetarian curry"):
        prefs.record_request(dish, None, None)
    tb = make_box(tmp_path, prefs, chicken_dish())
    out = await call(tb, "get_recipe", dish="biryani")
    assert len(tb.llm.prompts) == 1
    assert out["warnings"] == ["has chicken, and you usually cook vegetarian"]


async def test_learned_diet_quiet_when_the_dish_is_named(tmp_path, prefs):
    for dish in ("vegetarian pulao", "veg biryani", "vegetarian curry"):
        prefs.record_request(dish, None, None)
    out = await call(make_box(tmp_path, prefs, chicken_dish()), "get_recipe", dish="chicken biryani")
    assert "warnings" not in out


async def test_learned_vegetarian_does_not_flag_eggs_for_an_eggetarian(tmp_path, prefs, recipe):
    prefs.remember("diet", "eggetarian")
    for dish in ("veg pulao", "veg biryani", "veg curry"):
        prefs.record_request(dish, None, None)
    eggs = recipe.model_copy(update={"ingredients": [*recipe.ingredients, recipe.ingredients[0].model_copy(
        update={"name": "eggs", "quantity": 2, "unit": None, "note": None})]})
    tb = make_box(tmp_path, prefs, eggs)
    out = await call(tb, "get_recipe", dish="egg fried rice")
    assert "warnings" not in out
    assert "usually_asks_for" not in prefs.profile().get("learned_from_history", {})
