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
    out = await call(tb, "get_recipe", dish="pulao")  # applied to the next recipe in code
    assert {"more ghee (you usually like more)", "less cumin seeds (you usually like less)"} <= set(out["tailored_to"])
    assert "2 and 1/4 tbsp ghee" in tb.session.ingredient_lines() and "1/2 tsp cumin seeds" in tb.session.ingredient_lines()


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


# ------------------------------------------------------------ review fixes I3, I11, I7
from cooking_agent.preferences import usual_changes  # noqa: E402


def test_name_variants_count_as_one_habit(prefs):
    for load, name in enumerate(["onions", "onion", "red onion"]):
        prefs.record_adjustment("remove", name, load_id=load)
    for load, to in enumerate(["unsalted butter", "butter"]):
        prefs.record_adjustment("swap", "ghee", to, load_id=10 + load)
    learned = prefs.learned()
    assert learned["usually_leaves_out"] == ["onion"]
    assert learned["usually_swaps"] == [{"from": "ghee", "to": "butter"}]


async def test_forget_preference_drops_a_learned_habit(tmp_path, prefs, recipe):
    for load in (1, 2):
        prefs.record_adjustment("swap", "ghee", "butter", load_id=load)
        prefs.record_adjustment("remove", "cashews", load_id=load)
    tb = make_box(tmp_path, prefs, recipe)
    out = await call(tb, "forget_preference", value="butter")
    assert "error" not in out and any("butter" in f for f in out["forgot"])
    learned = prefs.learned()
    assert "usually_swaps" not in learned and learned["usually_leaves_out"] == ["cashews"]


def test_opposite_habits_the_latest_wins(prefs):
    for load in (1, 2):
        prefs.record_adjustment("remove", "cashews", load_id=load)
    for load in (3, 4):
        prefs.record_adjustment("add", "cashews", load_id=load)
    for load in (5, 6):
        prefs.record_adjustment("swap", "ghee", "butter", load_id=load)
    for load in (7, 8):
        prefs.record_adjustment("swap", "butter", "ghee", load_id=load)
    learned = prefs.learned()
    assert learned.get("usually_adds") == ["cashews"] and "usually_leaves_out" not in learned
    assert learned["usually_swaps"] == [{"from": "butter", "to": "ghee"}]


def test_more_of_something_a_condition_limits_is_not_passed_on(prefs):
    prefs.remember("condition", "high blood pressure")
    prefs.remember("condition", "diabetes")
    for load in (1, 2):
        prefs.record_adjustment("more", "salt", load_id=load)
        prefs.record_adjustment("more", "sugar", load_id=load)
        prefs.record_adjustment("more", "garlic", load_id=load)
    assert usual_changes(prefs.profile())["more"] == ["garlic"]
    assert "more salt" not in describe_for_recipe(prefs.profile())


def test_nut_allergy_also_keeps_peanut_habits_out(prefs):
    prefs.remember("allergy", "nuts")
    for load in (1, 2):
        prefs.record_adjustment("swap", "oil", "peanut oil", load_id=load)
    assert usual_changes(prefs.profile())["swaps"] == []


# ------------------------------------------------------------ review fix C1: habits applied in code
def _habits(prefs, *changes):
    for load in (101, 102):
        for kind, original, *replacement in changes:
            prefs.record_adjustment(kind, original, replacement[0] if replacement else None, load_id=load)


async def test_habits_are_applied_in_code_even_if_the_writer_ignores_them(tmp_path, prefs, recipe):
    _habits(prefs, ("swap", "ghee", "butter"), ("remove", "cumin seeds"), ("more", "basmati rice"))
    before = prefs.adjustments()
    tb = make_box(tmp_path, prefs, recipe)  # the fake writer returns ghee and cumin regardless
    out = await call(tb, "get_recipe", dish="pulao")
    lines = tb.session.ingredient_lines()
    assert "1 and 1/2 cups basmati rice, rinsed" in lines and not any("cumin" in x for x in lines)
    assert any(x.startswith("1 and 1/2 tbsp butter") for x in lines)
    assert {"butter instead of ghee (your usual swap)", "no cumin seeds (you usually leave it out)",
            "more basmati rice (you usually like more)"} <= set(out["tailored_to"])
    assert prefs.adjustments() == before  # applying a habit isn't new evidence for it


async def test_only_habits_that_were_applied_are_claimed(tmp_path, prefs, recipe):
    _habits(prefs, ("swap", "paneer", "tofu"), ("remove", "cashews"))
    out = await call(make_box(tmp_path, prefs, recipe), "get_recipe", dish="pulao")
    assert not any("tofu" in t or "cashew" in t for t in out.get("tailored_to", []))


async def test_an_ingredient_the_cook_asked_for_is_not_left_out(tmp_path, prefs, recipe):
    _habits(prefs, ("remove", "cumin seeds"))
    tb = make_box(tmp_path, prefs, recipe)
    await call(tb, "get_recipe", dish="cumin rice")
    assert "cumin seeds" in [i.name for i in tb.session.recipe.ingredients]


async def test_an_ambiguous_habit_is_skipped_not_an_error(tmp_path, prefs, recipe):
    two_rices = recipe.model_copy(update={"ingredients": [*recipe.ingredients,
                                                          recipe.ingredients[0].model_copy(update={"name": "rice flour"})]})
    _habits(prefs, ("remove", "rice"))
    out = await call(make_box(tmp_path, prefs, two_rices), "get_recipe", dish="pulao")
    assert "error" not in out and out["ingredient_count"] == 5


def test_more_and_less_are_left_to_the_code_not_the_writer(prefs):
    _habits(prefs, ("more", "garlic"), ("less", "chillies"))
    text = describe_for_recipe(prefs.profile())
    assert "garlic" not in text and "chilli" not in text


def test_undo_within_one_recipe_cancels_out(prefs):
    for load in (1, 2):  # took the cashews out, then put them back, in the same recipe
        prefs.record_adjustment("remove", "cashews", load_id=load)
        prefs.record_adjustment("add", "cashews", load_id=load)
    learned = prefs.learned()
    assert "usually_leaves_out" not in learned and "usually_adds" not in learned


def test_two_swaps_from_the_same_ingredient_keep_the_latest(prefs):
    for load in (1, 2):
        prefs.record_adjustment("swap", "ghee", "butter", load_id=load)
    for load in (3, 4):
        prefs.record_adjustment("swap", "ghee", "oil", load_id=load)
    assert prefs.learned()["usually_swaps"] == [{"from": "ghee", "to": "oil"}]


def test_latest_wins_even_within_the_same_clock_tick(prefs, monkeypatch):
    import cooking_agent.preferences as p
    monkeypatch.setattr(p.time, "time", lambda: 1000.0)  # Windows' clock often gives equal times
    for load in (1, 2):
        prefs.record_adjustment("more", "garlic", load_id=load)
    for load in (3, 4):
        prefs.record_adjustment("less", "garlic", load_id=load)
    assert prefs.learned()["usually_less"] == ["garlic"] and "usually_more" not in prefs.learned()
