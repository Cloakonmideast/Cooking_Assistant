"""Stated preferences, learned habits, and recipes tailored to them."""

import pytest

from cooking_agent.cli import main
from cooking_agent.preferences import PreferenceStore, describe_for_recipe, diets_in
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


@pytest.fixture
def prefs(tmp_path):
    return PreferenceStore(tmp_path / "book.db")


def test_diets_in():
    assert diets_in("vegetarian, no nuts") == {"vegetarian"}
    assert diets_in("veg biryani") == {"vegetarian"}
    assert diets_in("vegan gluten free") == {"vegan", "gluten-free"}
    assert diets_in("vegetable soup") == set()  # 'veg' only as a whole word
    assert diets_in(None) == set()


def test_stated_preferences_remember_and_forget(prefs):
    assert prefs.remember("diet", "Vegetarian")
    assert not prefs.remember("diet", "vegetarian ")  # normalised duplicate
    prefs.remember("allergy", "peanuts")
    prefs.remember("dislike", "mushrooms")
    assert prefs.profile() == {"allergy": ["peanuts"], "diet": ["vegetarian"], "dislike": ["mushrooms"]}
    assert [r["value"] for r in prefs.forget("mushroom")] == ["mushrooms"]
    with pytest.raises(ValueError):
        prefs.remember("horoscope", "leo")


def test_learns_diet_and_usual_servings_only_with_enough_evidence(prefs):
    prefs.record_request("dal", "vegetarian", 2)
    prefs.record_request("vegetarian lasagne", None, 2)
    assert prefs.learned() == {}  # two times is not a habit yet
    prefs.record_request("paneer tikka", "veg", 2)
    prefs.record_request("pasta", None, 4)
    assert prefs.learned() == {"usually_asks_for": ["vegetarian"], "usually_cooks_for": 2}
    # once stated, the learned copy is not repeated
    prefs.remember("diet", "vegetarian")
    assert "learned_from_history" not in prefs.profile() or \
        "usually_asks_for" not in prefs.profile()["learned_from_history"]


def test_learns_often_used_ingredients_from_recipes_actually_cooked(prefs):
    prefs.record_cooked(None, "Dal", ["toor dal", "Ghee", "salt", "fresh coriander"])
    prefs.record_cooked(None, "Jeera Rice", ["basmati rice", "ghee", "cumin seeds", "chopped coriander", "water"])
    assert set(prefs.learned()["often_cooks_with"]) == {"coriander", "ghee"}  # adjectives dropped, salt/water ignored
    prefs.forget_history()
    assert prefs.learned() == {}


def test_recipe_writer_instructions():
    text = describe_for_recipe({"allergy": ["peanuts"], "diet": ["vegetarian"], "pantry": ["paneer"],
                                "learned_from_history": {"often_cooks_with": ["ghee"]}})
    assert "ALLERGIES, never include" in text and "peanuts" in text
    assert "Diet: vegetarian" in text and "paneer" in text and "ghee" in text
    assert describe_for_recipe({}) == ""


# ------------------------------------------------------------------ tools
@pytest.fixture
def box(tmp_path, recipe):
    path = tmp_path / "book.db"
    llm = ScriptedLLM(recipe=recipe)
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), llm,
                   store=RecipeStore(path), prefs=PreferenceStore(path))


async def call(tb, name, /, **args):
    return (await tb.dispatch(ToolCall(name, args))).result


async def test_new_recipes_are_tailored_and_the_assistant_is_told(box):
    await call(box, "remember_preference", category="diet", value="vegetarian")
    await call(box, "remember_preference", category="allergy", value="cashews")
    out = await call(box, "get_recipe", dish="biryani")
    prompt = box.llm.prompts[-1]
    assert "Diet: vegetarian" in prompt and "never include" in prompt and "cashews" in prompt
    assert out["tailored_to"] == ["no cashews (allergy)", "vegetarian"]
    assert (await call(box, "get_preferences"))["diet"] == ["vegetarian"]
    assert (await call(box, "forget_preference", value="cashews"))["forgot"] == ["cashews"]
    assert "error" in await call(box, "forget_preference", value="olives")
    assert "error" in await call(box, "remember_preference", category="mood", value="happy")


async def test_usual_servings_are_used_when_not_said(box):
    for _ in range(3):
        await call(box, "get_recipe", dish="dal", servings=6)
    out = await call(box, "get_recipe", dish="jeera rice")
    assert out["servings"] == 6 and "usually cook for" in out["servings_note"]
    assert "Servings: 6" in box.llm.prompts[-1]


async def test_cooking_is_recorded_once_at_step_one(box):
    for _ in range(2):
        await call(box, "get_recipe", dish="jeera rice")
        await call(box, "next_step")
        await call(box, "next_step")
        await call(box, "previous_step")
    used = box.prefs.learned()["often_cooks_with"]
    assert set(used) == {"basmati rice", "cumin seeds", "ghee"}  # 2 recipes each, salt ignored
    # just loading a recipe without cooking it teaches nothing
    box.prefs.forget_history()
    await call(box, "get_recipe", dish="jeera rice")
    await call(box, "get_recipe", dish="jeera rice")
    assert "often_cooks_with" not in box.prefs.learned()


def test_cli_prefs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECIPE_DB", str(tmp_path / "book.db"))
    assert main(["prefs", "add", "pantry", "paneer"]) == 0
    assert main(["prefs"]) == 0
    assert "pantry     paneer" in capsys.readouterr().out
    assert main(["prefs", "remove", "paneer"]) == 0
    assert main(["prefs", "remove", "paneer"]) == 1
