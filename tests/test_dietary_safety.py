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


# ------------------------------------------------------------------ diets (vegetarian, vegan, Jain)
from cooking_agent.dietary import diet_hint  # noqa: E402


def diet_flagged(names, diet):
    return [(c.ingredient, c.reason) for c in find_conflicts(names, diets=[diet])]


def test_vegetarian_flags_meat_fish_egg():
    out = diet_flagged(["chicken stock", "2 eggs", "fish sauce", "gelatin"], "vegetarian")
    assert [i for i, _ in out] == ["chicken stock", "2 eggs", "fish sauce", "gelatin"]
    assert {r for _, r in out} == {"vegetarian diet"}


def test_vegetarian_lookalikes_ok():
    assert diet_flagged(["chickpeas", "eggplant", "vegetable stock", "eggless mayonnaise", "paneer"],
                        "vegetarian") == []


def test_vegan_flags_dairy_ghee_honey():
    assert [i for i, _ in diet_flagged(["ghee", "honey", "paneer"], "vegan")] == ["ghee", "honey", "paneer"]
    assert diet_flagged(["coconut milk", "oat milk", "tofu"], "vegan") == []


def test_vegan_plant_exemption_does_not_hide_meat():
    # "rice" makes "rice milk" fine for the dairy rule, but must not excuse chicken
    assert [i for i, _ in diet_flagged(["chicken fried rice"], "vegan")] == ["chicken fried rice"]


def test_jain_flags_root_vegetables():
    assert [i for i, _ in diet_flagged(["onion", "garlic", "potato", "ginger"], "jain")] == \
        ["onion", "garlic", "potato", "ginger"]
    assert diet_flagged(["dry ginger powder", "tomato", "paneer"], "jain") == []


def test_diet_names_are_case_insensitive():
    assert diet_flagged(["beef"], "Vegetarian") == [("beef", "Vegetarian diet")]


def test_diet_hint():
    hint = diet_hint("vegan")
    assert hint.startswith("vegan (") and "dairy" in hint and "honey" in hint
    assert diet_hint("keto") == "keto"


def test_recipe_writer_gets_diet_rules():
    assert "no meat" in describe_for_recipe({"diet": ["vegetarian"]})


def diet_box(tmp_path, llm, diet=None, allergy=None):
    path = tmp_path / "book.db"
    prefs = PreferenceStore(path)
    if diet:
        prefs.remember("diet", diet)
    if allergy:
        prefs.remember("allergy", allergy)
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), llm,
                   store=RecipeStore(path), prefs=prefs)


async def test_generated_meat_rewritten_for_vegetarian(tmp_path):
    llm = SequenceLLM([dish("chicken", "rice"), dish("paneer", "rice")])
    out = await call(diet_box(tmp_path, llm, diet="vegetarian"), "get_recipe", dish="biryani")
    assert len(llm.prompts) == 2
    assert "chicken (conflicts with vegetarian diet)" in llm.prompts[1]
    assert "warnings" not in out


async def test_request_overrides_diet(tmp_path):
    llm = SequenceLLM([dish("chicken", "rice")])
    out = await call(diet_box(tmp_path, llm, diet="vegetarian"), "get_recipe", dish="chicken biryani")
    assert len(llm.prompts) == 1 and "warnings" not in out


async def test_one_off_vegan_constraint_checked(tmp_path):
    llm = SequenceLLM([dish("ghee", "rice"), dish("coconut oil", "rice")])
    await call(diet_box(tmp_path, llm), "get_recipe", dish="kheer", constraints="vegan")
    assert len(llm.prompts) == 2 and "ghee (conflicts with vegan diet)" in llm.prompts[1]


async def test_saved_recipe_diet_warning(tmp_path):
    box = diet_box(tmp_path, SequenceLLM([dish("rice")]), diet="vegetarian")
    box.store.add(dish("chicken", "rice").model_copy(update={"title": "Mum's Special"}))
    out = await call(box, "get_recipe", dish="mum's special")
    assert out["source"] == "your recipe book"
    assert out["warnings"] == ["contains chicken (conflicts with vegetarian diet)"]
    assert [i.name for i in box.session.recipe.ingredients] == ["chicken", "rice"]


async def test_allergy_never_overridden_by_request(tmp_path):
    llm = SequenceLLM([dish("peanuts", "jaggery"), dish("sesame", "jaggery")])
    await call(diet_box(tmp_path, llm, allergy="peanuts"), "get_recipe", dish="peanut chikki")
    assert len(llm.prompts) == 2


# ------------------------------------------------------------------ eggetarian (Phase 2, Task 8)
from cooking_agent.preferences import diets_in  # noqa: E402
from cooking_agent.tools import RememberArgs  # noqa: E402


def test_eggetarian_allows_eggs_not_meat():
    assert diet_flagged(["2 eggs", "mayonnaise", "paneer"], "eggetarian") == []
    assert [i for i, _ in diet_flagged(["chicken", "fish sauce", "gelatin"], "eggetarian")] == \
        ["chicken", "fish sauce", "gelatin"]


@pytest.mark.parametrize("said", [
    "eggetarian", "Eggitarian", "ovo-vegetarian", "ovo vegetarian", "vegetarian but I eat eggs",
    "I'm vegetarian, eggs are fine", "veg but eggs are ok", "vegetarian who eats eggs",
    "I eat eggs but no meat", "we eat eggs, no meat or fish",
])
def test_egg_eating_vegetarian_phrases(said):
    assert diets_in(said) == {"eggetarian"}


@pytest.mark.parametrize("said", ["vegetarian, I don't eat eggs", "vegetarian and I never eat eggs",
                                  "vegetarian, no eggs"])
def test_vegetarian_without_eggs_stays_vegetarian(said):
    assert diets_in(said) == {"vegetarian"}


def test_eggless_unchanged():
    assert diets_in("eggless chocolate cake") == {"eggless"}
    assert [i for i, _ in diet_flagged(["2 eggs", "mayonnaise", "eggplant", "flax seeds"], "eggless")] == \
        ["2 eggs", "mayonnaise"]


def test_eggetarian_recipe_writer_rule():
    text = describe_for_recipe({"diet": ["vegetarian but I eat eggs"]})
    assert "eggs are fine" in text and "no meat" in text


def test_remember_tool_knows_eggetarian():
    assert "eggetarian" in RememberArgs.model_json_schema()["properties"]["category"]["description"]


async def test_stated_eggetarian_with_veg_request_keeps_eggs(tmp_path):
    llm = SequenceLLM([dish("eggs", "rice")])
    out = await call(diet_box(tmp_path, llm, diet="eggetarian"), "get_recipe", dish="veg fried rice")
    assert len(llm.prompts) == 1 and "warnings" not in out


async def test_stated_eggetarian_still_rewrites_meat(tmp_path):
    llm = SequenceLLM([dish("chicken", "eggs", "rice"), dish("eggs", "rice")])
    await call(diet_box(tmp_path, llm, diet="vegetarian but eggs are ok"), "get_recipe", dish="fried rice")
    assert len(llm.prompts) == 2 and "chicken (conflicts with eggetarian diet)" in llm.prompts[1]


async def test_eggless_request_checks_eggs_for_an_eggetarian(tmp_path):
    llm = SequenceLLM([dish("eggs", "flour"), dish("flax seeds", "flour")])
    await call(diet_box(tmp_path, llm, diet="eggetarian"), "get_recipe", dish="eggless sponge cake")
    assert len(llm.prompts) == 2 and "eggs (conflicts with eggless diet)" in llm.prompts[1]


# ------------------------------------------------------------ review fix I8: eggs negated anywhere in the clause
@pytest.mark.parametrize("said", [
    "vegetarian, I don't like to eat eggs", "vegetarian, I've stopped eating eggs", "vegetarian and I avoid eating eggs",
    "vegetarian, my husband eats eggs but I don't", "vegetarian, I can't eat eggs", "vegetarian, I gave up eating eggs",
])
def test_eggs_negated_in_the_clause_stays_vegetarian(said):
    assert diets_in(said) == {"vegetarian"}


@pytest.mark.parametrize("said", ["vegetarian with eggs", "egg vegetarian", "vegetarian except eggs",
                                  "vegetarian plus eggs", "we're eggetarians", "veg and eggs are fine"])
def test_more_ways_to_say_eggetarian(said):
    assert diets_in(said) == {"eggetarian"}


@pytest.mark.parametrize("said", ["pescatarian, I eat eggs and fish but no meat", "I eat eggs and fish, no meat"])
def test_someone_who_eats_fish_is_not_made_eggetarian(said):
    assert "eggetarian" not in diets_in(said)
