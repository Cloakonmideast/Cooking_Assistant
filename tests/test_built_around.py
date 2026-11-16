"""A usual change isn't made to a dish built around that ingredient ("leave out
cashews" in a kaju curry); the cook is told and offered alternatives instead."""

import pytest

from cooking_agent.agent import AgentConfig, CookingAgent
from cooking_agent.audio import SimulatedSink
from cooking_agent.dishes import built_around
from cooking_agent.preferences import PreferenceStore
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ToneTTS
from cooking_agent.recipe import Recipe
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import Awaiting, CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


@pytest.mark.parametrize("dish,ingredient", [
    ("kaju curry", "cashews"), ("kaju katli", "cashew nuts"), ("cashew pulao", "cashews"),
    ("cashew curry", "cashew nuts"), ("butter chicken", "unsalted butter"), ("aloo gobi", "potatoes"),
    ("rajma chawal", "kidney beans"), ("jeera rice", "cumin seeds"), ("garlic naan", "garlic cloves"),
    ("Mum's Kaju Masala", "cashews"), ("baingan bharta", "eggplant"), ("palak paneer", "spinach"),
])
def test_dish_named_after_the_ingredient(dish, ingredient):
    assert built_around(dish, ingredient)


@pytest.mark.parametrize("dish,ingredient", [
    ("vegetable pulao", "cashews"), ("milk cake", "coconut milk"), ("sugar-free cookies", "sugar"),
    ("paneer curry", "curry leaves"), ("pulao with cashews", "cashews"), ("dal without garlic", "garlic"),
    ("eggless cake", "eggs"), ("coconut rice", "coconut oil"),
])
def test_dish_not_built_around_it(dish, ingredient):
    assert not built_around(dish, ingredient)


def kaju_curry(title="Kaju Curry") -> Recipe:
    return Recipe.model_validate({
        "title": title, "servings": 2,
        "ingredients": [{"name": "cashews", "quantity": 1, "unit": "cup"},
                        {"name": "ghee", "quantity": 2, "unit": "tbsp"},
                        {"name": "onion", "quantity": 1},
                        {"name": "salt"}],
        "steps": [{"instruction": "Soak the cashews."}, {"instruction": "Cook the onion in the ghee."}]})


@pytest.fixture
def prefs(tmp_path):
    p = PreferenceStore(tmp_path / "book.db")
    yield p
    p.close()


def habits(prefs, *changes):
    for load in (101, 102):
        for kind, original, *replacement in changes:
            prefs.record_adjustment(kind, original, replacement[0] if replacement else None, load_id=load)


OPTIONS = {"use_instead": "almonds", "similar_dish": "paneer butter masala", "pick": "use_instead"}


def make_box(tmp_path, prefs, recipe, store=False, options=OPTIONS):
    llm = ScriptedLLM(recipe=recipe, structured={"BetterOptions": options})
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), llm,
                   store=RecipeStore(tmp_path / "book.db") if store else None, prefs=prefs)


async def get_recipe(tb, dish):
    result = await tb.dispatch(ToolCall("get_recipe", {"dish": dish}))
    return result.result, tb.follow_up([result])


async def test_kaju_curry_keeps_its_cashews_and_the_app_asks(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry())
    out, question = await get_recipe(tb, "kaju curry")
    assert "cashews" in [i.name for i in tb.session.recipe.ingredients]
    assert not any("cashew" in t for t in out.get("tailored_to", []))  # not claimed as done
    assert out["built_around"][0]["ingredient"] == "cashews"
    assert question == ("Kaju Curry is built around cashews, so I kept them in, though you usually leave them "
                        "out. You could use almonds instead, or make paneer butter masala. I'd go with "
                        "almonds. Shall I, or keep the cashews?")
    assert tb.session.awaiting is Awaiting.DISH_CHOICE


async def test_the_pick_can_be_the_similar_dish(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(), options=OPTIONS | {"pick": "similar_dish"})
    _, question = await get_recipe(tb, "kaju curry")
    assert question.endswith("I'd make paneer butter masala instead. Shall I, or keep the cashews?")


async def test_no_options_falls_back_to_a_plain_question(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(), options=RuntimeError("model down"))
    out, question = await get_recipe(tb, "kaju curry")
    assert "error" not in out
    assert question.endswith("Shall I leave them out anyway, or keep them?")


async def test_an_option_the_cook_cannot_have_is_dropped(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    prefs.remember("allergy", "almonds")
    tb = make_box(tmp_path, prefs, kaju_curry())
    _, question = await get_recipe(tb, "kaju curry")
    assert "almond" not in question
    assert question.endswith("I'd make paneer butter masala instead. Shall I, or keep the cashews?")


async def test_an_option_already_in_the_recipe_is_dropped(tmp_path, prefs):
    """qwen3 offered coconut milk for a curry that already had it: a "yes" made two coconut milk lines."""
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(), options=OPTIONS | {"use_instead": "Onions"})
    _, question = await get_recipe(tb, "kaju curry")
    assert "onion" not in question.lower()
    assert question.endswith("I'd make paneer butter masala instead. Shall I, or keep the cashews?")


async def test_a_similar_dish_built_around_it_too_is_dropped(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(), options=OPTIONS | {"similar_dish": "kaju masala"})
    _, question = await get_recipe(tb, "kaju curry")
    assert "masala" not in question and "I'd go with almonds" in question


async def test_the_writer_is_not_told_to_leave_out_the_main_ingredient(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"), ("remove", "onion"))
    tb = make_box(tmp_path, prefs, kaju_curry())
    await get_recipe(tb, "kaju curry")
    prompt = tb.llm.prompts[0]  # the recipe writer's (the next one asks for options)
    assert "leaves out cashews" not in prompt
    assert "usually leaves out onion" in prompt  # other habits still go to the writer


async def test_the_writer_is_told_what_an_indian_dish_name_means(tmp_path):
    """qwen3 wrote "almonds (kaju)" for a kaju curry in 2 of 8 runs."""
    tb = make_box(tmp_path, None, kaju_curry())
    await get_recipe(tb, "kaju curry")
    assert "In the dish name, kaju means cashew." in tb.llm.prompts[0]
    await get_recipe(tb, "vegetable pulao")
    assert "means" not in tb.llm.prompts[-1]


async def test_other_habits_are_still_made(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"), ("swap", "ghee", "butter"))
    tb = make_box(tmp_path, prefs, kaju_curry())
    out, _ = await get_recipe(tb, "kaju curry")
    names = [i.name for i in tb.session.recipe.ingredients]
    assert "cashews" in names and "butter" in names and "ghee" not in names
    assert "butter instead of ghee (your usual swap)" in out["tailored_to"]


async def test_a_dish_not_built_around_it_still_loses_it(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(title="Vegetable Korma"))
    out, question = await get_recipe(tb, "vegetable korma")
    assert "cashews" not in [i.name for i in tb.session.recipe.ingredients]
    assert "built_around" not in out and question is None
    assert len(tb.llm.prompts) == 1  # no options asked for


async def test_a_usual_swap_of_the_main_ingredient_is_offered_not_made(tmp_path, prefs):
    habits(prefs, ("swap", "ghee", "butter"))
    tb = make_box(tmp_path, prefs, kaju_curry(title="Ghee Rice"))
    out, question = await get_recipe(tb, "ghee rice")
    assert "ghee" in [i.name for i in tb.session.recipe.ingredients]
    assert out["built_around"][0]["ingredient"] == "ghee"
    assert question == ("Ghee Rice is built around ghee, so I kept it in, though you usually use butter instead. "
                        "Shall I use butter anyway, or keep the ghee?")
    assert len(tb.llm.prompts) == 1  # the usual swap is the option: no need to ask the model


async def test_no_note_when_the_recipe_does_not_have_it(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    no_cashews = kaju_curry().model_copy(update={"ingredients": kaju_curry().ingredients[1:]})
    out, question = await get_recipe(make_box(tmp_path, prefs, no_cashews), "kaju curry")
    assert "built_around" not in out and question is None


async def test_recipe_book_kaju_curry_is_not_offered_the_leave_out(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"), ("swap", "ghee", "butter"))
    tb = make_box(tmp_path, prefs, kaju_curry(), store=True)
    tb.store.add(kaju_curry(title="Mum's Kaju Masala"))
    out, question = await get_recipe(tb, "mum's kaju masala")
    assert out["source"] == "your recipe book"
    assert out["your_usual_changes"] == ["swap ghee for butter"]
    assert out["built_around"][0]["ingredient"] == "cashews"
    assert "cashews" in [i.name for i in tb.session.recipe.ingredients]
    assert question.startswith("Mum's Kaju Masala is built around cashews")


# ------------------------------------------------------------ the cook's answer, through the agent
def make_agent(prefs, options=OPTIONS):
    llm = ScriptedLLM(recipe=kaju_curry(), structured={"BetterOptions": options},
                      script={"kaju": [ToolCall("get_recipe", {"dish": "kaju curry", "servings": 2})]})
    return CookingAgent(QueueSTT(), llm, ToneTTS(ms_per_char=0), SimulatedSink(), AgentConfig(),
                        preferences=prefs)


async def asked(prefs, options=OPTIONS):
    habits(prefs, ("remove", "cashews"))
    agent = make_agent(prefs, options)
    await agent.submit_text("let's make kaju curry")
    assert "Shall I" in agent.tts.spoken[-1]
    return agent


def names(agent):
    return [i.name for i in agent.session.recipe.ingredients]


async def test_yes_makes_the_pick_and_does_not_start_cooking(prefs):
    agent = await asked(prefs)
    await agent.submit_text("yes")
    assert "almonds" in names(agent) and "cashews" not in names(agent)
    assert agent.session.step_index == -1
    assert agent.tts.spoken[-1] == "Okay, almonds instead of cashews."
    assert agent.session.dish_choice is None


async def test_keep_them_changes_nothing(prefs):
    agent = await asked(prefs)
    calls = len(agent.llm.calls)
    await agent.submit_text("no, keep them")
    assert "cashews" in names(agent) and agent.tts.spoken[-1] == "Okay, keeping the cashews."
    assert agent.session.dish_choice is None and len(agent.llm.calls) == calls


async def test_the_app_says_the_reply_itself_so_the_model_does_not_talk_about_it(prefs):
    """qwen3, told the app would ask, said "The app will ask if you want to leave out the cashews"
    and "I kept the cashews as you usually include them"."""
    agent = await asked(prefs)
    assert agent.tts.spoken[-2:] == [
        "Here's Kaju Curry for 2.",
        "Kaju Curry is built around cashews, so I kept them in, though you usually leave them out. You could use "
        "almonds instead, or make paneer butter masala. I'd go with almonds. Shall I, or keep the cashews?"]
    assert len(agent.llm.calls) == 1  # no second model round to describe the recipe


async def test_a_warning_is_still_left_to_the_model(prefs):
    prefs.remember("allergy", "onion")
    agent = await asked(prefs)
    assert len(agent.llm.calls) == 2


async def test_a_long_written_title_is_shortened(tmp_path, prefs):
    habits(prefs, ("remove", "cashews"))
    tb = make_box(tmp_path, prefs, kaju_curry(title="Kaju Curry (Cashew Curry) - 2 Servings"))
    _, question = await get_recipe(tb, "kaju curry")
    assert question.startswith("Kaju Curry is built around cashews")


@pytest.mark.parametrize("said", ["go with the one you'd pick", "yeah do that", "sounds good"])
async def test_your_pick(prefs, said):
    agent = await asked(prefs)
    await agent.submit_text(said)
    assert "almonds" in names(agent)


@pytest.mark.parametrize("said", ["almonds", "use the almonds", "the first one"])
async def test_naming_the_ingredient_option(prefs, said):
    agent = await asked(prefs, OPTIONS | {"pick": "similar_dish"})
    await agent.submit_text(said)
    assert "almonds" in names(agent) and "cashews" not in names(agent)


@pytest.mark.parametrize("said", ["the paneer one", "make the paneer butter masala", "the other one",
                                  "use the other option instead"])
async def test_naming_the_similar_dish(prefs, said):
    agent = await asked(prefs)
    await agent.submit_text(said)
    assert "recipe for: paneer butter masala" in agent.llm.prompts[-1]  # a new recipe was written for it


async def test_asked_even_when_the_model_ends_on_its_own_question(prefs, monkeypatch):
    monkeypatch.setattr(ScriptedLLM, "_describe", staticmethod(lambda name, result: "I found it. Shall we start?"))
    agent = await asked(prefs)
    assert agent.tts.spoken[-1].startswith("Kaju Curry is built around cashews")


async def test_yes_after_the_plain_question_leaves_them_out(prefs):
    agent = await asked(prefs, RuntimeError("model down"))
    await agent.submit_text("yes")
    assert "cashews" not in names(agent)


async def test_something_else_goes_to_the_model_and_clears_the_choice(prefs):
    agent = await asked(prefs)
    calls = len(agent.llm.calls)
    await agent.submit_text("how spicy is this dish usually?")
    assert len(agent.llm.calls) > calls and agent.session.dish_choice is None
