"""Questions and changes in the middle of a recipe (Phase 2, Task 9)."""

import pytest

from cooking_agent.agent import AgentConfig, CookingAgent
from cooking_agent.audio import SimulatedSink
from cooking_agent.commands import match_command
from cooking_agent.preferences import PreferenceStore
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ToneTTS
from cooking_agent.recipe import Ingredient, Recipe, Step, parse_ingredient_line
from cooking_agent.state import CookingPhase, CookingSession, InvalidTransition
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


def make_box(recipe, prefs=None):
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), ScriptedLLM(recipe=recipe), prefs=prefs)


async def call(tb, tool_name, /, **args):
    return (await tb.dispatch(ToolCall(tool_name, args))).result


async def on_step_2(tb):
    await call(tb, "get_recipe", dish="jeera rice")
    await call(tb, "next_step")
    await call(tb, "next_step")
    assert tb.session.step_index == 1


# ------------------------------------------------------------ the recipe in the state
def test_snapshot_includes_recipe_text_and_current_step(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    text = s.snapshot()["recipe_text"]
    assert "Jeera Rice, for 2" in text and "1 cup basmati rice, rinsed" in text
    assert "[current step]" not in text  # not cooking yet
    s.next_step()
    s.next_step()
    text = s.snapshot()["recipe_text"]
    assert "2. [current step] Heat ghee and add cumin seeds." in text
    assert "1. Soak the rice for 20 minutes." in text
    s.scale_servings(4)
    assert "2 cups basmati rice" in s.snapshot()["recipe_text"]


# ------------------------------------------------------------ change_recipe
async def test_change_recipe_remove_add_amount_keeps_step(recipe):
    tb = make_box(recipe)
    await on_step_2(tb)
    out = await call(tb, "change_recipe", action="remove_ingredient", ingredient="cumin")
    assert out["removed"] == "1 tsp cumin seeds"
    assert (await call(tb, "repeat_step"))["instruction"] == "Heat ghee and add cumin seeds. Leave out the cumin seeds."
    assert tb.session.step_index == 1
    out = await call(tb, "change_recipe", action="add_ingredient", ingredient="green peas", amount="1 cup")
    assert out["added"] == "1 cup green peas"
    assert tb.session.step_index == 1
    out = await call(tb, "change_recipe", action="change_amount", ingredient="ghee", amount="double")
    assert out["line"] == "3 tbsp ghee"
    assert tb.session.step_index == 1 and tb.session.phase is CookingPhase.COOKING
    names = [i.name for i in tb.session.recipe.ingredients]
    assert names == ["basmati rice", "ghee", "salt", "green peas"]


async def test_amounts_are_spoken_at_current_servings(recipe):
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    await call(tb, "scale_servings", servings=4)
    out = await call(tb, "change_recipe", action="change_amount", ingredient="ghee", amount="2 tbsp")
    assert out["line"] == "2 tbsp ghee"
    out = await call(tb, "change_recipe", action="add_ingredient", ingredient="peas", amount="2 cups")
    assert out["added"] == "2 cups peas"
    lines = (await call(tb, "scale_servings", servings=2))["ingredients"]
    assert "1 tbsp ghee" in lines and "1 cup peas" in lines


async def test_less_and_more(recipe):
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    assert (await call(tb, "change_recipe", action="change_amount", ingredient="cumin seeds",
                       amount="less"))["line"] == "1/2 tsp cumin seeds"
    assert (await call(tb, "change_recipe", action="change_amount", ingredient="salt", amount="more"))["line"] \
        == "salt, a little more than usual"


async def test_changed_imported_line_reads_new_amount(recipe):
    imported = recipe.model_copy(update={"ingredients": [parse_ingredient_line("1 1/2 tbsp ghee"),
                                                         parse_ingredient_line("1 cup basmati rice")]})
    tb = make_box(imported)
    await call(tb, "get_recipe", dish="jeera rice")
    out = await call(tb, "change_recipe", action="change_amount", ingredient="ghee", amount="2 tbsp")
    assert out["line"] == "2 tbsp ghee"


async def test_change_recipe_added_allergen_warned(recipe):
    prefs = PreferenceStore(":memory:")
    prefs.remember("allergy", "peanuts")
    tb = make_box(recipe, prefs)
    await call(tb, "get_recipe", dish="jeera rice")
    out = await call(tb, "change_recipe", action="add_ingredient", ingredient="roasted peanuts", amount="2 tbsp")
    assert out["warnings"] == ["contains roasted peanuts (conflicts with peanuts allergy)"]


async def test_dish_named_in_request_is_not_warned_on_change(recipe):
    prefs = PreferenceStore(":memory:")
    prefs.remember("diet", "vegetarian")
    tb = make_box(recipe.model_copy(update={"title": "Chicken Rice"}), prefs)
    await call(tb, "get_recipe", dish="chicken rice")
    out = await call(tb, "change_recipe", action="add_ingredient", ingredient="chicken thighs", amount="200 g")
    assert "warnings" not in out
    out = await call(tb, "change_recipe", action="add_ingredient", ingredient="bacon", amount="2 slices")
    assert out["warnings"] == ["contains bacon (conflicts with vegetarian diet)"]


async def test_change_recipe_unknown_ingredient_lists_ingredients(recipe):
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    out = await call(tb, "change_recipe", action="remove_ingredient", ingredient="cashews")
    assert "can't find cashews" in out["error"] and "basmati rice, cumin seeds, ghee, salt" in out["error"]


async def test_remove_matches_whole_words_only(recipe):
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    assert "error" in await call(tb, "change_recipe", action="remove_ingredient", ingredient="ice")
    assert (await call(tb, "change_recipe", action="remove_ingredient", ingredient="the rice"))["removed"] \
        == "1 cup basmati rice, rinsed"


async def test_cannot_remove_the_last_ingredient(recipe):
    one = recipe.model_copy(update={"ingredients": recipe.ingredients[:1]})
    tb = make_box(one)
    await call(tb, "get_recipe", dish="jeera rice")
    assert "only ingredient" in (await call(tb, "change_recipe", action="remove_ingredient",
                                            ingredient="rice"))["error"]


async def test_change_step_rewrites_current_step_and_its_timer(recipe):
    tb = make_box(recipe)
    await on_step_2(tb)
    assert tb.session.timer_offer is None
    out = await call(tb, "change_recipe", action="change_step", step_number=2,
                     text="Heat ghee and fry for 5 minutes.")
    assert out == {"step_number": 2, "instruction": "Heat ghee and fry for 5 minutes.",
                   "suggested_timer_minutes": 5}
    assert tb.session.timer_offer == {"minutes": 5, "step_number": 2}
    assert (await call(tb, "repeat_step"))["instruction"] == "Heat ghee and fry for 5 minutes."
    assert tb.session.step_index == 1


async def test_change_step_defaults_to_current_and_checks_number(recipe):
    tb = make_box(recipe)
    await on_step_2(tb)
    out = await call(tb, "change_recipe", action="change_step", text="Heat ghee only.")
    assert out["step_number"] == 2
    assert "only 3 steps" in (await call(tb, "change_recipe", action="change_step", step_number=9,
                                         text="x"))["error"]


async def test_change_recipe_offered_only_with_a_recipe(recipe):
    tb = make_box(recipe)
    assert "change_recipe" not in {s.name for s in tb.specs_for(tb.session)}
    await call(tb, "get_recipe", dish="jeera rice")
    assert "change_recipe" in {s.name for s in tb.specs_for(tb.session)}


@pytest.mark.parametrize("said", ["skip the cumin", "add some peas", "double the garlic", "make it less spicy",
                                  "can I use brown rice instead", "how do I know the rice is done"])
def test_changes_and_questions_are_not_instant_commands(recipe, said):
    s = CookingSession()
    s.load_recipe(recipe)
    s.next_step()
    assert match_command(said, s, None, [], None) is None


# ------------------------------------------------------------ the agent
async def test_question_mid_recipe_does_not_move_step(recipe):
    llm = ScriptedLLM(recipe=recipe, script={
        "recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})],
        "how do i know": ["It's done when the water is gone and the grains are soft."],
    })
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next")
    await agent.submit_text("next")
    await agent.submit_text("how do I know the rice is done?")
    assert agent.session.step_index == 1
    assert "[current step] Heat ghee" in agent._system_prompt()
    assert "change_recipe" in agent._system_prompt()


async def test_substitute_accepts_an_amount_said_with_its_unit(recipe):
    """qwen3 sent quantity "2 tbsp"; the cook heard a validation error instead of the swap."""
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    out = await call(tb, "substitute_ingredient", ingredient="ghee", replacement="butter", quantity="2 tbsp",
                     unit="tbsp")
    assert out["line"] == "2 tbsp butter, instead of ghee"
    out = await call(tb, "substitute_ingredient", ingredient="cumin", replacement="caraway seeds",
                     quantity="1 1/2 tsp")
    assert out["line"] == "1 and 1/2 tsp caraway seeds, instead of cumin seeds"
    out = await call(tb, "substitute_ingredient", ingredient="salt", replacement="rock salt", quantity="to taste")
    assert out["line"] == "rock salt, instead of salt"


# ------------------------------------------------------------ instant turns in the history
async def test_instant_turns_are_tool_shaped_for_models_that_learn_from_them(recipe):
    """With instant turns as plain text ("next" -> "Step 1. ..."), qwen3 copied the
    pattern and wrote "next_step" as text instead of calling it (21/33 right vs 28/33
    with tool-shaped history, measured)."""
    llm = ScriptedLLM(recipe=recipe, script={"recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})]})
    llm.tool_history = True
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("find a jeera rice recipe")
    n = len(agent.history)
    await agent.submit_text("next")
    turn = agent.history[n:]
    assert [m.role for m in turn] == ["user", "assistant", "tool", "assistant"]
    assert [c.name for c in turn[1].tool_calls] == ["next_step"] and turn[1].tool_calls[0].id
    assert turn[2].tool_results[0].result["step_number"] == 1
    assert turn[2].tool_results[0].id == turn[1].tool_calls[0].id
    assert turn[3].text.startswith("Step 1. Soak the rice") and turn[3].text.endswith("20 minutes?")


async def test_instant_turns_stay_plain_text_for_other_models(recipe):
    llm = ScriptedLLM(recipe=recipe, script={"recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})]})
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("find a jeera rice recipe")
    n = len(agent.history)
    await agent.submit_text("next")
    assert [m.role for m in agent.history[n:]] == ["user", "assistant"]


async def test_canned_instant_reply_stays_plain(recipe):
    llm = ScriptedLLM(recipe=recipe, script={"recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})]})
    llm.tool_history = True
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next")
    n = len(agent.history)
    await agent.submit_text("not yet")  # "No problem, take your time." with no tool
    assert [m.role for m in agent.history[n:]] == ["user", "assistant"]


def test_ollama_asks_for_tool_shaped_history():
    from cooking_agent.providers.ollama import OllamaLLM

    assert OllamaLLM(model="qwen3:8b", client=object()).tool_history is True


# ------------------------------------------------------------ review fixes: matching (I5, I6)
def _session(*names, steps=("Cook it.",)):
    s = CookingSession()
    s.load_recipe(Recipe.model_validate({"title": "T", "servings": 2,
                                         "ingredients": [{"name": n, "quantity": 1} for n in names],
                                         "steps": [{"instruction": t} for t in steps]}))
    return s


@pytest.mark.parametrize("said", ["chilli", "chili", "chillies", "the green chilies", "green chilli"])
def test_chilli_spellings_and_plurals_match(said):
    s = _session("green chillies", "onion")
    assert s.remove_ingredient(said)["ingredient"] == "green chillies"


def test_leaf_and_leaves_match():
    assert _session("bay leaves", "rice").remove_ingredient("bay leaf")["ingredient"] == "bay leaves"
    assert _session("bay leaf", "rice").remove_ingredient("the bay leaves")["ingredient"] == "bay leaf"


def test_steps_found_by_other_spellings_and_by_the_head_noun():
    s = _session("green chillies", "olive oil", "rice",
                 steps=("Heat the oil.", "Add the chilli and fry.", "Add the rice."))
    assert [x["step_number"] for x in s.steps_mentioning("green chillies")] == [2]
    assert [x["step_number"] for x in s.steps_mentioning("olive oil")] == [1]


def test_ambiguous_ingredient_asks_which():
    s = _session("rice flour", "basmati rice", "salt", "black pepper")
    with pytest.raises(InvalidTransition, match="Which one.*rice flour.*basmati rice"):
        s.remove_ingredient("rice")
    with pytest.raises(InvalidTransition, match="Which one"):
        s.remove_ingredient("salt and pepper")
    assert s.remove_ingredient("basmati")["ingredient"] == "basmati rice"
    assert s.remove_ingredient("rice flour")["ingredient"] == "rice flour"


# ------------------------------------------------------------ review fixes: amounts (I4)
def _amounts():
    s = CookingSession()
    s.load_recipe(Recipe.model_validate({"title": "T", "servings": 2, "ingredients": [
        {"name": "basmati rice", "quantity": 2, "unit": "cups"}, {"name": "garlic", "quantity": 2, "unit": "cloves"},
        {"name": "salt"}], "steps": [{"instruction": "Cook it."}]}))
    return s


@pytest.mark.parametrize("amount,line,more", [
    ("one and a half cups", "1 and 1/2 cups basmati rice", False),
    ("half a cup", "1/2 cup basmati rice", False),
    ("half a cup less", "1 and 1/2 cups basmati rice", False),
    ("1 cup more", "3 cups basmati rice", True),
    ("3 cups", "3 cups basmati rice", True),
    ("double", "4 cups basmati rice", True),
    ("twice as much", "4 cups basmati rice", True),
    ("half as much", "1 cup basmati rice", False),
    ("a bit more", "3 cups basmati rice", True),
    ("less", "1 cup basmati rice", False),
])
def test_amounts_read_correctly(amount, line, more):
    out = _amounts().change_amount("rice", amount)
    assert (out["line"], out["more"]) == (line, more)


def test_more_cloves_adds_to_the_amount():
    assert _amounts().change_amount("garlic", "2 more cloves")["line"] == "4 cloves garlic"
    assert _amounts().change_amount("garlic", "2 more")["line"] == "4 cloves garlic"


def test_zero_amount_is_refused():
    with pytest.raises(InvalidTransition, match="leave it out"):
        _amounts().change_amount("garlic", "0")


# ------------------------------------------------------------ review fixes: steps after a change (I1)
def _mint():
    s = CookingSession()
    s.load_recipe(Recipe.model_validate({"title": "T", "servings": 2, "ingredients": [
        {"name": "basmati rice", "quantity": 1, "unit": "cup"}, {"name": "ghee", "quantity": 1, "unit": "tbsp"},
        {"name": "mint leaves", "quantity": 10}], "steps": [
        {"instruction": "Heat the ghee."}, {"instruction": "Add the rice and cook."},
        {"instruction": "Scatter the mint over the rice and serve."}]}))
    return s


def test_a_removed_ingredient_is_left_out_when_the_step_is_read():
    s = _mint()
    s.next_step()
    out = s.remove_ingredient("mint")
    assert "steps_to_update" not in out
    s.next_step()
    assert s.next_step()["instruction"] == "Scatter the mint over the rice and serve. Leave out the mint leaves."
    assert "3. [current step] Scatter the mint over the rice and serve. Leave out the mint leaves." in s.recipe_text()


def test_a_swapped_ingredient_is_named_when_the_step_is_read():
    s = _mint()
    s.substitute("ghee", Ingredient(name="butter"))
    assert s.next_step()["instruction"] == "Heat the ghee. Use butter instead of ghee."


def test_adding_it_back_or_a_new_recipe_clears_the_note():
    s = _mint()
    s.remove_ingredient("mint")
    s.add_ingredient("10 mint leaves")
    assert "Leave out" not in s.recipe_text()
    s.remove_ingredient("mint")
    s.load_recipe(_mint().recipe)
    assert "Leave out" not in s.recipe_text()


def test_prompt_no_longer_asks_the_model_to_rewrite_steps():
    from cooking_agent.agent import SYSTEM_PROMPT

    assert "steps_to_update" not in SYSTEM_PROMPT


def test_the_verb_leave_is_not_the_leaves_ingredient():
    s = _session("coriander leaves", "rice", steps=("Garnish with coriander.", "Leave to rest for 5 minutes."))
    s.remove_ingredient("coriander leaves")
    assert s.step_text(1) == "Leave to rest for 5 minutes."
    assert s.step_text(0).endswith("Leave out the coriander leaves.")


def test_half_cup_without_an_article():
    assert _amounts().change_amount("rice", "half cup less")["line"] == "1 and 1/2 cups basmati rice"
    assert _amounts().change_amount("rice", "half cup")["line"] == "1/2 cup basmati rice"
    assert _amounts().change_amount("rice", "half the rice")["line"] == "1 cup basmati rice"


async def test_a_swap_sent_to_change_recipe_is_routed_to_substitute(recipe):
    """qwen3 sometimes calls change_recipe with action "substitute_ingredient"."""
    tb = make_box(recipe)
    await call(tb, "get_recipe", dish="jeera rice")
    out = await call(tb, "change_recipe", action="substitute_ingredient", ingredient="ghee", replacement="butter",
                     quantity="2 tbsp", unit="tbsp")
    assert out["line"] == "2 tbsp butter, instead of ghee"
