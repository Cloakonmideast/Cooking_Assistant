"""Instant command matcher: phrases that skip the LLM entirely."""

import pytest

from cooking_agent.commands import CommandMatch, match_command, speak_result, words_to_digits
from cooking_agent.providers.base import ToolCall, ToolResult
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import Awaiting, CookingSession, RecipeDraft


@pytest.fixture
def cooking(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    return s


@pytest.fixture
def store(tmp_path, recipe):
    st = RecipeStore(tmp_path / "book.db")
    st.add(recipe.model_copy(update={"title": "Mum's Dal"}), source="test")
    return st


def calls(m):
    assert isinstance(m, CommandMatch), m
    return [(c.name, c.args) for c in m.calls]


def m(text, session=None, store=None, timers=(), awaiting=None, setup_open=False):
    return match_command(text, session or CookingSession(), store, list(timers), awaiting, setup_open=setup_open)


# ------------------------------------------------------------ one row each
def test_yes_answers_timer_offer(cooking):
    assert calls(m("yes", cooking, awaiting=Awaiting.TIMER)) == [("start_step_timer", {})]


def test_yes_answers_next_step(cooking):
    assert calls(m("Yeah, go ahead.", cooking, awaiting=Awaiting.NEXT_STEP)) == [("next_step", {})]


def test_no_gives_reply(cooking):
    out = m("not yet", cooking, awaiting=Awaiting.NEXT_STEP)
    assert out.calls == [] and out.reply == "No problem, take your time."


@pytest.mark.parametrize("text", ["next", "Next step please.", "ready", "I'm done", "okay, continue"])
def test_next(cooking, text):
    assert calls(m(text, cooking)) == [("next_step", {})]


@pytest.mark.parametrize("text", ["back", "go back", "previous step"])
def test_back(cooking, text):
    assert calls(m(text, cooking)) == [("previous_step", {})]


@pytest.mark.parametrize("text", ["repeat", "say that again", "what was that?"])
def test_repeat(cooking, text):
    assert calls(m(text, cooking)) == [("repeat_step", {})]


@pytest.mark.parametrize("text, name, minutes", [
    ("set a rice timer for 12 minutes", "rice", 12),
    ("set a timer for 5 minutes", "kitchen", 5),
    ("can you start a pasta timer for 2 hours", "pasta", 120),
    ("set a 10 minute timer", "kitchen", 10),
])
def test_set_timer(text, name, minutes):
    assert calls(m(text)) == [("set_timer", {"name": name, "minutes": minutes})]


def test_cancel_named_timer():
    assert calls(m("cancel the rice timer", timers=["rice", "pasta"])) == [("cancel_timer", {"name": "rice"})]


@pytest.mark.parametrize("text", ["how long is left", "what timers are running", "how much time is left?"])
def test_list_timers(text):
    assert calls(m(text)) == [("list_timers", {})]


@pytest.mark.parametrize("text", ["what do I need", "ingredients", "read me the ingredients"])
def test_ingredients(cooking, text):
    assert calls(m(text, cooking)) == [("list_ingredients", {})]


@pytest.mark.parametrize("text", ["make it for 6 people", "scale it for six", "make it for four servings"])
def test_scale(cooking, text):
    out = calls(m(text, cooking))
    assert out[0][0] == "scale_servings" and out[0][1]["servings"] in (4, 6)


def test_bye():
    assert calls(m("that's all, bye")) == [("end_session", {})]


# ------------------------------------------------------------ the risky cases
@pytest.mark.parametrize("text, minutes", [
    ("set a rice timer for twelve minutes", 12),
    ("set a timer for a minute", 1),
    ("set a timer for one and a half minutes", 1.5),
    ("set a timer for thirty seconds", 0.5),
    ("set a timer for twenty five minutes", 25),
    ("set a timer for half an hour", 30),
])
def test_number_words(text, minutes):
    (name, args), = calls(m(text))
    assert name == "set_timer" and args["minutes"] == pytest.approx(minutes)


def test_words_to_digits():
    assert words_to_digits("set a rice timer for twelve minutes") == "set a rice timer for 12 minutes"
    assert words_to_digits("one and a half minutes") == "1.5 minutes"
    assert words_to_digits("half an hour") == "30 minutes"


@pytest.mark.parametrize("text", [
    "next time make it spicier",
    "I don't have ghee what can I use",
    "what goes with the rice timer",
    "set a timer for when the rice boils",
    "go back to the store later",
])
def test_longer_sentences_fall_through(cooking, text):
    assert m(text, cooking, timers=["rice"]) is None


def test_yes_no_need_awaiting(cooking):
    assert m("yes", cooking) is None
    assert m("no", cooking) is None


def test_step_commands_need_a_recipe():
    assert m("next") is None
    assert m("what do I need") is None


def test_dictation_only_timers(cooking):
    cooking.draft = RecipeDraft(title="Mum's Dal")
    assert m("next", cooking) is None
    assert m("that's all", cooking) is None
    assert calls(m("set a pasta timer for 5 minutes", cooking)) == [("set_timer", {"name": "pasta", "minutes": 5})]


def test_setup_open_holds_back_possible_answers(cooking, store):
    # Right after "any allergies? ... if not, just say none", these answer the question.
    assert m("no", cooking, awaiting=Awaiting.NEXT_STEP, setup_open=True) is None
    assert m("yes", cooking, awaiting=Awaiting.NEXT_STEP, setup_open=True) is None
    assert m("next", cooking, setup_open=True) is None
    assert m("that's all", cooking, setup_open=True) is None
    # Clear commands still work.
    assert calls(m("set a pasta timer for 5 minutes", cooking, setup_open=True))[0][0] == "set_timer"
    assert calls(m("cook my dal", CookingSession(), store, setup_open=True)) == [("get_recipe", {"dish": "dal"})]


def test_stop_rules():
    assert m("stop", timers=["rice"]) is None
    assert calls(m("stop the timer", timers=["rice"])) == [("cancel_timer", {"name": "rice"})]
    assert m("stop the timer", timers=["rice", "pasta"]) is None
    assert m("stop the timer", timers=[]) is None


def test_cook_saved_only(store):
    assert calls(m("cook my dal", store=store)) == [("get_recipe", {"dish": "dal"})]
    assert m("cook my biryani", store=store) is None
    assert m("cook my dal", store=None) is None


# ------------------------------------------------------------ spoken replies
def test_speak_result_templates():
    def say(name, result):
        return speak_result([ToolResult(name, result)])

    assert say("set_timer", {"name": "rice", "seconds": 720, "replaced_existing": False}) == \
        "Okay, rice timer set for 12 minutes."
    assert say("start_step_timer", {"name": "step 3", "seconds": 600, "duration": "10 minutes"}) == \
        "Okay, timer started for 10 minutes."
    assert say("cancel_timer", {"cancelled": "rice"}) == "Rice timer cancelled."
    assert say("list_timers", {"timers": [{"name": "rice", "remaining_seconds": 245},
                                          {"name": "pasta", "remaining_seconds": 30}]}) == \
        "Rice: 4 minutes left. Pasta: 30 seconds left."
    assert say("list_timers", {"timers": []}) == "No timers are running."
    assert say("list_ingredients", {"servings": 2, "ingredients": ["1 cup rice", "2 cups water"]}) == \
        "For 2: 1 cup rice. 2 cups water."
    assert say("scale_servings", {"servings": 4, "ingredients": ["2 cups rice"]}) == "Okay, for 4: 2 cups rice."
    assert say("get_recipe", {"title": "Mum's Dal", "servings": 4, "source": "your recipe book"}) == \
        "Found Mum's Dal in your recipe book, for 4. Say ready to start."
    assert say("get_recipe", {"title": "Kheer", "servings": 4, "source": "your recipe book",
                              "warnings": ["contains milk (conflicts with lactose intolerance)"]}) == \
        "Heads up: this recipe contains milk (conflicts with lactose intolerance). " \
        "Found Kheer in your recipe book, for 4. Say ready to start."
    assert say("end_session", {"ok": True}) == "Goodbye, happy cooking!"
    assert say("cancel_timer", {"error": "There's no running timer called rice."}) == \
        "There's no running timer called rice."


def test_no_template_ends_with_a_question():
    results = [ToolResult("set_timer", {"name": "rice", "seconds": 60}),
               ToolResult("list_timers", {"timers": []}),
               ToolResult("list_ingredients", {"servings": 2, "ingredients": ["salt"]})]
    for r in results:
        assert not speak_result([r]).endswith("?")


def test_step_results_left_to_agent():
    assert speak_result([ToolResult("next_step", {"step_number": 1, "instruction": "x"})]) is None


def test_finished_recipe_message_is_spoken():
    assert speak_result([ToolResult("next_step", {"finished": True, "message": "That was the last step."})]) ==         "That was the last step."


@pytest.mark.parametrize("said", ["right, that's toasted, what's next", "okay done, what now?",
                                  "what's next", "that's done, what do I do next"])
def test_whats_next_moves_on_while_cooking(cooking, said):
    cooking.next_step()
    assert calls(m(said, cooking)) == [("next_step", {})]


@pytest.mark.parametrize("said", ["I burnt the onions, what now", "it's too salty, what do I do now",
                                  "the oil is smoking, what now?"])
def test_a_problem_then_what_now_is_not_next_step(cooking, said):
    cooking.next_step()
    out = m(said, cooking)
    assert out is None or ("next_step", {}) not in calls(out)


@pytest.mark.parametrize("said", ["what now", "okay done, what now?", "finished, what now"])
def test_done_then_what_now_is_next_step(cooking, said):
    cooking.next_step()
    assert calls(m(said, cooking)) == [("next_step", {})]
