"""A diet said in conversation is saved in code (review fix C2)."""

import pytest

from cooking_agent.agent import AgentConfig, CookingAgent
from cooking_agent.audio import SimulatedSink
from cooking_agent.preferences import PreferenceStore, describe_for_recipe, stated_diets
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ToneTTS


@pytest.mark.parametrize("said,diets", [
    ("I'm vegetarian", {"vegetarian"}),
    ("i am a vegetarian but I eat eggs", {"eggetarian"}),
    ("I'm vegetarian with eggs", {"eggetarian"}),
    ("I'm a vegetarian who eats eggs", {"eggetarian"}),
    ("I eat eggs but no meat", {"eggetarian"}),
    ("I'm eggetarian", {"eggetarian"}),
    ("we're vegan", {"vegan"}),
    ("I'm strictly jain, find me a dal recipe", {"jain"}),
    ("no allergies, I eat eggs but no meat", {"eggetarian"}),
    ("I've gone vegan", {"vegan"}),
])
def test_first_person_diet_statements(said, diets):
    assert stated_diets(said) == diets


@pytest.mark.parametrize("said", [
    "I'm making a vegetarian biryani", "I'm cooking for a vegan friend", "find me a vegan kheer",
    "I'm not vegetarian", "I'm vegan today", "is this vegetarian", "my wife is vegetarian",
    "I'm vegetarian just for this week",
])
def test_not_a_statement_about_the_cook(said):
    assert stated_diets(said) == set()


def make_agent(prefs):
    llm = ScriptedLLM(script={"": ["Got it, I'll remember that."]})
    return CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig(), preferences=prefs)


async def test_saved_even_when_the_model_calls_no_tool():
    prefs = PreferenceStore(":memory:")
    prefs.mark_setup_done()
    agent = make_agent(prefs)
    await agent.submit_text("I'm vegetarian but I eat eggs")
    assert prefs.profile()["diet"] == ["eggetarian"]


async def test_the_newest_diet_replaces_the_old_one():
    prefs = PreferenceStore(":memory:")
    prefs.mark_setup_done()
    prefs.remember("diet", "eggetarian")
    prefs.remember("diet", "halal")
    agent = make_agent(prefs)
    await agent.submit_text("I'm vegetarian now, no eggs")
    assert sorted(prefs.profile()["diet"]) == ["halal", "vegetarian"]


def test_recipe_writer_gets_one_rule_when_both_are_stored():
    text = describe_for_recipe({"diet": ["vegetarian", "eggetarian"]})
    assert "eggs are fine" in text
    assert "seafood, eggs" not in text  # the plain vegetarian rule ("no ... seafood, eggs, ...") isn't sent too


@pytest.mark.parametrize("said,diets", [
    ("honestly I'm a vegetarian, but eggs are okay for me", {"eggetarian"}),
    ("well we're vegan", {"vegan"}),
    ("oh and I'm a strict jain", {"jain"}),
])
def test_statement_after_a_filler_word(said, diets):
    """qwen3 stored "vegetarian (egg okay)" for the first one, which reads as plain vegetarian."""
    assert stated_diets(said) == diets


@pytest.mark.parametrize("said", ["I think I'm making a vegetarian biryani", "honestly I'm not vegan"])
def test_filler_word_does_not_widen_to_requests(said):
    assert stated_diets(said) == set()


@pytest.mark.parametrize("said", ["what if I'm vegan", "if I'm vegetarian, can I eat this",
                                  "suppose we're vegan, what then", "say I'm jain, is this okay"])
def test_hypotheticals_are_not_saved(said):
    assert stated_diets(said) == set()
