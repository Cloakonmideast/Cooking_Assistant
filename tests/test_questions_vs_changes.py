"""Asking whether vs asking to (review fixes I2, I9)."""

import pytest

from cooking_agent.agent import AgentConfig, CookingAgent
from cooking_agent.audio import SimulatedSink
from cooking_agent.preferences import PreferenceStore
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ToneTTS
from cooking_agent.state import Awaiting
from cooking_agent.utterance import asks_permission, is_question


@pytest.mark.parametrize("text", [
    "could I leave out the garlic", "Can I skip the mint?", "is it okay to skip the mint",
    "would it work without the yogurt", "what if I left out the chillies", "should we add more salt",
    "okay so, could I use less oil?", "do I have to add the cream",
])
def test_asking_permission(text):
    assert asks_permission(text)


@pytest.mark.parametrize("text", [
    "can you leave out the garlic", "could you make it milder", "leave out the garlic", "skip the mint",
    "I'll use butter instead", "please add peas", "no onion for me today",
])
def test_asking_for_a_change(text):
    assert not asks_permission(text)


@pytest.mark.parametrize("text,expected", [
    ("why do we soak it", True), ("how long is left?", True), ("is it done", True), ("what now", True),
    ("I'll keep an eye on it", False), ("nah, no timer", False), ("sounds good", False),
])
def test_is_question(text, expected):
    assert is_question(text) is expected


def agent_with(recipe, script, prefs=None):
    llm = ScriptedLLM(recipe=recipe, script={"recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})], **script})
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig(), preferences=prefs)
    return agent


async def on_step_2(agent):
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next")
    await agent.submit_text("next")
    assert agent.session.step_index == 1


REMOVE_CUMIN = [ToolCall("change_recipe", {"action": "remove_ingredient", "ingredient": "cumin"})]


async def test_could_i_does_not_change_the_recipe_or_teach_a_habit(recipe):
    prefs = PreferenceStore(":memory:")
    prefs.mark_setup_done()
    agent = agent_with(recipe, {"could i": REMOVE_CUMIN}, prefs)
    await on_step_2(agent)
    await agent.submit_text("could I leave out the cumin")
    assert "cumin seeds" in [i.name for i in agent.session.recipe.ingredients]
    assert prefs.adjustments() == []
    result = agent.history[-2].tool_results[0].result
    assert result["not_changed"] and "only asked" in result["note"]


async def test_can_you_does_change_it(recipe):
    agent = agent_with(recipe, {"can you": REMOVE_CUMIN})
    await on_step_2(agent)
    await agent.submit_text("can you leave out the cumin")
    assert "cumin seeds" not in [i.name for i in agent.session.recipe.ingredients]


async def test_yes_after_the_assistants_own_offer_makes_the_change_not_the_next_step(recipe):
    agent = agent_with(recipe, {"could i": ["You can, it will just taste milder. Want me to take it out?"],
                                "yes": REMOVE_CUMIN})
    await on_step_2(agent)
    await agent.submit_text("could I leave out the cumin")
    assert agent.session.awaiting is None  # the app's "ready?" is no longer what "yes" answers
    await agent.submit_text("yes")
    assert agent.session.step_index == 1
    assert "cumin seeds" not in [i.name for i in agent.session.recipe.ingredients]


async def test_a_question_keeps_the_timer_offer(recipe):
    agent = agent_with(recipe, {"why": ["It helps the grains cook evenly."]})
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next")  # step 1: soak 20 minutes -> timer offer
    assert agent.session.awaiting is Awaiting.TIMER
    await agent.submit_text("why do we soak it?")
    assert agent.tts.spoken[-1] == "Do you want me to start a timer for 20 minutes?"
    await agent.submit_text("yes")
    assert agent.timers.active_names == ["step 1"]
    await agent.timers.aclose()


async def test_a_statement_still_moves_past_the_timer_offer(recipe):
    agent = agent_with(recipe, {"keep an eye": ["Okay, I'll leave it to you."]})
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next")
    await agent.submit_text("I'll keep an eye on it myself")
    assert agent.tts.spoken[-1] == "Are you ready for the next step?"


# ------------------------------------------------------------ found re-running the eval on qwen3
def test_prompt_names_actions_only_as_change_recipe_actions():
    """qwen3 called a tool named "change_amount" (the prompt's wording); Ollama drops a
    call to a tool that doesn't exist, and the cook heard nothing."""
    import re

    from cooking_agent.agent import SYSTEM_PROMPT
    for m in re.finditer(r"change_amount|remove_ingredient|add_ingredient|change_step", SYSTEM_PROMPT):
        assert SYSTEM_PROMPT[max(0, m.start() - 40):m.start()].count("change_recipe"), SYSTEM_PROMPT[m.start() - 60:m.end()]
    assert "instead of sugar" not in SYSTEM_PROMPT  # a swap question as an answer-only example


class SilentThenTalking(ScriptedLLM):
    rounds: int = 0

    async def stream(self, system, history, tools):
        self.rounds += 1
        if self.rounds == 1:
            from cooking_agent.providers.base import Message, TurnComplete
            yield TurnComplete(Message("assistant", ""))
            return
        async for ev in super().stream(system, history, tools):
            yield ev


async def test_an_empty_model_reply_is_retried_once(recipe):
    llm = SilentThenTalking(recipe=recipe, script={"": ["Sure, here you go."]})
    from cooking_agent.providers.fake import QueueSTT as Q
    agent = CookingAgent(Q(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("tell me something")
    assert llm.rounds == 2 and agent.tts.spoken[-1] == "Sure, here you go."


async def test_a_model_that_stays_silent_still_gets_a_reply(recipe):
    llm = ScriptedLLM(recipe=recipe, script={"": [""]})
    agent = CookingAgent(QueueSTT(), llm, ToneTTS(), SimulatedSink(), AgentConfig())
    await agent.submit_text("tell me something")
    assert agent.tts.spoken[-1] == "Sorry, I didn't catch that. Could you say it another way?"
