"""Fixes for the issues found in the Fable review of the local-model branch.

Each test reproduces a reviewer probe (exact input, wrong behaviour) and
pins the corrected behaviour.
"""

import asyncio
import time

import pytest

from cooking_agent.agent import AgentConfig, CookingAgent, _Output
from cooking_agent.audio import SimulatedSink
from cooking_agent.commands import NO_REPLY, match_command
from cooking_agent.preferences import PreferenceStore, describe_for_recipe, diets_in
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import QueueSTT, ToneTTS
from cooking_agent.recipe import Recipe
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import Awaiting, CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


# ------------------------------------------------------------------ helpers
def dish(*names, title="Test"):
    return Recipe.model_validate({"title": title, "servings": 2,
                                  "ingredients": [{"name": n, "quantity": 1} for n in names],
                                  "steps": [{"instruction": "Cook."}]})


class SequenceLLM:
    name = "fake"

    def __init__(self, *recipes):
        self.queue = list(recipes)
        self.prompts = []

    async def generate_structured(self, prompt, schema, media=None):
        self.prompts.append(prompt)
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def box_with(tmp_path, llm, **prefs_by_category):
    prefs = PreferenceStore(tmp_path / "p.db")
    for category, value in prefs_by_category.items():
        prefs.remember(category, value)
    return ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), llm,
                   store=RecipeStore(tmp_path / "p.db"), prefs=prefs)


async def get_recipe(box, **args):
    return (await box.dispatch(ToolCall("get_recipe", args))).result


class NoLLM:
    name = "no-llm"

    def __init__(self):
        self.calls = 0

    async def stream(self, system, history, tools):
        self.calls += 1
        raise ConnectionError("down")
        yield  # pragma: no cover

    async def generate_structured(self, prompt, schema, media=None):
        raise ConnectionError("down")


def agent_with(recipe, *, realtime=False):
    llm, tts = NoLLM(), ToneTTS()
    agent = CookingAgent(QueueSTT(), llm, tts, SimulatedSink(realtime=realtime), AgentConfig())
    agent.session.load_recipe(recipe)
    return agent, llm, tts


# ------------------------------------------------------------------ 1. "non-veg" is not vegetarian
@pytest.mark.parametrize("text", ["non-veg biryani", "non-vegetarian", "not vegetarian", "non veg",
                                  "with lots of veg", "less veg more meat"])
def test_negated_or_vegetable_veg_is_not_a_diet(text):
    assert diets_in(text) == set()


@pytest.mark.parametrize("text", ["vegetarian", "I'm vegetarian", "veg biryani", "pure veg", "vegan"])
def test_real_diets_still_found(text):
    assert diets_in(text)


async def test_non_vegetarian_profile_is_not_forced_vegetarian(tmp_path):
    llm = SequenceLLM(dish("chicken", "rice"))
    out = await get_recipe(box_with(tmp_path, llm, diet="non-vegetarian"), dish="biryani")
    assert len(llm.prompts) == 1 and "warnings" not in out
    assert "no meat" not in describe_for_recipe({"diet": ["non-vegetarian"]})


async def test_non_veg_request_is_not_vegetarian(tmp_path):
    llm = SequenceLLM(dish("chicken", "rice"))
    out = await get_recipe(box_with(tmp_path, llm), dish="non-veg biryani")
    assert len(llm.prompts) == 1 and "warnings" not in out


# ------------------------------------------------------------------ 2. restating a rule keeps it on
async def test_jain_no_onion_no_garlic_still_checked(tmp_path):
    llm = SequenceLLM(dish("potato", "honey", "lentils"), dish("lentils", "tomato"))
    await get_recipe(box_with(tmp_path, llm, diet="jain"), dish="dal", constraints="no onion no garlic")
    assert len(llm.prompts) == 2


async def test_vegetarian_no_egg_still_checked(tmp_path):
    llm = SequenceLLM(dish("chicken", "rice"), dish("tofu", "rice"))
    await get_recipe(box_with(tmp_path, llm, diet="vegetarian"), dish="fried rice", constraints="no egg")
    assert len(llm.prompts) == 2


# ------------------------------------------------------------------ 3. naming one item allows only that item
async def test_vegan_palak_paneer_allows_paneer_only(tmp_path):
    llm = SequenceLLM(dish("paneer", "spinach", "cream"), dish("paneer", "spinach", "coconut cream"))
    out = await get_recipe(box_with(tmp_path, llm, diet="vegan"), dish="palak paneer")
    assert len(llm.prompts) == 2
    assert "cream (conflicts with vegan diet)" in llm.prompts[1]
    assert "paneer (conflicts" not in llm.prompts[1]
    assert "warnings" not in out


async def test_chicken_biryani_still_allows_chicken(tmp_path):
    llm = SequenceLLM(dish("chicken", "rice"))
    out = await get_recipe(box_with(tmp_path, llm, diet="vegetarian"), dish="chicken biryani")
    assert len(llm.prompts) == 1 and "warnings" not in out


# ------------------------------------------------------------------ 4. "hold on" keeps the timer offer
def m(text, session, awaiting=None, timers=()):
    return match_command(text, session, None, list(timers), awaiting)


@pytest.fixture
def cooking(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    s.next_step()
    return s


@pytest.mark.parametrize("text", ["hold on", "not yet", "wait", "give me a minute"])
def test_wait_keeps_timer_offer_open(cooking, text):
    out = m(text, cooking, awaiting=Awaiting.TIMER)
    assert out.calls == [] and out.reply == NO_REPLY and out.keep_open


def test_decline_closes_timer_offer(cooking):
    out = m("no thanks", cooking, awaiting=Awaiting.TIMER)
    assert out.calls == [] and not out.keep_open


def test_start_it_only_answers_a_timer_offer(cooking):
    assert m("start it", cooking, awaiting=Awaiting.NEXT_STEP) is None
    assert [c.name for c in m("start it", cooking, awaiting=Awaiting.TIMER).calls] == ["start_step_timer"]


async def test_hold_on_then_yes_starts_the_timer(recipe):
    agent, llm, tts = agent_with(recipe)
    await agent.submit_text("next")
    await agent.submit_text("hold on")
    assert tts.spoken[-1] == NO_REPLY
    assert agent.session.awaiting is Awaiting.TIMER
    await agent.submit_text("yes")
    assert agent.timers.list() and agent.session.step_index == 0 and llm.calls == 0
    await agent.timers.aclose()


# ------------------------------------------------------------------ 5. no accidental quitting
def test_im_finished_does_not_quit(cooking):
    assert m("I'm finished", cooking) is None
    assert m("I'm finished", CookingSession()) is None


def test_thats_all_does_not_quit_mid_cooking(cooking):
    assert m("that's all", cooking) is None
    assert m("that's all", CookingSession(), timers=["rice"]) is None
    setup = CookingSession()
    setup.setup_pending = True
    assert m("that's all", setup) is None
    assert [c.name for c in m("that's all", CookingSession()).calls] == ["end_session"]
    assert [c.name for c in m("bye", cooking).calls] == ["end_session"]


# ------------------------------------------------------------------ 6. "yes" only to a question just heard
async def test_announcement_after_question_makes_yes_go_to_llm(recipe):
    agent, llm, _ = agent_with(recipe)
    await agent.submit_text("next")
    await agent.submit_text("no")                       # -> "Are you ready for the next step?"
    await agent._speak_only("Your rice timer is done.", _Output())
    await agent.submit_text("okay")
    assert llm.calls == 1 and agent.session.step_index == 0


async def test_old_question_expires(recipe):
    agent, llm, _ = agent_with(recipe)
    await agent.submit_text("next")
    await agent.submit_text("no")
    agent._last_spoke_at = time.perf_counter() - 120     # two minutes later
    await agent.submit_text("yes")
    assert llm.calls == 1 and agent.session.step_index == 0


async def test_question_cut_off_is_not_fresh(recipe):
    agent, llm, _ = agent_with(recipe, realtime=True)
    task = agent.on_final("next")                       # step 1 + "start a timer?" being spoken
    await asyncio.sleep(0.1)
    agent.barge_in()                                    # cut off before the question was heard
    await asyncio.gather(task, return_exceptions=True)
    await agent.submit_text("yes")
    assert llm.calls == 1 and agent.timers.list() == []


# ------------------------------------------------------------------ 7. unnamed timers don't overwrite
def test_second_unnamed_timer_gets_its_own_name():
    s = CookingSession()
    assert m("set a timer for 5 minutes", s, timers=["kitchen"]).calls[0].args["name"] == "kitchen 2"
    assert m("set a timer for 5 minutes", s, timers=["kitchen", "kitchen 2"]).calls[0].args["name"] == "kitchen 3"
    assert m("set a rice timer for 5 minutes", s, timers=["rice"]).calls[0].args["name"] == "rice"


# ------------------------------------------------------------------ 8. compound durations
@pytest.mark.parametrize("text, minutes", [
    ("set a timer for five minutes and thirty seconds", 5.5),
    ("set a timer for one minute and thirty seconds", 1.5),
    ("set a timer for one hour and twenty minutes", 80),
])
def test_compound_durations_add_up(text, minutes):
    assert m(text, CookingSession()).calls[0].args["minutes"] == pytest.approx(minutes)


# ------------------------------------------------------------------ 9. no recipe swap mid-cook
def test_cook_saved_recipe_not_while_cooking(cooking, tmp_path, recipe):
    store = RecipeStore(tmp_path / "b.db")
    store.add(recipe.model_copy(update={"title": "Mum's Dal"}))
    assert match_command("cook my dal", cooking, store, [], None) is None
    assert match_command("cook my dal", CookingSession(), store, [], None) is not None


# ------------------------------------------------------------------ 10. warm-up and timeout
async def test_ollama_warmup_loads_the_model():
    from cooking_agent.providers.ollama import OllamaLLM

    class FakeClient:
        def __init__(self):
            self.generated = []

        async def generate(self, **kwargs):
            self.generated.append(kwargs)
            return {"response": ""}

    client = FakeClient()
    await OllamaLLM(model="qwen3:8b", client=client).warmup()
    assert client.generated and client.generated[0]["model"] == "qwen3:8b"
    assert client.generated[0]["keep_alive"] == "30m"


def test_ollama_client_has_a_timeout(monkeypatch):
    import ollama

    from cooking_agent.providers.ollama import OllamaLLM

    seen = {}

    class Capture:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(ollama, "AsyncClient", Capture)
    OllamaLLM(model="m")
    assert seen.get("timeout") and seen["timeout"] <= 180
