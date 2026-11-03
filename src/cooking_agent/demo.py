"""Offline scripted demo: shows tool calling, timers firing mid-speech and
barge-in using fake providers. Runs anywhere; no keys, no audio hardware."""

from __future__ import annotations

import asyncio

from .agent import AgentConfig, CookingAgent
from .audio import SimulatedSink
from .metrics import MetricsSink
from .providers.base import ToolCall
from .providers.fake import QueueSTT, ScriptedLLM, ToneTTS
from .recipe import Recipe
from .recipe_store import RecipeStore

DEMO_RECIPE = Recipe.model_validate({
    "title": "Simple Egg Fried Rice",
    "servings": 2,
    "ingredients": [
        {"name": "cooked rice", "quantity": 2, "unit": "cups", "note": "day-old if possible"},
        {"name": "eggs", "quantity": 2},
        {"name": "soy sauce", "quantity": 1.5, "unit": "tbsp"},
        {"name": "spring onions", "quantity": 2, "note": "sliced"},
        {"name": "oil", "quantity": 1, "unit": "tbsp"},
    ],
    "steps": [
        {"instruction": "Heat the oil in a wok over high heat."},
        {"instruction": "Scramble the eggs until just set, then push them aside."},
        {"instruction": "Add the rice and stir fry for three minutes.", "timer_minutes": 3},
        {"instruction": "Add soy sauce and spring onions, toss and serve."},
    ],
})


def demo_llm() -> ScriptedLLM:
    return ScriptedLLM(
        recipe=DEMO_RECIPE,
        script={
            "kheer for four": ["Let me check.", ToolCall("get_recipe", {"dish": "kheer", "servings": 4})],
            "save my own": [ToolCall("start_recipe_dictation", {"title": "Grandma's Kheer", "servings": 2})],
            "recipe": ["Sure, let me find one.", ToolCall("get_recipe", {"dish": "egg fried rice", "servings": 2})],
            "litre": [ToolCall("add_to_dictated_recipe", {"ingredients": ["1 litre milk", "1/4 cup rice, washed",
                                                                           "3 tbsp sugar"]})],
            "boil": [ToolCall("add_to_dictated_recipe", {"steps": [
                "Boil the milk.", "Add the rice and simmer for 40 minutes, stirring often.", "Stir in the sugar."]})],
            "save it": [ToolCall("save_dictated_recipe", {})],
            "not yet": ["No problem, take your time."],
            "next": [ToolCall("next_step", {})],
            "ready": [ToolCall("next_step", {})],
            "back": [ToolCall("previous_step", {})],
            "again": [ToolCall("repeat_step", {})],
            "stop the": [ToolCall("cancel_timer", {"name": "rice"})],
            "timer": [ToolCall("set_timer", {"name": "rice", "minutes": 0.02})],
            "long": ["This is a long answer that keeps going so that you have time to interrupt "
                     "it. It goes on for a while, sentence after sentence. And then some more."],
            "bye": [ToolCall("end_session", {})],
        },
    )


async def run_demo() -> None:
    sink = SimulatedSink(realtime=True, speed=10)  # 10x real time so the demo is quick
    tts = ToneTTS()
    agent = CookingAgent(QueueSTT(), demo_llm(), tts, sink, AgentConfig(), MetricsSink(),
                         on_text=lambda who, t: print(f"[{'you' if who == 'user' else 'assistant'}] {t}"),
                         recipes=RecipeStore(":memory:"))
    announcer = asyncio.create_task(agent._announcer())
    for line in ["find me a fried rice recipe", "next step", "not yet", "ready", "ready",
                 "set a rice timer for a few seconds", "say it again"]:
        await agent.submit_text(line)
    await asyncio.sleep(1.5)  # rice timer (1.2 s) fires while idle -> standalone announcement
    while agent._output:
        await asyncio.sleep(0.05)
    print("-- simulating barge-in during a long reply --")
    task = agent.on_final("tell me something long")
    await asyncio.sleep(0.1)
    agent.barge_in()
    await asyncio.gather(task, return_exceptions=True)
    await agent.submit_text("go back")
    print("-- dictating a recipe, then cooking it from the recipe book --")
    for line in ["I want to save my own recipe", "1 litre milk, a quarter cup of rice, 3 tbsp sugar",
                 "boil the milk, add the rice and simmer for 40 minutes, then stir in the sugar", "save it",
                 "make the kheer for four", "ready"]:
        await agent.submit_text(line)
    print("   ingredients for 4:", "; ".join(agent.session.ingredient_lines()))
    await agent.submit_text("bye")
    announcer.cancel()
    await agent.timers.aclose()
    print(f"-- tts segments spoken: {len(tts.spoken)}, turns: {len(agent.metrics.turns)}, "
          f"interrupted turns: {sum(t['interrupted'] for t in agent.metrics.turns)}")
