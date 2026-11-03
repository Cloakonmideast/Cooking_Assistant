import asyncio

import pytest

from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox

REQUIRED = {"get_recipe", "next_step", "previous_step", "repeat_step", "list_ingredients", "set_timer",
            "cancel_timer", "list_timers", "scale_servings", "substitute_ingredient"}


@pytest.fixture
def box(recipe):
    fired = []
    tb = ToolBox(CookingSession(), TimerManager(on_fire=fired.append), ScriptedLLM(recipe=recipe))
    tb.fired = fired
    return tb


async def call(tb, tool_name, /, **args):
    return (await tb.dispatch(ToolCall(tool_name, args))).result


def test_all_required_tools_have_clean_schemas(box):
    specs = {s.name: s for s in box.specs()}
    assert REQUIRED <= specs.keys()
    for s in specs.values():
        assert s.parameters["type"] == "object"
        assert "title" not in s.parameters
        assert s.description
    assert specs["set_timer"].parameters["required"] == ["name", "minutes"]
    assert specs["set_timer"].parameters["properties"]["minutes"]["exclusiveMinimum"] == 0


async def test_get_recipe_loads_validated_recipe(box):
    out = await call(box, "get_recipe", dish="jeera rice", servings=2)
    assert out["title"] == "Jeera Rice" and out["step_count"] == 3
    assert box.session.recipe is not None


async def test_navigation_tools(box):
    assert "error" in await call(box, "next_step")  # no recipe yet
    await call(box, "get_recipe", dish="jeera rice")
    assert (await call(box, "next_step"))["step_number"] == 1
    assert (await call(box, "next_step"))["step_number"] == 2
    assert (await call(box, "repeat_step"))["step_number"] == 2
    assert (await call(box, "previous_step"))["step_number"] == 1
    assert "first step" in (await call(box, "previous_step"))["error"]


async def test_ingredient_tools(box):
    await call(box, "get_recipe", dish="jeera rice")
    assert (await call(box, "list_ingredients"))["ingredients"][0] == "1 cup basmati rice, rinsed"
    assert (await call(box, "scale_servings", servings=4))["factor"] == 2
    out = await call(box, "substitute_ingredient", ingredient="ghee", replacement="butter", quantity=1.5, unit="tbsp")
    assert out["line"] == "3 tbsp butter, instead of ghee"


async def test_timer_tools(box):
    assert (await call(box, "set_timer", name="rice", minutes=0.001))["name"] == "rice"
    await call(box, "set_timer", name="dal", minutes=30)
    names = [t["name"] for t in (await call(box, "list_timers"))["timers"]]
    assert names == ["rice", "dal"]
    await asyncio.sleep(0.12)
    assert box.fired == ["rice"]
    assert await call(box, "cancel_timer", name="dal timer") == {"cancelled": "dal"}
    assert "no running timer" in (await call(box, "cancel_timer", name="dal"))["error"]


@pytest.mark.parametrize("name,args,needle", [
    ("set_timer", {"name": "rice"}, "minutes: Field required"),
    ("set_timer", {"name": "rice", "minutes": -1}, "greater than 0"),
    ("set_timer", {"name": "rice", "minutes": "soon"}, "minutes"),
    ("scale_servings", {"servings": 0}, "greater than or equal to 1"),
    ("get_recipe", {}, "dish: Field required"),
])
async def test_bad_arguments_return_errors_not_exceptions(box, name, args, needle):
    out = await call(box, name, **args)
    assert needle in out["error"]


async def test_unknown_tool(box):
    assert "Unknown tool" in (await call(box, "launch_rocket"))["error"]


async def test_provider_failure_becomes_error_result(box):
    box.llm.recipe = None  # ScriptedLLM raises
    out = await call(box, "get_recipe", dish="anything")
    assert out == {"error": "get_recipe failed: RuntimeError"}


async def test_end_session_only_via_tool(box):
    assert not box.end_requested
    await call(box, "end_session")
    assert box.end_requested
    await box.timers.aclose()
