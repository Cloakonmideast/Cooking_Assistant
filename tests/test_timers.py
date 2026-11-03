import asyncio

import pytest

from cooking_agent.timers import TimerError, TimerManager, normalize_name


async def test_concurrent_timers_fire_in_deadline_order():
    fired = []
    tm = TimerManager(on_fire=fired.append)
    tm.set("pasta", 0.06)
    tm.set("rice", 0.02)
    tm.set("Oven Timer", 0.04)
    assert tm.active_names == ["oven", "pasta", "rice"]
    await asyncio.sleep(0.12)
    assert fired == ["rice", "oven", "pasta"]
    assert tm.list() == []
    assert tm.recently_finished() == ["rice", "oven", "pasta"]


async def test_cancel_prevents_fire_and_reports_unknown():
    fired = []
    tm = TimerManager(on_fire=fired.append)
    tm.set("rice", 0.03)
    tm.set("eggs", 0.03)
    assert tm.cancel("the rice timer".replace("the ", "")) == {"cancelled": "rice"}
    await asyncio.sleep(0.06)
    assert fired == ["eggs"]
    with pytest.raises(TimerError, match="no running timer called rice"):
        tm.cancel("rice")


async def test_resetting_same_name_replaces_and_old_task_does_not_fire():
    fired = []
    tm = TimerManager(on_fire=fired.append)
    tm.set("rice", 0.02)
    out = tm.set("rice", 0.08)
    assert out["replaced_existing"] is True
    await asyncio.sleep(0.05)
    assert fired == [] and tm.active_names == ["rice"]
    await asyncio.sleep(0.06)
    assert fired == ["rice"]


async def test_async_callback_and_list_remaining():
    got = asyncio.Event()

    async def on_fire(name):
        got.set()

    tm = TimerManager(on_fire=on_fire)
    tm.set("tea", 0.02)
    tm.set("cake", 600)
    rem = {t["name"]: t["remaining_seconds"] for t in tm.list()}
    assert rem["cake"] in (599, 600) and rem["tea"] == 0
    await asyncio.wait_for(got.wait(), 1)
    await tm.aclose()
    assert tm.list() == []


@pytest.mark.parametrize("name,secs", [("", 10), ("rice", 0), ("rice", -5), ("rice", 25 * 3600)])
async def test_invalid_timers(name, secs):
    tm = TimerManager(on_fire=lambda n: None)
    with pytest.raises(TimerError):
        tm.set(name, secs)


async def test_aclose_cancels_everything():
    fired = []
    tm = TimerManager(on_fire=fired.append)
    for n in "abc":
        tm.set(n, 0.02)
    await tm.aclose()
    await asyncio.sleep(0.04)
    assert fired == []


def test_normalize_name():
    assert normalize_name("  Rice  Timer ") == "rice"
    assert normalize_name("PASTA") == "pasta"
