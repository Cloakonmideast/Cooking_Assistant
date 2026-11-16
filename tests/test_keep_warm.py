"""Keeping the local model loaded while cooking (Phase 2, Task 11)."""

import asyncio
from dataclasses import dataclass

from cooking_agent.agent import AgentConfig, CookingAgent, keep_warm_due
from cooking_agent.audio import SimulatedSink, silence
from cooking_agent.providers.base import FRAME_BYTES
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ScriptedSTT, ToneTTS


# ------------------------------------------------------------ the decision (pure)
def test_keep_warm_pings_every_interval():
    assert keep_warm_due(now=900, last_ping=0, deadlines={}, pinged=set(), every_s=900, before_s=60) == (True, set())
    assert keep_warm_due(now=899, last_ping=0, deadlines={}, pinged=set(), every_s=900, before_s=60) == (False, set())


def test_keep_warm_pings_once_before_each_timer_ends():
    deadlines = {"rice": 1000.0}
    assert keep_warm_due(now=900, last_ping=800, deadlines=deadlines, pinged=set(), every_s=900,
                         before_s=60) == (False, set())
    due, keys = keep_warm_due(now=945, last_ping=800, deadlines=deadlines, pinged=set(), every_s=900, before_s=60)
    assert due and keys == {("rice", 1000.0)}
    # already pinged for this timer: not again
    assert keep_warm_due(now=950, last_ping=945, deadlines=deadlines, pinged=keys, every_s=900,
                         before_s=60) == (False, set())
    # the same name set again is a new timer: pinged again
    due, keys = keep_warm_due(now=1945, last_ping=1900, deadlines={"rice": 2000.0}, pinged=keys, every_s=900,
                              before_s=60)
    assert due and keys == {("rice", 2000.0)}


def test_finished_timers_are_ignored():
    assert keep_warm_due(now=1100, last_ping=1000, deadlines={}, pinged=set(), every_s=900,
                         before_s=60) == (False, set())


# ------------------------------------------------------------ the loop
@dataclass
class WarmLLM(ScriptedLLM):
    warmups: int = 0
    fail: bool = False

    async def warmup(self) -> None:
        self.warmups += 1
        if self.fail:
            raise ConnectionError("ollama not running")


def agent_with(llm, **cfg):
    cfg = {"keep_warm_every_s": 0.001, "keep_warm_poll_s": 0.001, **cfg}
    tts = ToneTTS()
    return CookingAgent(QueueSTT(), llm, tts, SimulatedSink(), AgentConfig(**cfg)), tts


async def run_briefly(coro, seconds=0.1):
    task = asyncio.create_task(coro)
    await asyncio.sleep(seconds)
    done = task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return done


async def test_keep_warm_loop_pings_repeatedly():
    llm = WarmLLM()
    agent, _ = agent_with(llm)
    await run_briefly(agent.keep_warm())
    assert llm.warmups >= 2


async def test_keep_warm_pings_before_a_timer_ends_even_if_recently_used():
    llm = WarmLLM()
    agent, _ = agent_with(llm, keep_warm_every_s=3600, keep_warm_before_timer_s=60)
    agent.timers.set("rice", 30)  # ends within the minute
    await run_briefly(agent.keep_warm())
    await agent.timers.aclose()
    assert llm.warmups == 1


async def test_keep_warm_failure_is_silent():
    llm = WarmLLM(fail=True)
    agent, tts = agent_with(llm)
    finished_early = await run_briefly(agent.keep_warm())
    assert not finished_early and llm.warmups >= 2  # kept trying, didn't crash
    assert tts.spoken == []


async def test_no_warmup_method_no_loop():
    agent, _ = agent_with(ScriptedLLM())
    assert await run_briefly(agent.keep_warm(), seconds=0.05)  # returned straight away


async def test_disabled_with_zero_interval():
    llm = WarmLLM()
    agent, _ = agent_with(llm, keep_warm_every_s=0)
    assert await run_briefly(agent.keep_warm(), seconds=0.05)
    assert llm.warmups == 0


async def test_run_keeps_the_model_warm():
    llm = WarmLLM()
    agent = CookingAgent(ScriptedSTT([]), llm, ToneTTS(), SimulatedSink(),
                         AgentConfig(keep_warm_every_s=0.001, keep_warm_poll_s=0.001))

    async def mic():
        for _ in range(20):
            yield silence(FRAME_BYTES)
            await asyncio.sleep(0.005)

    await asyncio.wait_for(agent.run(mic()), 5)
    assert llm.warmups >= 1


# ------------------------------------------------------------ the ping must not reload the model
async def test_ollama_ping_uses_the_same_context_size_as_chat():
    """Ollama reloads a model whose num_ctx changes: a ping at the default
    4096 made the next reply wait ~6 s for a reload (measured on qwen3:8b)."""
    from cooking_agent.providers.ollama import OllamaLLM

    class FakeClient:
        def __init__(self):
            self.generated = []

        async def generate(self, **kwargs):
            self.generated.append(kwargs)
            return {"response": ""}

    client = FakeClient()
    llm = OllamaLLM(model="qwen3:8b", client=client, num_ctx=8192)
    await llm.warmup()
    assert client.generated[0]["options"]["num_ctx"] == 8192
