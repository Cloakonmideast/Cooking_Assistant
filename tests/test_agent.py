"""End-to-end agent behaviour with fake STT/LLM/TTS and a simulated speaker."""

import asyncio

import numpy as np
import pytest

from cooking_agent.agent import FILLER_GET_RECIPE, AgentConfig, CookingAgent
from cooking_agent.audio import MicGate, SimulatedSink, silence
from cooking_agent.metrics import TurnMetrics
from cooking_agent.providers.base import FRAME_BYTES, FRAME_SAMPLES, STT_SAMPLE_RATE, ToolCall
from cooking_agent.providers.fake import QueueSTT, ScriptedLLM, ScriptedSTT, ToneTTS
from cooking_agent.state import TurnState


def make_agent(recipe, *, script=None, realtime=False, speed=1.0, llm_delay=0.0, first_delay=0.0, **cfg):
    llm = ScriptedLLM(recipe=recipe, delay_s=llm_delay, first_token_delay_s=first_delay, script=script or {
        "recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})],
        "next": [ToolCall("next_step", {})],
        "stop the timer": [ToolCall("cancel_timer", {"name": "rice"})],  # before "timer": first match wins
        "timer": [ToolCall("set_timer", {"name": "rice", "minutes": 0.001})],
        "long": ["One two three four five. Six seven eight nine ten. Eleven twelve thirteen. "
                 "Fourteen fifteen sixteen. Seventeen eighteen nineteen twenty."],
        "bye": [ToolCall("end_session", {})],
    })
    sink = SimulatedSink(realtime=realtime, speed=speed)
    tts = ToneTTS()
    agent = CookingAgent(QueueSTT(), llm, tts, sink, AgentConfig(**cfg))
    return agent, llm, tts, sink


async def test_tool_turn_updates_state_and_history(recipe):
    agent, llm, tts, _ = make_agent(recipe)
    await agent.submit_text("find a jeera rice recipe")
    await agent.submit_text("next step please")
    assert agent.session.step_index == 0
    assert tts.spoken[-2:] == ["Step 1. Soak the rice for 20 minutes.",
                               "Do you want me to start a timer for 20 minutes?"]
    roles = [m.role for m in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant"] * 2
    assert agent.metrics.turns[1]["tool_calls"] == ["next_step"]
    assert agent.turn.state is TurnState.LISTENING


async def test_get_recipe_speaks_filler_when_model_says_nothing_first(recipe):
    agent, _, tts, _ = make_agent(recipe)
    await agent.submit_text("recipe for jeera rice")
    assert tts.spoken[0] == FILLER_GET_RECIPE
    assert agent.metrics.turns[0]["first_audio_was_filler"] is True


async def test_stop_the_timer_does_not_end_session(recipe):
    """Regression for the old keyword matcher, where 'stop' anywhere quit the app."""
    agent, _, _, _ = make_agent(recipe)
    agent.timers.set("rice", 60)
    await agent.submit_text("stop the timer")
    assert not agent.stopped and agent.timers.list() == []
    await agent.submit_text("bye")
    assert agent.stopped


async def test_tts_starts_before_llm_finishes(recipe):
    """Streaming overlap: first audio must arrive long before the LLM is done."""
    agent, _, _, _ = make_agent(recipe, llm_delay=0.02)  # 20 words -> ~0.4 s of generation
    await agent.submit_text("say something long")
    t = agent.metrics.turns[0]
    assert t["first_audio_ms"] < 0.4 * t["turn_total_ms"]


async def test_timer_announcement_is_spliced_into_active_reply(recipe):
    agent, _, tts, _ = make_agent(recipe, llm_delay=0.02)
    announcer = asyncio.create_task(agent._announcer())
    agent.timers.set("pasta", 0.15)  # fires after sentence 1, while the reply is still generating
    await agent.submit_text("tell me something long")
    announcer.cancel()
    assert "Your pasta timer is done." in tts.spoken
    idx = tts.spoken.index("Your pasta timer is done.")
    assert 0 < idx < len(tts.spoken) - 1  # in the middle, at a sentence boundary
    assert len(agent.metrics.turns) == 1


async def test_timer_announcement_when_idle_is_standalone(recipe):
    agent, _, tts, _ = make_agent(recipe)
    announcer = asyncio.create_task(agent._announcer())
    agent.timers.set("rice", 0.01)
    await asyncio.sleep(0.1)
    announcer.cancel()
    assert tts.spoken == ["Your rice timer is done."]
    assert agent._output is None and agent.turn.state is TurnState.LISTENING


async def test_new_utterance_cancels_current_turn_and_keeps_history_valid(recipe):
    agent, _, _, sink = make_agent(recipe, realtime=True, speed=4)
    first = agent.on_final("tell me something long")
    await asyncio.sleep(0.15)
    assert agent.turn.state is TurnState.SPEAKING
    second = agent.on_final("next step")
    await asyncio.gather(first, return_exceptions=True)
    await second
    assert first.cancelled()
    assert agent.metrics.turns[0]["interrupted"] is True
    assert sink.stopped >= 1
    # The LLM had finished writing before playback was cut, so the full text is in
    # history; the next user message tells the model it was interrupted.
    assert [m.role for m in agent.history[:2]] == ["user", "assistant"]
    assert agent.history[2].role == "user"
    assert agent.history[2].text.startswith("[The user interrupted")
    assert agent.history[2].text.endswith("next step")


async def test_interrupt_mid_tool_round_drops_dangling_call(recipe):
    agent, llm, _, _ = make_agent(recipe)
    agent.session.load_recipe(recipe)
    gate = asyncio.Event()
    orig = agent.tools.dispatch

    async def slow_dispatch(call):
        gate.set()
        await asyncio.sleep(10)
        return await orig(call)

    agent.tools.dispatch = slow_dispatch
    task = agent.on_final("next step")
    await gate.wait()
    agent.barge_in()
    await asyncio.gather(task, return_exceptions=True)
    # No assistant function-call message without its tool result.
    assert [m.role for m in agent.history] == ["user", "assistant"]
    assert not agent.history[1].tool_calls


# ------------------------------------------------------------ mic gate / barge-in
def tone(dbfs_level: float, n_frames: int) -> list[bytes]:
    amp = 10 ** (dbfs_level / 20) * np.sqrt(2) * 32767
    t = np.arange(FRAME_SAMPLES * n_frames) / STT_SAMPLE_RATE
    pcm = (np.sin(2 * np.pi * 300 * t) * amp).astype(np.int16).tobytes()
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


class FakePlayingSink:
    def __init__(self, playing=True, ref=0.0):
        self.is_playing = playing
        self.last_played_at = 0.0
        self.reference_rms = ref


def test_gate_passes_audio_when_not_playing():
    g = MicGate(FakePlayingSink(playing=False), on_barge_in=lambda: None)
    f = tone(-20, 1)[0]
    assert g.filter(f) == f


def test_gate_silences_echo_level_audio_while_playing():
    hits = []
    # speaker reference at -30 dBFS, mic hears echo at -32 -> below ratio, gated
    g = MicGate(FakePlayingSink(ref=10 ** (-30 / 20)), on_barge_in=lambda: hits.append(1))
    out = [g.filter(f) for f in tone(-32, 50)]
    assert all(o == silence(FRAME_BYTES) for o in out)
    assert hits == [] and g.gated_frames == 50


def test_gate_detects_barge_in_and_releases_preroll():
    hits = []
    g = MicGate(FakePlayingSink(ref=10 ** (-40 / 20)), on_barge_in=lambda: hits.append(1),
                min_ms=200, preroll_ms=400)
    out = [g.filter(f) for f in tone(-15, 10)]
    assert hits == [1]
    assert all(o == silence(FRAME_BYTES) for o in out[:9])
    assert len(out[9]) == 10 * FRAME_BYTES  # all 10 frames (200 ms) released at once


def test_gate_short_blip_is_not_barge_in():
    hits = []
    g = MicGate(FakePlayingSink(), on_barge_in=lambda: hits.append(1), min_ms=200)
    frames = tone(-15, 5) + [silence(FRAME_BYTES)] * 20 + tone(-15, 5)
    for f in frames:
        g.filter(f)
    assert hits == []


def test_gate_disabled_never_barges_in():
    hits = []
    g = MicGate(FakePlayingSink(), on_barge_in=lambda: hits.append(1), enabled=False)
    for f in tone(-10, 50):
        assert g.filter(f) == silence(FRAME_BYTES)
    assert hits == []


async def test_barge_in_through_audio_pipeline(recipe):
    """Mic frames -> gate -> STT. Loud speech during playback stops the reply."""
    agent, _, tts, sink = make_agent(recipe, realtime=True, speed=1)
    agent.stt = ScriptedSTT(["tell me something long"], frames_per_utterance=3)

    async def mic():
        for f in [silence(FRAME_BYTES)] * 3:  # triggers the scripted utterance
            yield f
        while not sink.is_playing:
            await asyncio.sleep(0.01)
            yield silence(FRAME_BYTES)
        for f in tone(-10, 15):  # user talks over the assistant
            await asyncio.sleep(0.005)
            yield f

    await asyncio.wait_for(agent.run(mic()), 5)
    assert agent.gate.barge_ins == 1
    assert agent.metrics.turns[0]["interrupted"] is True
    assert sink.stopped >= 1


# ------------------------------------------------------------ wake word
async def test_wake_word_and_follow_up_window(recipe):
    agent, _, tts, _ = make_agent(recipe, wake_word="chef", follow_up_window_s=0.2)
    agent.session.load_recipe(recipe)
    await agent.submit_text("next step")  # ignored: no wake word, never spoke
    assert agent.metrics.turns == []
    await agent.submit_text("Chef, next step")
    assert agent.metrics.turns[0]["user_text"] == "next step"
    await agent.submit_text("next step")  # within follow-up window
    assert len(agent.metrics.turns) == 2
    await asyncio.sleep(0.25)
    await agent.submit_text("next step")  # window expired
    assert len(agent.metrics.turns) == 2


# ------------------------------------------------------------ metrics
def test_metrics_summary_math():
    m = TurnMetrics(turn_id=1, user_text="x", t_speech_end=10.0, t_transcript=10.3)
    m.t_first_llm_event, m.t_first_audio, m.t_done = 10.8, 11.1, 12.0
    s = m.summary()
    assert (s["stt_endpoint_ms"], s["llm_first_token_ms"], s["first_audio_ms"], s["e2e_first_audio_ms"]) == \
        (300.0, 500.0, 800.0, 1100.0)


def test_percentile():
    from cooking_agent.metrics import percentile

    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95 and percentile([7], 95) == 7


# ------------------------------------------------------------------ step pacing
def pacing_agent(recipe):
    return make_agent(recipe, script={
        "recipe": [ToolCall("get_recipe", {"dish": "jeera rice"})],
        "start it": ["Okay.", ToolCall("start_step_timer", {"name": "rice"})],
        "no thanks": ["No problem."],
        "not yet": ["No problem, take your time."],
        "ready": [ToolCall("next_step", {})],
        "question": ["Is the water boiling?"],
    })


async def test_step_with_time_offers_timer_then_asks_if_ready(recipe):
    from cooking_agent.state import Awaiting
    from cooking_agent.tools import READY_QUESTION

    agent, _, tts, _ = pacing_agent(recipe)
    await agent.submit_text("recipe please")
    await agent.submit_text("ready")
    assert tts.spoken[-1] == "Do you want me to start a timer for 20 minutes?"
    assert agent.session.awaiting is Awaiting.TIMER
    assert agent.session.snapshot()["awaiting_answer_to"] == "start_timer"

    await agent.submit_text("yes start it")
    assert agent.timers.list()[0]["name"] == "rice"
    assert tts.spoken[-1] == READY_QUESTION and agent.session.awaiting is Awaiting.NEXT_STEP
    # the question is part of the history, so the model knows what "yes" answers
    assert agent.history[-1].text.endswith(READY_QUESTION)

    n = len(tts.spoken)
    await agent.submit_text("not yet")
    assert tts.spoken[n:] == ["No problem, take your time."]  # not nagged again
    assert agent.session.step_index == 0

    await agent.submit_text("ready")  # step 2 has no time -> straight to "ready?"
    assert agent.session.step_index == 1
    assert tts.spoken[-1] == READY_QUESTION
    await agent.timers.aclose()


async def test_declining_the_timer_moves_on_to_ready_question(recipe):
    from cooking_agent.tools import READY_QUESTION

    agent, _, tts, _ = pacing_agent(recipe)
    await agent.submit_text("recipe please")
    await agent.submit_text("ready")
    await agent.submit_text("no thanks")
    assert tts.spoken[-2:] == ["No problem.", READY_QUESTION]
    assert agent.timers.list() == []


async def test_last_step_offers_timer_but_not_another_step(recipe):
    agent, _, tts, _ = pacing_agent(recipe)
    await agent.submit_text("recipe please")
    for _ in range(3):
        await agent.submit_text("ready")
    assert agent.session.on_last_step
    assert tts.spoken[-1] == "Do you want me to start a timer for 12 minutes?"
    await agent.submit_text("yes start it")
    assert tts.spoken[-1] == "Okay, rice timer started for 12 minutes."  # no "ready for the next step?"
    assert agent.session.awaiting is None
    await agent.timers.aclose()


async def test_no_second_question_when_model_already_asked_one(recipe):
    agent, llm, tts, _ = pacing_agent(recipe)
    await agent.submit_text("recipe please")
    await agent.submit_text("ready")
    llm._describe = lambda name, result: "Timer started. Ready for the next step?"
    await agent.submit_text("yes start it")
    assert tts.spoken[-1] == "Ready for the next step?"
    assert tts.spoken.count("Are you ready for the next step?") == 0
    assert agent.session.awaiting is not None  # "yes" still routes to next_step
    await agent.timers.aclose()


async def test_steps_are_read_word_for_word_without_another_llm_round(recipe):
    agent, llm, tts, _ = pacing_agent(recipe)
    await agent.submit_text("recipe please")
    rounds = len(llm.calls)
    await agent.submit_text("ready")
    assert len(llm.calls) == rounds + 1  # only the round that chose next_step
    assert "Step 1. Soak the rice for 20 minutes." in tts.spoken
    await agent.submit_text("ready")
    await agent.submit_text("ready")
    assert "Step 3, the last step. Add rice, water and salt; cook covered for 12 minutes." in tts.spoken
    # history stays well-formed: call, result, then what was said
    assert [m.role for m in agent.history[-3:]] == ["assistant", "tool", "assistant"]
    assert agent.history[-1].text.startswith("Step 3, the last step.")
