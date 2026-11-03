"""The real-time voice agent.

Concurrency model: everything runs on one asyncio event loop. Shared state
(cooking session, timers, conversation history, turn state) is only touched by
coroutines on that loop, so there are no locks and no data races. The only
other threads are the PortAudio callbacks (which hand frames to the loop with
``call_soon_threadsafe``) and ``asyncio.to_thread`` workers in the offline
providers (which touch no shared state).

Pipeline for one user turn::

    mic frames -> MicGate -> STT (streaming) --final--> LLM (streaming, tools)
        -> SentenceSegmenter -> segment queue -> TTS (streaming) -> speaker

The LLM producer and the TTS/playback consumer run concurrently, so the first
sentence is being spoken while the LLM is still generating the rest.

At most one *output task* (a user turn, or a standalone timer announcement)
is active at a time. A new final transcript or a barge-in cancels it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from .audio import AudioSink, MicGate
from .metrics import MetricsSink, TurnMetrics, now
from .providers.base import (
    LLMProvider,
    Message,
    STTProvider,
    SttEvent,
    TextDelta,
    ToolCallEvent,
    ToolResult,
    TTSProvider,
    TurnComplete,
)
from .preferences import PreferenceStore
from .recipe_store import RecipeStore
from .segmenter import SentenceSegmenter
from .state import CookingSession, TurnState, TurnStateMachine
from .timers import TimerManager
from .tools import ToolBox

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a hands-free voice cooking assistant. The cook is busy, so everything \
you say is spoken aloud.

Speaking style:
- Reply in one to three short sentences of plain spoken English. No markdown, \
lists, emojis or symbols.
- When reading a step, say the step number and the instruction. Do not read \
ahead unless asked.
- After reading a step or starting a step timer, do NOT ask a follow-up \
question: the app itself asks "Do you want me to start a timer for N minutes?" \
or "Are you ready for the next step?" right after your reply.
- When listing ingredients, read at most six at a time and offer the rest.

Steps and timers:
- The cook moves on only when they say so. Never call next_step on your own.
- awaiting_answer_to in the state says which question the app just asked. \
If it is start_timer and the cook agrees ("yes", "sure", "go ahead"), call \
start_step_timer. If it is next_step and the cook agrees ("yes", "ready", \
"done"), call next_step. If they say no or not yet, reply briefly (e.g. "No \
problem, take your time.") and call nothing.
- For a timer with a different length than the step's, use set_timer.
- "Stop" or "cancel" about a timer means cancel_timer, never end_session. \
Only call end_session if the cook clearly wants to quit the assistant.

Recipes:
- get_recipe checks the cook's saved recipe book first, then writes a new \
recipe. Pass servings when the cook says how many people. Mention whether \
the recipe came from their recipe book.
- Before calling get_recipe, say one short sentence first (for example \
"Sure, let me find a recipe for that."), because it can take a few seconds.
- The cook can save the current recipe (save_recipe) or dictate their own. \
For dictation: start_recipe_dictation with the title (ask for it if not \
given); then, every time the cook dictates, call add_to_dictated_recipe with \
exactly what they said, as ingredient lines and/or steps, and answer in two \
or three words ("Got it." / "Okay, next?"). Ask how many people it serves if \
they haven't said. Fix mistakes with remove_from_dictated_recipe. When they \
say they are done, call read_dictated_recipe, read back the title and the \
number of ingredients and steps, and ask if you should save it; only then \
call save_dictated_recipe.

First-time setup:
- If first_time_setup is in the state, the app has just asked the cook: \
"Do you have any food allergies, or conditions like lactose intolerance, \
celiac disease or diabetes? And do you follow a diet?" For each thing they \
mention, call remember_preference (allergy, condition or diet; also dislike \
or pantry if they volunteer it), then call finish_setup. "None" means \
finish_setup straight away; "skip" or "later" means finish_setup with \
skipped true. Then confirm in one sentence what you saved, say it can be \
changed any time, and ask what they'd like to cook.
- If the answer is unclear (e.g. "dairy problems"), ask one short follow-up \
before saving. If they ask for something else instead, help them; setup can \
wait.

Safety:
- If any tool result has warnings, say them first, clearly and briefly \
(e.g. "Heads up: this recipe has paneer, which has lactose."), and offer a \
substitute before going on. Never drop a warning.

About the cook:
- cook_profile in the state holds their stated preferences and habits learned \
from what they cook. Use it when suggesting dishes. get_recipe already \
tailors new recipes to it; if the result has tailored_to or servings_note, \
mention it in a few words (e.g. "I kept it vegetarian like you usually do.").
- When the cook states a lasting preference ("I'm vegetarian", "I'm allergic \
to nuts", "I don't like coriander", "I always have paneer at home"), call \
remember_preference. A one-off ("no onions today") is a get_recipe \
constraint instead. "Forget that I..." means forget_preference.
- Allergies matter for safety: if the cook asks for something containing a \
known allergen, point it out before going ahead.

General:
- Use tools for anything that reads or changes the recipe, step position, \
timers or the recipe book. Never invent a step or quantity: get it from a \
tool result.
- If a tool returns an error, explain it briefly and suggest what to say next.

Current state (authoritative, refreshed every turn):
{state}
"""

FILLER_GET_RECIPE = "One moment while I find a recipe."
SETUP_QUESTION = (
    "Hi, I'm your cooking assistant. Before we start: do you have any food allergies, or conditions like "
    "lactose intolerance, celiac disease or diabetes that I should keep in mind? And do you follow a diet, "
    "like vegetarian or vegan? If not, just say none."
)
INTERRUPTED_NOTE = "[The user interrupted your previous reply while it was being spoken.]"
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 40


@dataclass
class AgentConfig:
    wake_word: Optional[str] = None  # None = always listening
    follow_up_window_s: float = 8.0  # with a wake word: no wake word needed this soon after we spoke
    barge_in: bool = True
    barge_in_threshold_dbfs: float = -35.0
    barge_in_ratio: float = 1.5
    echo_tail_ms: int = 250


@dataclass
class _Output:
    """One active speaking task and its segment queue (for timer announcements)."""

    task: Optional[asyncio.Task] = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    accepting: bool = True
    metrics: Optional[TurnMetrics] = None


_END = object()


class CookingAgent:
    def __init__(self, stt: STTProvider, llm: LLMProvider, tts: TTSProvider, sink: AudioSink,
                 config: AgentConfig | None = None, metrics: MetricsSink | None = None,
                 on_text: Callable[[str, str], None] | None = None,
                 recipes: RecipeStore | None = None, preferences: PreferenceStore | None = None):
        self.stt, self.llm, self.tts, self.sink = stt, llm, tts, sink
        self.config = config or AgentConfig()
        self.metrics = metrics or MetricsSink()
        self.on_text = on_text or (lambda who, text: None)
        self.on_turn_end: Callable[[TurnMetrics], None] | None = None

        self.session = CookingSession()
        self.timers = TimerManager(on_fire=self._on_timer_fired)
        self.tools = ToolBox(self.session, self.timers, llm, store=recipes, prefs=preferences)
        self.turn = TurnStateMachine(on_change=lambda a, b: log.debug("turn %s -> %s", a.value, b.value))
        self.gate = MicGate(sink, on_barge_in=self.barge_in, enabled=self.config.barge_in,
                            threshold_dbfs=self.config.barge_in_threshold_dbfs,
                            ratio=self.config.barge_in_ratio, tail_ms=self.config.echo_tail_ms)
        self.history: list[Message] = []
        self._output: Optional[_Output] = None
        self._announcements: asyncio.Queue[str] = asyncio.Queue()
        self._turn_counter = 0
        self._last_spoke_at = 0.0
        self._interrupted_last_turn = False
        self._stopped = asyncio.Event()

    # ================================================================ running
    async def run(self, frames: AsyncIterator[bytes],
                  speech_end_fn: Callable[[SttEvent], Optional[float]] | None = None) -> None:
        """Run until the STT stream ends or the user ends the session."""
        announcer = asyncio.create_task(self._announcer(), name="announcer")
        stt_iter = self.stt.stream(self.gate.process(frames))
        stop_wait = asyncio.create_task(self._stopped.wait())
        try:
            while not self._stopped.is_set():
                nxt = asyncio.ensure_future(anext(stt_iter))
                done, _ = await asyncio.wait({nxt, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                if nxt not in done:
                    nxt.cancel()
                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                        await nxt
                    break
                try:
                    ev = nxt.result()
                except StopAsyncIteration:
                    break
                if ev.kind == "final":
                    t_end = speech_end_fn(ev) if speech_end_fn else None
                    self.on_final(ev.text, t_speech_end=t_end)
            # STT ended (e.g. end of WAV fixture): let the in-flight turn finish.
            if self._output and self._output.task:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._output.task
        finally:
            stop_wait.cancel()
            await self._cancel_output()
            announcer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await announcer
            await self.timers.aclose()
            with contextlib.suppress(Exception):
                await stt_iter.aclose()
            if self.turn.state != TurnState.STOPPED:
                self.turn.to(TurnState.STOPPED)

    def begin_setup(self) -> bool:
        """First run: ask about allergies, conditions and diet. Spoken as soon
        as the assistant is idle (needs the announcer running, as in ``run``)."""
        if self.tools.prefs is None:
            return False
        self.session.setup_pending = True
        self._announcements.put_nowait(SETUP_QUESTION)
        return True

    def stop(self) -> None:
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    # ================================================================ input
    def on_final(self, text: str, t_speech_end: Optional[float] = None) -> Optional[asyncio.Task]:
        """Handle an end-of-utterance transcript. Returns the turn task, if one was started."""
        text = text.strip()
        if not text or self.stopped:
            return None
        accepted = self._apply_wake_word(text)
        if accepted is None:
            log.info("ignored (no wake word): %r", text)
            return None
        self.on_text("user", text)
        # A new utterance supersedes whatever we were doing.
        self._interrupt_output(reason="new utterance")
        self._turn_counter += 1
        m = TurnMetrics(turn_id=self._turn_counter, user_text=accepted, t_speech_end=t_speech_end,
                        providers={"stt": self.stt.name, "llm": self.llm.name, "tts": self.tts.name})
        self.turn.to(TurnState.THINKING)
        out = _Output(metrics=m)
        # The LLM may have finished writing a reply the user only partly heard;
        # tell it so on the next turn (provider-neutral; no history rewriting).
        llm_text = f"{INTERRUPTED_NOTE} {accepted}" if self._interrupted_last_turn else accepted
        self._interrupted_last_turn = False
        out.task = asyncio.create_task(self._run_turn(llm_text, out), name=f"turn-{m.turn_id}")
        self._output = out
        return out.task

    async def submit_text(self, text: str) -> None:
        """Text-mode entry point (``chat`` command and tests): run one turn to completion."""
        task = self.on_final(text)
        if task:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _apply_wake_word(self, text: str) -> Optional[str]:
        ww = (self.config.wake_word or "").lower().strip()
        if not ww:
            return text
        lower = text.lower()
        idx = lower.find(ww)
        if idx != -1:
            rest = text[idx + len(ww):].strip(" ,.!?")
            return rest or "Hello"
        if time.perf_counter() - self._last_spoke_at <= self.config.follow_up_window_s:
            return text
        return None

    # ================================================================ barge-in
    def barge_in(self) -> None:
        """Called by the MicGate when the user talks over the assistant."""
        if self._output is None:
            return
        log.info("barge-in: stopping playback")
        self._interrupt_output(reason="barge-in")

    def _interrupt_output(self, reason: str) -> None:
        out = self._output
        self.sink.stop()
        if out is None or out.task is None or out.task.done():
            self._output = None
            return
        if out.metrics:
            out.metrics.interrupted = True
            self._interrupted_last_turn = True
        out.accepting = False
        out.task.cancel()
        self._output = None
        log.debug("cancelled output task (%s)", reason)
        if self.turn.state in (TurnState.THINKING, TurnState.SPEAKING):
            self.turn.to(TurnState.LISTENING)

    async def _cancel_output(self) -> None:
        out = self._output
        self._interrupt_output(reason="shutdown")
        if out and out.task:
            with contextlib.suppress(asyncio.CancelledError):
                await out.task

    # ================================================================ timers
    def _on_timer_fired(self, name: str) -> None:
        self._announcements.put_nowait(f"Your {name} timer is done.")

    async def _announcer(self) -> None:
        """Route timer announcements: into the active reply (spoken at the next
        sentence boundary) or, if idle, as a standalone utterance."""
        while True:
            text = await self._announcements.get()
            out = self._output
            if out and out.accepting and out.task and not out.task.done():
                out.queue.put_nowait(text)
                continue
            new = _Output()
            new.task = asyncio.create_task(self._speak_only(text, new), name="announcement")
            self._output = new

    async def _speak_only(self, text: str, out: _Output) -> None:
        self.on_text("assistant", text)
        out.queue.put_nowait(text)
        out.queue.put_nowait(_END)
        try:
            await self._play(out)
        finally:
            if self._output is out:
                self._output = None
                if self.turn.state == TurnState.SPEAKING:
                    self.turn.to(TurnState.LISTENING)

    # ================================================================ turn
    async def _run_turn(self, user_text: str, out: _Output) -> None:
        m = out.metrics
        assert m is not None
        producer = asyncio.create_task(self._think(user_text, out), name=f"llm-{m.turn_id}")
        try:
            await self._play(out)
            await producer  # surface LLM/tool errors
        except asyncio.CancelledError:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer
            raise
        except Exception:
            log.exception("turn %d failed", m.turn_id)
            producer.cancel()
            await self._say_error(out)
        finally:
            m.t_done = now()
            self.metrics.record(m)
            if self.on_turn_end:
                self.on_turn_end(m)
            if self._output is out:
                self._output = None
                if self.turn.state in (TurnState.THINKING, TurnState.SPEAKING):
                    self.turn.to(TurnState.LISTENING)
            if self.tools.end_requested:
                self.stop()

    async def _say_error(self, out: _Output) -> None:
        q = asyncio.Queue()
        q.put_nowait("Sorry, something went wrong on my side. Please try again.")
        q.put_nowait(_END)
        with contextlib.suppress(Exception):
            await self._play(_Output(queue=q))

    def _system_prompt(self) -> str:
        state = self.session.snapshot()
        state["timers"] = self.timers.list()
        if self.timers.recently_finished():
            state["finished_timers"] = self.timers.recently_finished()
        if self.tools.prefs is not None:
            state["cook_profile"] = self.tools.prefs.profile()
        return SYSTEM_PROMPT.format(state=json.dumps(state))

    async def _think(self, user_text: str, out: _Output) -> None:
        """LLM producer: stream model rounds, run tool calls, push speakable segments."""
        m = out.metrics
        assert m is not None
        q = out.queue
        committed: list[Message] = [Message("user", user_text)]
        spoken: list[str] = []
        all_results: list[ToolResult] = []
        gave_up = False

        def emit(seg: str) -> None:
            spoken.append(seg)
            q.put_nowait(seg)

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                m.llm_rounds += 1
                seg = SentenceSegmenter()
                calls = []
                reply: Optional[Message] = None
                async for ev in self.llm.stream(self._system_prompt(), self.history + committed, self.tools.specs()):
                    m.mark_llm_event()
                    if isinstance(ev, TextDelta):
                        for s in seg.push(ev.text):
                            emit(s)
                    elif isinstance(ev, ToolCallEvent):
                        calls.append(ev.call)
                        for s in seg.flush():  # speak what came before the tool call now
                            emit(s)
                    elif isinstance(ev, TurnComplete):
                        reply = ev.message
                for s in seg.flush():
                    emit(s)
                if reply is None:
                    raise RuntimeError("LLM stream ended without TurnComplete")
                if reply.text:
                    self.on_text("assistant", reply.text)
                if not calls:
                    committed.append(reply)
                    break
                if any(c.name == "get_recipe" for c in calls) and not spoken:
                    emit(FILLER_GET_RECIPE)
                    m.first_audio_was_filler = True  # refined in _play if audio came earlier
                results = []
                for c in calls:  # sequential on purpose: order matters (e.g. next_step twice)
                    m.tool_calls.append(c.name)
                    results.append(await self.tools.dispatch(c))
                all_results += results
                committed.append(reply)
                committed.append(Message("tool", tool_results=results))
            else:
                gave_up = True
                emit("Sorry, I got stuck on that one. Could you say it another way?")
                committed.append(Message("assistant", "(gave up after too many tool calls)"))
            if not gave_up:
                self._ask_follow_up(all_results, committed, spoken, emit)
        except asyncio.CancelledError:
            self._commit_interrupted(committed, spoken)
            raise
        else:
            self._commit(committed)
        finally:
            q.put_nowait(_END)

    def _ask_follow_up(self, results: list[ToolResult], committed: list[Message], spoken: list[str],
                       emit: Callable[[str], None]) -> None:
        """Append the app's own question after a step ("start a timer?" /
        "ready for the next step?"). Done here rather than left to the model
        so the cook is asked every time, in the same words."""
        question = self.tools.follow_up(results)
        if not question:
            return
        if " ".join(spoken).rstrip().endswith("?"):
            return  # the model already ended on a question; don't stack a second one
        emit(question)
        self.on_text("assistant", question)
        last = committed[-1]  # the final reply of the turn, text only
        committed[-1] = Message("assistant", f"{last.text} {question}".strip())

    def _commit(self, msgs: list[Message]) -> None:
        self.history.extend(msgs)
        if len(self.history) > MAX_HISTORY_MESSAGES:
            # Trim from the front, keeping the history starting at a user message
            # so tool calls are never separated from their results.
            cut = len(self.history) - MAX_HISTORY_MESSAGES
            while cut < len(self.history) and self.history[cut].role != "user":
                cut += 1
            self.history = self.history[cut:]

    def _commit_interrupted(self, msgs: list[Message], spoken: list[str]) -> None:
        """Keep the history well-formed after a cancelled turn: keep complete
        (call, result) rounds, drop a half-finished round, and record what had
        been said so the model knows it was cut off."""
        complete = [msgs[0]]
        i = 1
        while i < len(msgs):
            if msgs[i].tool_calls and i + 1 < len(msgs) and msgs[i + 1].role == "tool":
                complete += msgs[i : i + 2]
                i += 2
            elif msgs[i].role == "assistant" and not msgs[i].tool_calls:
                complete.append(msgs[i])
                i += 1
            else:
                break
        if complete[-1].role != "assistant" or complete[-1].tool_calls:
            said = " ".join(spoken)
            complete.append(Message("assistant", (said + " " if said else "") + "[interrupted by the user]"))
        self._commit(complete)

    # ================================================================ playback
    async def _segments(self, out: _Output) -> AsyncIterator[str]:
        while True:
            item = await out.queue.get()
            if item is _END:
                break
            yield item
        # No awaits between here and the drain: the announcer cannot slip a
        # message in after we stop accepting, so nothing is lost.
        out.accepting = False
        while not out.queue.empty():
            item = out.queue.get_nowait()
            if item is not _END:
                yield item

    async def _play(self, out: _Output) -> None:
        m = out.metrics
        first = True
        segments = self._segments(out)
        audio = self.tts.stream(segments)
        try:
            async for chunk in audio:
                if first:
                    first = False
                    if m:
                        m.mark_audio(filler=m.first_audio_was_filler)
                    if self.turn.state in (TurnState.LISTENING, TurnState.THINKING):
                        self.turn.to(TurnState.SPEAKING)
                await self.sink.play(chunk)
                self._last_spoke_at = time.perf_counter()
            await self.sink.drain()
            self._last_spoke_at = time.perf_counter()
        finally:
            with contextlib.suppress(Exception):
                await audio.aclose()
            with contextlib.suppress(Exception):
                await segments.aclose()
