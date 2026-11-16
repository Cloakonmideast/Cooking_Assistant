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
from .commands import match_command, speak_result
from .metrics import MetricsSink, TurnMetrics, now
from .providers.base import (
    LLMProvider,
    Message,
    ToolCall,
    STTProvider,
    SttEvent,
    TextDelta,
    ToolCallEvent,
    ToolResult,
    TTSProvider,
    TurnComplete,
)
from .preferences import PreferenceStore, stated_diets
from .recipe_store import RecipeStore
from .segmenter import SentenceSegmenter
from .state import Awaiting, CookingSession, TurnState, TurnStateMachine
from .timers import TimerManager
from .tools import STEP_TOOLS, ToolBox

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a hands-free voice cooking assistant. The cook is busy, so everything \
you say is spoken aloud.

Speaking style:
- Reply in one to three short sentences of plain spoken English. No markdown, \
lists, emojis or symbols.
- The app reads steps aloud itself, word for word, when you call next_step, \
previous_step or repeat_step; you will not be asked to speak after those. \
Do not read ahead unless asked.
- Read ingredient lines exactly as the tool gives them; don't reword them.
- After starting a step timer, do NOT ask a follow-up question: the app \
itself asks "Are you ready for the next step?" right after your reply.
- When listing ingredients, read at most six at a time and offer the rest.

Steps and timers:
- The cook moves on only when they say so. Never call next_step on your own. \
When they say they've finished the step or ask what's next, that is saying \
so: call next_step.
- awaiting_answer_to in the state says which question the app just asked. \
If it is start_timer and the cook agrees ("yes", "sure", "go ahead"), call \
start_step_timer. If it is next_step and the cook agrees ("yes", "ready", \
"done"), call next_step. If they say no or not yet, reply briefly (e.g. "No \
problem, take your time.") and call nothing.
- For a timer with a different length than the step's, use set_timer.
- "Stop" or "cancel" about a timer means cancel_timer, never end_session. \
Only call end_session if the cook clearly wants to quit the assistant.

During a recipe:
- recipe_text in the state is the recipe as the cook has it now, with \
[current step] marked. Never read a step out of it yourself: steps are only \
read by calling next_step, previous_step or repeat_step, so the app knows \
where the cook is. Answer questions about the dish ("is the oil hot \
enough?", "why rest the dough?", "how thick should the sauce be?") from it \
in one or two sentences, calling no tool, and stay on the same step.
- When the cook asks to change the recipe (leave something out, add \
something, more or less of something, do a step differently), call \
change_recipe; to swap one ingredient for another, call \
substitute_ingredient. "Make it less spicy" means change_recipe with \
action "change_amount" and amount "less" on the hot ingredient. Only change it when they ask for a change, not when \
they ask whether they could.
- The app itself tells the cook to leave out a removed ingredient, or use \
the substitute, whenever a step mentions it, so don't rewrite steps for \
that. Confirm the change in one sentence.
- A change never moves the step: do not call next_step after it.

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
- If get_recipe returns your_usual_changes (a recipe from their book), \
offer them in one sentence ("You usually swap ghee for butter, shall I?") \
and make them only if the cook agrees.
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
LLM_UNAVAILABLE = ("I can't reach the AI model right now. I can still go to the next step, repeat it, "
                   "set timers, and cook recipes from your recipe book.")
SETUP_LATER = "You can finish setup later by typing cooking-agent setup."
NOT_CAUGHT = "Sorry, I didn't catch that. Could you say it another way?"
MAX_TOOL_ROUNDS = 4
QUESTION_FRESH_S = 60.0  # a bare "yes" answers the app's question only this soon after it was heard
MAX_HISTORY_MESSAGES = 40


@dataclass
class AgentConfig:
    wake_word: Optional[str] = None  # None = always listening
    follow_up_window_s: float = 8.0  # with a wake word: no wake word needed this soon after we spoke
    barge_in: bool = True
    barge_in_threshold_dbfs: float = -35.0
    barge_in_ratio: float = 1.5
    echo_tail_ms: int = 250
    instant_commands: bool = True  # handle everyday commands without the LLM (commands.py)
    # Keep a local model loaded (llm.warmup): every this many seconds, and once
    # this long before each timer ends, so the answer after a long simmer isn't
    # a cold load. 0 turns it off.
    keep_warm_every_s: float = 900.0
    keep_warm_before_timer_s: float = 60.0
    keep_warm_poll_s: float = 5.0


def keep_warm_due(now: float, last_ping: float, deadlines: dict[str, float], pinged: set[tuple[str, float]],
                  every_s: float, before_s: float) -> tuple[bool, set[tuple[str, float]]]:
    """Whether to ping the model now, and the timers this ping covers.

    Due when ``every_s`` has passed since the model was last used or pinged,
    or when a running timer ends within ``before_s`` and hasn't had its ping
    yet. Timers are keyed by (name, deadline), so a timer set again under the
    same name gets a new ping.
    """
    keys = {(name, end) for name, end in deadlines.items() if end - now <= before_s and (name, end) not in pinged}
    return (now - last_ping >= every_s or bool(keys)), keys


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
        # True only while the app's own question was the last thing asked, so a
        # bare "yes"/"no" can be answered instantly; any other turn clears it.
        self._question_fresh = False
        self._question_pending = False  # asked this turn; becomes fresh once the whole reply was heard
        self._setup_fresh = False  # same, for the first-run allergies/diet question
        self._stopped = asyncio.Event()
        self._last_llm_use = self.timers.now()  # for keep_warm: a real turn also keeps the model loaded

    # ================================================================ running
    async def run(self, frames: AsyncIterator[bytes],
                  speech_end_fn: Callable[[SttEvent], Optional[float]] | None = None) -> None:
        """Run until the STT stream ends or the user ends the session."""
        announcer = asyncio.create_task(self._announcer(), name="announcer")
        warm = asyncio.create_task(self.keep_warm(), name="keep-warm")
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
            warm.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await announcer
            with contextlib.suppress(asyncio.CancelledError):
                await warm
            await self.timers.aclose()
            with contextlib.suppress(Exception):
                await stt_iter.aclose()
            if self.turn.state != TurnState.STOPPED:
                self.turn.to(TurnState.STOPPED)

    async def keep_warm(self) -> None:
        """Ping the model so Ollama doesn't unload it during a long simmer
        (see ``keep_warm_due``). Failures are logged, never spoken. Returns at
        once if the LLM has no ``warmup`` or the interval is 0."""
        warmup = getattr(self.llm, "warmup", None)
        every = self.config.keep_warm_every_s
        if warmup is None or every <= 0:
            return
        last_ping = self.timers.now()
        pinged: set[tuple[str, float]] = set()
        while True:
            await asyncio.sleep(self.config.keep_warm_poll_s)
            now = self.timers.now()
            deadlines = self.timers.deadlines()
            pinged &= set(deadlines.items())  # forget timers that ended or were cancelled
            due, keys = keep_warm_due(now, max(last_ping, self._last_llm_use), deadlines, pinged, every,
                                      self.config.keep_warm_before_timer_s)
            pinged |= keys
            if not due:
                continue
            last_ping = now
            try:
                await warmup()
                log.debug("keep-warm ping (%s)", ", ".join(n for n, _ in keys) or "interval")
            except Exception as e:  # Ollama stopped, etc.: the next real turn says so
                log.warning("keep-warm ping failed: %s: %s", type(e).__name__, e)

    def begin_setup(self) -> bool:
        """First run: ask about allergies, conditions and diet. Spoken as soon
        as the assistant is idle (needs the announcer running, as in ``run``)."""
        if self.tools.prefs is None:
            return False
        self.session.setup_pending = True
        self._setup_fresh = True
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
        # Commands are matched on the words as spoken, without the note above.
        out.task = asyncio.create_task(self._run_turn(llm_text, out, raw=accepted), name=f"turn-{m.turn_id}")
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
                self._question_pending = False  # the question is no longer the last thing heard
                continue
            new = _Output()
            new.task = asyncio.create_task(self._speak_only(text, new), name="announcement")
            self._output = new

    async def _speak_only(self, text: str, out: _Output) -> None:
        self._question_fresh = False  # an announcement now comes after the app's question
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
    async def _run_turn(self, user_text: str, out: _Output, raw: Optional[str] = None) -> None:
        m = out.metrics
        assert m is not None
        producer = asyncio.create_task(self._think(user_text, out, raw), name=f"llm-{m.turn_id}")
        try:
            await self._play(out)
            await producer  # surface LLM/tool errors
            # The reply, including any question at its end, was heard in full.
            self._question_fresh = self._question_pending
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

    async def _think(self, user_text: str, out: _Output, raw: Optional[str] = None) -> None:
        """Producer for one turn: an instant command if the words are one,
        otherwise stream model rounds and run tool calls; push speakable segments."""
        m = out.metrics
        assert m is not None
        q = out.queue
        raw = user_text if raw is None else raw
        self.tools.utterance = raw
        if self.tools.prefs is not None and (diets := stated_diets(raw)):
            self.tools.prefs.set_stated_diets(diets)  # "I'm vegetarian": saved whatever the model does
        committed: list[Message] = [Message("user", user_text)]
        spoken: list[str] = []

        def emit(seg: str) -> None:
            spoken.append(seg)
            q.put_nowait(seg)

        # A question the app asked counts as "just asked" for this turn only.
        heard_recently = time.perf_counter() - self._last_spoke_at <= QUESTION_FRESH_S
        question_fresh, setup_fresh = self._question_fresh and heard_recently, self._setup_fresh
        self._question_fresh = self._setup_fresh = self._question_pending = False
        match = None
        if self.config.instant_commands:
            match = match_command(raw, self.session, self.tools.store, self.timers.active_names,
                                  self.session.awaiting if question_fresh else None,
                                  setup_open=setup_fresh and self.session.setup_pending)
        try:
            if match is not None:
                committed = [Message("user", raw)]
                await self._instant(match, committed, spoken, emit, m)
            else:
                try:
                    await self._llm_rounds(committed, spoken, emit, m)
                except Exception:
                    log.exception("LLM turn failed")
                    msg = LLM_UNAVAILABLE + (f" {SETUP_LATER}" if self.session.setup_pending else "")
                    emit(msg)
                    self.on_text("assistant", msg)
                    committed = [committed[0], Message("assistant", msg)]  # drop any half-finished round
        except asyncio.CancelledError:
            self._commit_interrupted(committed, spoken)
            raise
        else:
            self._commit(committed)
        finally:
            q.put_nowait(_END)

    async def _instant(self, match, committed: list[Message], spoken: list[str],
                       emit: Callable[[str], None], m: TurnMetrics) -> None:
        """Run an instant command's tools and speak a fixed reply; no LLM.

        Recorded in the history as plain text, valid for every provider; or,
        for an LLM with ``tool_history`` (Ollama), shaped as if the model had
        called the tool itself. A local model copies the history's pattern:
        with "next" -> "Step 1. ..." as plain text, qwen3 began writing
        "next_step" as words instead of calling it."""
        results: list[ToolResult] = []
        calls = [ToolCall(c.name, c.args, id=f"instant-{self._turn_counter}-{i}") for i, c in enumerate(match.calls)]
        for c in calls:
            m.tool_calls.append(c.name)
            results.append(await self.tools.dispatch(c))
        text = self._step_speech(results) or speak_result(results) or match.reply or ""
        if text:
            emit(text)
            self.on_text("assistant", text)
        if calls and getattr(self.llm, "tool_history", False):
            committed.append(Message("assistant", "", calls))
            committed.append(Message("tool", tool_results=results))
        committed.append(Message("assistant", text))
        if match.keep_open:
            self._question_pending = True  # "hold on": the same question is still open
        else:
            self._ask_follow_up(results, committed, spoken, emit)

    async def _llm_rounds(self, committed: list[Message], spoken: list[str], emit: Callable[[str], None],
                          m: TurnMetrics) -> None:
        """Stream model rounds, run tool calls, and append the turn to ``committed``."""
        all_results: list[ToolResult] = []
        retried_empty = False
        for _ in range(MAX_TOOL_ROUNDS):
            m.llm_rounds += 1
            self._last_llm_use = self.timers.now()
            seg = SentenceSegmenter()
            calls = []
            reply: Optional[Message] = None
            async for ev in self.llm.stream(self._system_prompt(), self.history + committed,
                                          self.tools.specs_for(self.session)):
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
            if not calls and not reply.text.strip() and not all_results:
                # Nothing at all: Ollama drops a call to a tool that doesn't exist (qwen3 tried
                # "change_amount"), which left the cook in silence. Ask once more, then say so.
                if not retried_empty:
                    retried_empty = True
                    log.warning("empty reply from the model; asking again")
                    continue
                emit(NOT_CAUGHT)
                self.on_text("assistant", NOT_CAUGHT)
                committed.append(Message("assistant", NOT_CAUGHT))
                break
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
            step_text = (self._step_speech(results) or self._amount_speech(calls, results)
                         or self._dish_choice_speech(results))
            if step_text is not None:
                # Read the step ourselves, word for word, instead of asking
                # the model to: exact wording and one LLM round fewer.
                emit(step_text)
                self.on_text("assistant", step_text)
                committed.append(Message("assistant", step_text))
                break
        else:
            emit("Sorry, I got stuck on that one. Could you say it another way?")
            committed.append(Message("assistant", "(gave up after too many tool calls)"))
            return
        self._ask_follow_up(all_results, committed, spoken, emit)

    @staticmethod
    def _amount_speech(calls: list[ToolCall], results: list[ToolResult]) -> Optional[str]:
        """'Okay, now 1 and 1/2 cups basmati rice.' when every call this round was
        a successful amount change: read by the app, because qwen3 misread the
        new amount ("adjusted the rice to 1 cup" for 1 1/2 cups)."""
        if not results or not all(c.name == "change_recipe" and (c.args or {}).get("action") == "change_amount"
                                  for c in calls):
            return None
        if any("line" not in r.result or r.result.get("warnings") for r in results):
            return None  # a warning must be said: leave it to the model
        return " ".join(f"Okay, now {r.result['line']}." for r in results)

    def _dish_choice_speech(self, results: list[ToolResult]) -> Optional[str]:
        """"Here's Kaju Curry for 2." when a new recipe is built around something
        the cook usually leaves out: the app asks about it next, and the model,
        left to describe the recipe, talked about it wrongly ("I kept the
        cashews as you usually include them"). A warning is left to the model."""
        choice = self.session.dish_choice
        if choice is None or len(results) != 1 or results[0].name != "get_recipe":
            return None
        res = results[0].result
        if "built_around" not in res or res.get("warnings"):
            return None
        tailored = [t.replace(" (", ", ").replace(")", "") for t in res.get("tailored_to", [])]
        made = f" Tailored to you: {'; '.join(tailored)}." if tailored else ""
        return f"Here's {choice['name']} for {res['servings']}.{made}"

    @staticmethod
    def _step_speech(results: list[ToolResult]) -> Optional[str]:
        """'Step 3. <instruction exactly as stored>' when every call this round
        was step navigation that succeeded; None to let the model answer."""
        if not results or not all(r.name in STEP_TOOLS and "instruction" in r.result for r in results):
            return None
        step = results[-1].result  # 'go back two steps' -> read where we ended up
        last = ", the last step" if step.get("is_last_step") else ""
        return f"Step {step['step_number']}{last}. {step['instruction']}"

    def _ask_follow_up(self, results: list[ToolResult], committed: list[Message], spoken: list[str],
                       emit: Callable[[str], None]) -> None:
        """Append the app's own question after a step ("start a timer?" /
        "ready for the next step?"). Done here rather than left to the model
        so the cook is asked every time, in the same words."""
        model_asked = " ".join(spoken).rstrip().endswith("?")
        question = self.tools.follow_up(results)
        if model_asked and question is None and not any(r.name in STEP_TOOLS for r in results):
            # The model asked its own question ("want me to take it out?") and the app had
            # nothing to ask: a "yes" now answers that, not the earlier "ready for the next step?".
            self.session.awaiting = None
            return
        if not question:
            return
        if model_asked and self.session.awaiting is not Awaiting.DISH_CHOICE:
            return  # the model already ended on a question; don't stack a second one
        emit(question)
        self.on_text("assistant", question)
        self._question_pending = True  # once heard in full, a bare "yes"/"no" answers this
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
