"""Deterministic fake providers for tests and offline demos. No network, no audio hardware."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Type

import numpy as np

from .base import (
    M,
    AudioChunk,
    LLMEvent,
    Message,
    SttEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
)


# ------------------------------------------------------------------ STT
class ScriptedSTT:
    """Emits pre-set transcripts. Each utterance is released when the audio
    stream has delivered ``frames_per_utterance`` frames, so timing is driven by
    the audio source like a real STT."""

    name = "fake-stt"

    def __init__(self, utterances: list[str], frames_per_utterance: int = 5):
        self.utterances = list(utterances)
        self.frames_per_utterance = frames_per_utterance

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        count = 0
        queue = list(self.utterances)
        async for _ in audio:
            if not queue:
                continue
            count += 1
            if count == 1:
                yield SttEvent("speech_started")
            if count >= self.frames_per_utterance:
                yield SttEvent("final", queue.pop(0))
                count = 0


class QueueSTT:
    """STT driven directly by the test: ``await stt.say("next step")``.
    Ignores audio content but consumes the audio iterator so the mic gate runs."""

    name = "queue-stt"

    def __init__(self) -> None:
        self._q: asyncio.Queue[SttEvent] = asyncio.Queue()

    async def say(self, text: str) -> None:
        await self._q.put(SttEvent("speech_started"))
        await self._q.put(SttEvent("final", text))

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        async def drain() -> None:
            async for _ in audio:
                pass

        drainer = asyncio.create_task(drain())
        try:
            while True:
                yield await self._q.get()
        finally:
            drainer.cancel()


# ------------------------------------------------------------------ LLM
Rule = Callable[[str, list[Message]], Optional[list[LLMEvent | str]]]


@dataclass
class ScriptedLLM:
    """A tiny rule-based stand-in for the LLM.

    ``script`` maps a user-text keyword to the first round's output: a list of
    strings (streamed as text, word by word) and ``ToolCall`` objects. After a
    tool round, ``after_tool`` produces the spoken follow-up from the tool
    results. ``delay_s`` simulates per-token latency.
    """

    name: str = "fake-llm"
    script: dict[str, list] = field(default_factory=dict)
    recipe: Optional[object] = None
    delay_s: float = 0.0
    first_token_delay_s: float = 0.0
    calls: list[list[Message]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)  # generate_structured prompts, for tests

    async def stream(self, system: str, history: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]:
        self.calls.append(list(history))
        if self.first_token_delay_s:
            await asyncio.sleep(self.first_token_delay_s)
        last = history[-1]
        items: list = []
        if last.role == "user":
            text = last.text.lower()
            for key, out in self.script.items():
                if key in text:
                    items = list(out)
                    break
            else:
                items = ["Sorry, I didn't catch that."]
        elif last.role == "tool":
            items = [self._describe(r.name, r.result) for r in last.tool_results]
        text_out: list[str] = []
        calls: list[ToolCall] = []
        for item in items:
            if isinstance(item, ToolCall):
                calls.append(item)
                yield ToolCallEvent(item)
                continue
            for word in item.split(" "):
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                piece = word + " "
                text_out.append(piece)
                yield TextDelta(piece)
        yield TurnComplete(Message("assistant", "".join(text_out).strip(), calls))

    @staticmethod
    def _describe(name: str, result: dict) -> str:
        if "error" in result:
            return f"{result['error']}"
        if "instruction" in result:
            return f"Step {result['step_number']}. {result['instruction']}"
        if name == "set_timer":
            return f"Okay, {result['name']} timer set."
        if name == "cancel_timer":
            return f"Cancelled the {result['cancelled']} timer."
        if name == "get_recipe":
            where = " from your recipe book" if result.get("source") == "your recipe book" else ""
            return f"I found {result['title']}{where}. It has {result['step_count']} steps."
        if name == "start_step_timer":
            return f"Okay, {result['name']} timer started for {result['duration']}."
        if name == "save_dictated_recipe":
            return f"Saved {result['saved']} to your recipe book."
        if name == "start_recipe_dictation":
            return f"Okay, go ahead with {result['started']}."
        if name == "add_to_dictated_recipe":
            return "Got it."
        if name == "end_session":
            return "Goodbye, happy cooking!"
        return "Done."

    async def generate_structured(self, prompt: str, schema: Type[M]) -> M:
        self.prompts.append(prompt)
        if self.recipe is None:
            raise RuntimeError("ScriptedLLM has no recipe configured")
        if isinstance(self.recipe, schema):
            return self.recipe.model_copy(deep=True)
        return schema.model_validate(self.recipe)


# ------------------------------------------------------------------ TTS
class ToneTTS:
    """'Synthesises' a quiet tone whose length is proportional to the text, in
    ``chunk_ms`` pieces, so playback/barge-in timing behaves like real speech."""

    name = "fake-tts"

    def __init__(self, sample_rate: int = 16_000, ms_per_char: float = 10.0, chunk_ms: int = 100,
                 amplitude: float = 0.05, delay_s: float = 0.0):
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.chunk_ms = chunk_ms
        self.amplitude = amplitude
        self.delay_s = delay_s
        self.spoken: list[str] = []

    async def stream(self, segments: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        async for seg in segments:
            self.spoken.append(seg)
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            total = int(len(seg) * self.ms_per_char * self.sample_rate / 1000)
            t = np.arange(total) / self.sample_rate
            pcm = (np.sin(2 * np.pi * 220 * t) * self.amplitude * 32767).astype(np.int16).tobytes()
            step = self.sample_rate * self.chunk_ms // 1000 * 2
            for i in range(0, len(pcm), step):
                yield AudioChunk(pcm[i : i + step], self.sample_rate)
                await asyncio.sleep(0)
