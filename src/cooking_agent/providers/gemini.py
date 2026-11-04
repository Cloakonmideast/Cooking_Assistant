"""Gemini adapter (google-genai SDK).

* Streaming: ``client.aio.models.generate_content_stream`` with function
  declarations; automatic function calling is disabled because the agent runs
  the tools itself (they touch session state and timers).
* Structured output: ``response_json_schema`` from the pydantic model, then
  ``model_validate_json``; one retry with the validation error fed back.
* Thought signatures: the model's own ``Content`` (all parts, unchanged) is
  stored in ``Message.raw`` and sent back verbatim on the next round, which is
  what the API requires when thinking models make function calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Optional, Type

from pydantic import ValidationError

from .base import (
    M,
    LLMEvent,
    MediaPart,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 recipe_model: Optional[str] = None, thinking_level: Optional[str] = None,
                 temperature: float = 0.4):
        from google import genai  # lazy import keeps tests SDK-free
        from google.genai import types

        self._types = types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model or DEFAULT_MODEL
        self.recipe_model = recipe_model or self.model
        self.thinking_level = thinking_level
        self.temperature = temperature

    # ------------------------------------------------------------ conversion
    def _contents(self, history: list[Message]) -> list[Any]:
        t = self._types
        out: list[Any] = []
        for m in history:
            if m.role == "user":
                out.append(t.Content(role="user", parts=[t.Part.from_text(text=m.text)]))
            elif m.role == "assistant":
                if m.raw is not None:
                    out.append(m.raw)
                    continue
                parts = []
                if m.text:
                    parts.append(t.Part.from_text(text=m.text))
                for c in m.tool_calls:
                    parts.append(t.Part(function_call=t.FunctionCall(name=c.name, args=c.args, id=c.id)))
                if parts:
                    out.append(t.Content(role="model", parts=parts))
            elif m.role == "tool":
                parts = [
                    t.Part(function_response=t.FunctionResponse(name=r.name, response=r.result, id=r.id))
                    for r in m.tool_results
                ]
                out.append(t.Content(role="user", parts=parts))
        return out

    def _config(self, system: str, tools: list[ToolSpec]) -> Any:
        t = self._types
        kwargs: dict[str, Any] = dict(
            system_instruction=system,
            temperature=self.temperature,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            kwargs["tools"] = [t.Tool(function_declarations=[
                t.FunctionDeclaration(name=s.name, description=s.description, parameters_json_schema=s.parameters)
                for s in tools
            ])]
        if self.thinking_level:
            kwargs["thinking_config"] = t.ThinkingConfig(thinking_level=self.thinking_level.upper())
        return t.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------ streaming
    async def stream(self, system: str, history: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]:
        t = self._types
        parts: list[Any] = []
        text: list[str] = []
        calls: list[ToolCall] = []
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model, contents=self._contents(history), config=self._config(system, tools)
        )
        async for chunk in stream:
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for part in chunk.candidates[0].content.parts or []:
                parts.append(part)
                if part.function_call:
                    fc = part.function_call
                    call = ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id)
                    calls.append(call)
                    yield ToolCallEvent(call)
                elif part.text and not part.thought:
                    text.append(part.text)
                    yield TextDelta(part.text)
        raw = t.Content(role="model", parts=parts) if parts else None
        yield TurnComplete(Message(role="assistant", text="".join(text), tool_calls=calls, raw=raw))

    # ------------------------------------------------------------ structured
    async def generate_structured(self, prompt: str, schema: Type[M],
                                  media: Optional[list[MediaPart]] = None) -> M:
        t = self._types
        config = t.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=self.temperature,
        )
        if self.thinking_level:
            config.thinking_config = t.ThinkingConfig(thinking_level=self.thinking_level.upper())
        parts = [t.Part.from_bytes(data=m.data, mime_type=m.mime_type) for m in media or []]
        contents = prompt
        last_err: Optional[Exception] = None
        for attempt in range(2):
            resp = await self.client.aio.models.generate_content(
                model=self.recipe_model, contents=[*parts, contents] if parts else contents, config=config
            )
            try:
                return schema.model_validate_json(resp.text or "")
            except ValidationError as e:
                last_err = e
                log.warning("structured output failed validation (attempt %d): %s", attempt + 1, e)
                contents = f"{prompt}\n\nYour previous answer was invalid: {e}. Return only valid JSON."
        raise RuntimeError(f"model did not return a valid {schema.__name__}") from last_err
