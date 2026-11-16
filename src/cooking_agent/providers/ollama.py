"""Local LLM through Ollama (``ollama`` Python client, server on localhost).

* Streaming: ``AsyncClient.chat(stream=True)`` with tool definitions; the
  agent runs the tools itself, as with Gemini.
* ``think=False``: reasoning models would otherwise think before every reply,
  which is too slow for a voice turn.
* ``num_ctx`` is always sent. The system prompt plus tool schemas are ~3,200
  tokens before any history, and a smaller server default would silently cut
  the start of the prompt.
* ``keep_alive`` keeps the model loaded between turns, so a turn never waits
  for the model to load from disk again.
* Structured output: ``format`` set to the pydantic JSON schema, validated
  with one retry, like the Gemini adapter.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional, Type

from pydantic import ValidationError

from .base import (
    LLMEvent,
    MediaPart,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
    M,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
KEEP_ALIVE = "30m"


def _get(obj: Any, key: str) -> Any:
    """Read a field from a plain dict (tests) or an ``ollama`` response model."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, url: str = DEFAULT_URL, client: Any = None,
                 temperature: float = 0.4, num_ctx: int = 8192, timeout: float = 120.0):
        if client is None:
            import ollama  # lazy import keeps tests and other providers independent

            # Without a timeout a hung server would mean silence forever instead
            # of "I can't reach the AI model right now".
            client = ollama.AsyncClient(host=url, timeout=timeout)
        self.client = client
        self.model = model
        self.url = url
        self.temperature = temperature
        self.num_ctx = num_ctx

    async def warmup(self) -> None:
        """Load the model into memory now, so the first reply doesn't wait for
        it (the first load can take about a minute)."""
        # Same options as chat: Ollama reloads a model whose num_ctx changes.
        await self.client.generate(model=self.model, prompt="", keep_alive=KEEP_ALIVE, options=self._options())

    # ------------------------------------------------------------ conversion
    @staticmethod
    def _messages(system: str, history: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in history:
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                msg: dict = {"role": "assistant", "content": m.text}
                if m.tool_calls:
                    msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m.tool_calls]
                out.append(msg)
            elif m.role == "tool":
                for r in m.tool_results:
                    out.append({"role": "tool", "tool_name": r.name, "content": json.dumps(r.result)})
        return out

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict]:
        return [{"type": "function", "function": {"name": s.name, "description": s.description,
                                                  "parameters": s.parameters}} for s in tools]

    def _options(self) -> dict:
        return {"temperature": self.temperature, "num_ctx": self.num_ctx}

    # ------------------------------------------------------------ streaming
    async def stream(self, system: str, history: list[Message], tools: list[ToolSpec]) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = dict(model=self.model, messages=self._messages(system, history), stream=True,
                                      think=False, keep_alive=KEEP_ALIVE, options=self._options())
        if tools:
            kwargs["tools"] = self._tools(tools)
        text: list[str] = []
        calls: list[ToolCall] = []
        async for part in await self.client.chat(**kwargs):
            msg = _get(part, "message")
            content = _get(msg, "content") or ""
            if content:
                text.append(content)
                yield TextDelta(content)
            for tc in _get(msg, "tool_calls") or []:
                fn = _get(tc, "function")
                args = _get(fn, "arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                call = ToolCall(name=_get(fn, "name"), args=dict(args))
                calls.append(call)
                yield ToolCallEvent(call)
        yield TurnComplete(Message(role="assistant", text="".join(text), tool_calls=calls))

    # ------------------------------------------------------------ structured
    async def generate_structured(self, prompt: str, schema: Type[M],
                                  media: Optional[list[MediaPart]] = None) -> M:
        if media:
            from ..recipe_import import RecipeImportError

            raise RecipeImportError("Reading photos, PDFs and recordings needs Gemini: set GEMINI_API_KEY in .env.")
        messages = [{"role": "user", "content": prompt}]
        last_err: Optional[Exception] = None
        for attempt in range(2):
            resp = await self.client.chat(model=self.model, messages=messages, stream=False, think=False,
                                          format=schema.model_json_schema(), keep_alive=KEEP_ALIVE,
                                          options=self._options())
            raw = _get(_get(resp, "message"), "content") or ""
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last_err = e
                log.warning("structured output failed validation (attempt %d): %s", attempt + 1, e)
                messages = [{"role": "user",
                             "content": f"{prompt}\n\nYour previous answer was invalid: {e}. Return only valid JSON."}]
        raise RuntimeError(f"model did not return a valid {schema.__name__}") from last_err
