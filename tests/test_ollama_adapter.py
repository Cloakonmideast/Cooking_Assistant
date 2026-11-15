"""OllamaLLM against a fake ``ollama.AsyncClient`` (no server, no network).

These pin the message shapes the adapter sends; Task 2 of the local-model plan
checks the same shapes against a real Ollama server.
"""

import json

import pytest

from cooking_agent.config import Settings, make_import_llm, make_llm
from cooking_agent.media_import import make_transcriber
from cooking_agent.providers.base import (
    MediaPart, Message, TextDelta, ToolCall, ToolCallEvent, ToolResult, ToolSpec, TurnComplete,
)
from cooking_agent.providers.ollama import OllamaLLM
from cooking_agent.recipe import Recipe
from cooking_agent.recipe_import import RecipeImportError


class FakeOllama:
    """Records every ``chat()`` call; streams ``chunks`` or returns ``replies`` in order."""

    def __init__(self, chunks=None, replies=None):
        self.chunks = chunks or []
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def gen():
                for c in self.chunks:
                    yield c
            return gen()
        return {"message": {"role": "assistant", "content": self.replies.pop(0)}}


def chunk(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": False}


SPEC = ToolSpec("set_timer", "Start a timer.", {"type": "object", "properties": {"minutes": {"type": "number"}}})


async def drain(llm, history, tools=()):
    return [ev async for ev in llm.stream("SYSTEM", list(history), list(tools))]


async def test_history_conversion():
    fake = FakeOllama(chunks=[chunk("ok")])
    llm = OllamaLLM(model="m", client=fake)
    history = [
        Message("user", "next please"),
        Message("assistant", "Here you go.", tool_calls=[ToolCall("next_step", {})]),
        Message("tool", tool_results=[ToolResult("next_step", {"step_number": 1})]),
        Message("assistant", "Step 1. Soak the rice."),
    ]
    await drain(llm, history)
    msgs = fake.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert msgs[1] == {"role": "user", "content": "next please"}
    assert msgs[2] == {"role": "assistant", "content": "Here you go.",
                       "tool_calls": [{"function": {"name": "next_step", "arguments": {}}}]}
    assert msgs[3] == {"role": "tool", "tool_name": "next_step", "content": json.dumps({"step_number": 1})}
    assert msgs[4] == {"role": "assistant", "content": "Step 1. Soak the rice."}


async def test_stream_text_then_tool_call():
    fake = FakeOllama(chunks=[
        chunk("Sure, "),
        chunk("one sec."),
        chunk(tool_calls=[{"function": {"name": "set_timer", "arguments": {"name": "rice", "minutes": 12}}}]),
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ])
    events = await drain(OllamaLLM(model="m", client=fake), [Message("user", "rice timer 12 minutes")], [SPEC])
    assert events[0] == TextDelta("Sure, ")
    assert events[1] == TextDelta("one sec.")
    assert events[2] == ToolCallEvent(ToolCall("set_timer", {"name": "rice", "minutes": 12}))
    assert isinstance(events[3], TurnComplete) and len(events) == 4
    final = events[3].message
    assert final.text == "Sure, one sec."
    assert final.tool_calls == [ToolCall("set_timer", {"name": "rice", "minutes": 12})]


async def test_chat_called_with_options():
    fake = FakeOllama(chunks=[chunk("hi")])
    await drain(OllamaLLM(model="qwen3:8b", client=fake), [Message("user", "hi")], [SPEC])
    kw = fake.calls[0]
    assert kw["model"] == "qwen3:8b"
    assert kw["stream"] is True and kw["think"] is False and kw["keep_alive"] == "30m"
    assert kw["options"]["num_ctx"] == 8192 and kw["options"]["temperature"] == 0.4
    assert kw["tools"] == [{"type": "function", "function": {
        "name": "set_timer", "description": "Start a timer.", "parameters": SPEC.parameters}}]


async def test_no_tools_sends_no_tools_key():
    fake = FakeOllama(chunks=[chunk("hi")])
    await drain(OllamaLLM(model="m", client=fake), [Message("user", "hi")], [])
    assert "tools" not in fake.calls[0]


async def test_generate_structured_retries_once(recipe):
    fake = FakeOllama(replies=["not json", recipe.model_dump_json()])
    out = await OllamaLLM(model="m", client=fake).generate_structured("write it", Recipe)
    assert out == recipe
    assert len(fake.calls) == 2
    assert fake.calls[0]["format"] == Recipe.model_json_schema()
    assert fake.calls[0]["options"]["num_ctx"] == 8192
    assert fake.calls[0]["stream"] is False
    assert "invalid" in fake.calls[1]["messages"][-1]["content"]


async def test_generate_structured_gives_up():
    fake = FakeOllama(replies=["nope", "still nope"])
    with pytest.raises(RuntimeError, match="valid Recipe"):
        await OllamaLLM(model="m", client=fake).generate_structured("write it", Recipe)


async def test_generate_structured_media_rejected():
    llm = OllamaLLM(model="m", client=FakeOllama())
    with pytest.raises(RecipeImportError, match="GEMINI_API_KEY"):
        await llm.generate_structured("read this photo", Recipe, media=[MediaPart(b"x", "image/png")])


def test_make_llm_ollama():
    llm = make_llm(Settings(llm="ollama", ollama_model="llama3.1:8b"))
    assert isinstance(llm, OllamaLLM) and llm.model == "llama3.1:8b"


def test_settings_default_is_ollama(monkeypatch):
    for k in ("LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.llm == "ollama" and s.ollama_model == "qwen3:8b" and s.ollama_url == "http://localhost:11434"


def test_make_import_llm_prefers_gemini_when_key_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert make_import_llm(Settings(llm="ollama")).name == "gemini"


def test_make_import_llm_ollama_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert make_import_llm(Settings(llm="ollama")).name == "ollama"


def test_make_transcriber_none_for_ollama(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert make_transcriber(OllamaLLM(model="m", client=FakeOllama())) is None
