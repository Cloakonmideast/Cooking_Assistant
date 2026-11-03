"""Protocol tests for the cloud adapters, against local stand-ins (no network).

These check that the adapters speak the documented wire protocol and map
events correctly. They do not prove the real services behave identically.
"""

import json
import types as pytypes

import pytest
from websockets.asyncio.server import serve

from cooking_agent.providers.base import Message, TextDelta, ToolCall, ToolCallEvent, ToolResult, ToolSpec, TurnComplete
from cooking_agent.recipe import Recipe


# ------------------------------------------------------------ Deepgram STT
def results(text, is_final, speech_final=False, end=1.0):
    return json.dumps({"type": "Results", "is_final": is_final, "speech_final": speech_final,
                       "channel": {"alternatives": [{"transcript": text, "words": [{"word": "x", "end": end}] if text else []}]}})


async def test_deepgram_stt_assembles_finals():
    received = {"binary": 0, "control": []}

    async def handler(ws):
        assert ws.request.headers["Authorization"] == "Token test-key"
        assert "endpointing=300" in ws.request.path and "encoding=linear16" in ws.request.path
        await ws.send(json.dumps({"type": "SpeechStarted", "channel": [0], "timestamp": 0.1}))
        await ws.send(results("next", False))
        await ws.send(results("next step", True, end=0.9))
        await ws.send(results("please", True, speech_final=True, end=1.4))
        await ws.send(results("set a timer", True, end=3.0))
        await ws.send(json.dumps({"type": "UtteranceEnd", "channel": [0], "last_word_end": 3.1}))
        async for msg in ws:
            if isinstance(msg, bytes):
                received["binary"] += 1
            else:
                received["control"].append(json.loads(msg)["type"])
                break

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        from cooking_agent.providers.deepgram import DeepgramSTT

        stt = DeepgramSTT(api_key="test-key", base_url=f"ws://127.0.0.1:{port}")

        async def audio():
            for _ in range(3):
                yield b"\x00" * 640

        events = [e async for e in stt.stream(audio())]

    kinds = [(e.kind, e.text) for e in events]
    assert kinds == [("speech_started", ""), ("partial", "next"), ("final", "next step please"),
                     ("final", "set a timer")]
    assert events[2].audio_end_s == 1.4 and events[3].audio_end_s == 3.1
    assert received == {"binary": 3, "control": ["CloseStream"]}


# ------------------------------------------------------------ Deepgram TTS
async def test_deepgram_tts_streams_audio_per_segment_and_reuses_socket():
    connections = []

    async def handler(ws):
        connections.append(ws)
        async for msg in ws:
            m = json.loads(msg)
            if m["type"] == "Speak":
                text = m["text"]
            elif m["type"] == "Flush":
                await ws.send(text.encode()[:4] * 10)  # fake "audio"
                await ws.send(json.dumps({"type": "Flushed", "sequence_id": 0}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        from cooking_agent.providers.deepgram import DeepgramTTS

        tts = DeepgramTTS(api_key="k", base_url=f"ws://127.0.0.1:{port}", sample_rate=24000)

        async def segs(*xs):
            for x in xs:
                yield x

        chunks = [c async for c in tts.stream(segs("Step one.", "Step two."))]
        assert [c.pcm[:4] for c in chunks] == [b"Step", b"Step"]
        assert all(c.sample_rate == 24000 for c in chunks)
        chunks = [c async for c in tts.stream(segs("Again."))]
        assert len(chunks) == 1
        assert len(connections) == 1  # socket reused across turns
        await tts.aclose()


# ------------------------------------------------------------ Gemini
class _FakeModels:
    def __init__(self, chunks, structured):
        self.chunks, self.structured, self.calls = chunks, structured, []

    async def generate_content_stream(self, **kw):
        self.calls.append(kw)

        async def gen():
            for c in self.chunks:
                yield c

        return gen()

    async def generate_content(self, **kw):
        self.calls.append(kw)
        return pytypes.SimpleNamespace(text=self.structured.pop(0))


def _chunk(*parts):
    from google.genai import types as t

    return t.GenerateContentResponse(candidates=[t.Candidate(content=t.Content(role="model", parts=list(parts)))])


@pytest.fixture
def gemini(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import types as t

    from cooking_agent.providers.gemini import GeminiLLM

    llm = GeminiLLM(api_key="test", model="test-model")
    sig_part = t.Part(function_call=t.FunctionCall(name="next_step", args={}, id="c1"), thought_signature=b"sig")
    llm.client = pytypes.SimpleNamespace(aio=pytypes.SimpleNamespace(models=_FakeModels(
        [_chunk(t.Part(text="Sure, ")), _chunk(t.Part(text="hidden", thought=True)), _chunk(sig_part)], [])))
    return llm


async def test_gemini_stream_maps_events_and_preserves_raw(gemini):
    from google.genai import types as t

    events = [e async for e in gemini.stream("sys", [Message("user", "next")],
                                            [ToolSpec("next_step", "d", {"type": "object", "properties": {}})])]
    assert isinstance(events[0], TextDelta) and events[0].text == "Sure, "
    assert isinstance(events[1], ToolCallEvent) and events[1].call == ToolCall("next_step", {}, "c1")
    done = events[-1]
    assert isinstance(done, TurnComplete) and done.message.text == "Sure, "  # thought text not spoken
    # Raw content keeps every part, including the thought signature, for the next round.
    assert done.message.raw.parts[-1].thought_signature == b"sig"

    cfg = gemini.client.aio.models.calls[0]["config"]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.tools[0].function_declarations[0].name == "next_step"

    history = [Message("user", "next"), done.message,
               Message("tool", tool_results=[ToolResult("next_step", {"step_number": 1}, "c1")])]
    contents = gemini._contents(history)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is done.message.raw
    fr = contents[2].parts[0].function_response
    assert (fr.name, fr.id, fr.response) == ("next_step", "c1", {"step_number": 1})
    assert isinstance(contents[0], t.Content)


async def test_gemini_structured_output_validates_and_retries(gemini, recipe):
    models = gemini.client.aio.models
    models.structured = ['{"title": "broken"}', recipe.model_dump_json()]
    out = await gemini.generate_structured("jeera rice", Recipe)
    assert out == recipe
    first_cfg = models.calls[0]["config"]
    assert first_cfg.response_mime_type == "application/json"
    assert first_cfg.response_json_schema["title"] == "Recipe"
    assert "previous answer was invalid" in models.calls[1]["contents"]


async def test_gemini_structured_output_gives_up(gemini):
    gemini.client.aio.models.structured = ["{}", "not json"]
    with pytest.raises(RuntimeError, match="valid Recipe"):
        await gemini.generate_structured("x", Recipe)
