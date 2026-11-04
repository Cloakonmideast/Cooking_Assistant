"""Deepgram streaming STT (``/v1/listen``) and TTS (``/v1/speak``) over raw WebSockets.

Raw ``websockets`` instead of the Deepgram SDK keeps the protocol visible and
the dependency list short; both endpoints are small JSON + binary protocols.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from .base import STT_SAMPLE_RATE, AudioChunk, SttEvent

log = logging.getLogger(__name__)

BASE_URL = "wss://api.deepgram.com"


def _key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    return key


class DeepgramSTT:
    """Streaming transcription with Deepgram's endpointer.

    Finals are assembled from ``is_final`` segments and emitted on
    ``speech_final`` (pause detected, ``endpointing`` ms) or, as a fallback in
    noisy rooms where ``speech_final`` may not arrive, on ``UtteranceEnd``.
    """

    name = "deepgram"

    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3",
                 endpointing_ms: int = 300, utterance_end_ms: int = 1000, language: str = "en",
                 base_url: str = BASE_URL):
        self.key = _key(api_key)
        self.base_url = base_url
        self.params = {
            "model": model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": STT_SAMPLE_RATE,
            "channels": 1,
            "interim_results": "true",
            "endpointing": endpointing_ms,
            "utterance_end_ms": utterance_end_ms,
            "vad_events": "true",
            "smart_format": "true",
        }

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        from websockets.asyncio.client import connect

        url = f"{self.base_url}/v1/listen?{urlencode(self.params)}"
        async with connect(url, additional_headers={"Authorization": f"Token {self.key}"},
                           max_size=None) as ws:

            async def sender() -> None:
                try:
                    async for frame in audio:
                        await ws.send(frame)
                finally:
                    try:
                        await ws.send(json.dumps({"type": "CloseStream"}))
                    except Exception:  # connection may already be gone
                        pass

            send_task = asyncio.create_task(sender(), name="deepgram-stt-sender")
            pending: list[str] = []
            last_word_end: Optional[float] = None
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == "SpeechStarted":
                        yield SttEvent("speech_started")
                    elif kind == "Results":
                        alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
                        text = (alt.get("transcript") or "").strip()
                        words = alt.get("words") or []
                        if not msg.get("is_final"):
                            if text:
                                yield SttEvent("partial", " ".join(pending + [text]))
                            continue
                        if text:
                            pending.append(text)
                            if words:
                                last_word_end = words[-1].get("end")
                        if msg.get("speech_final") and pending:
                            yield SttEvent("final", " ".join(pending), audio_end_s=last_word_end)
                            pending, last_word_end = [], None
                    elif kind == "UtteranceEnd" and pending:
                        yield SttEvent("final", " ".join(pending),
                                       audio_end_s=msg.get("last_word_end") or last_word_end)
                        pending, last_word_end = [], None
            finally:
                send_task.cancel()
                await asyncio.gather(send_task, return_exceptions=True)


class DeepgramTTS:
    """Streaming Aura TTS. Each text segment is sent as ``Speak`` + ``Flush`` so
    audio for sentence 1 comes back while the LLM is still writing sentence 2.

    One WebSocket is kept open across turns (avoids a TLS + WS handshake per
    reply). If a turn is cancelled mid-stream (barge-in) the socket is dropped
    rather than drained, and the next turn reconnects.
    """

    name = "deepgram"

    def __init__(self, api_key: Optional[str] = None, model: str = "aura-2-thalia-en",
                 sample_rate: int = 24_000, base_url: str = BASE_URL):
        self.key = _key(api_key)
        self.sample_rate = sample_rate
        self.url = f"{base_url}/v1/speak?{urlencode({'model': model, 'encoding': 'linear16', 'sample_rate': sample_rate})}"
        self._ws = None

    async def _conn(self):
        from websockets.asyncio.client import connect
        from websockets.protocol import State

        if self._ws is None or self._ws.state is not State.OPEN:
            self._ws = await connect(self.url, additional_headers={"Authorization": f"Token {self.key}"},
                                     max_size=None)
        return self._ws

    async def warmup(self) -> None:
        await self._conn()

    async def aclose(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def stream(self, segments: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        ws = await self._conn()
        sent = 0
        flushed = 0
        sender_done = asyncio.Event()

        async def sender() -> None:
            nonlocal sent
            try:
                async for seg in segments:
                    await ws.send(json.dumps({"type": "Speak", "text": seg}))
                    await ws.send(json.dumps({"type": "Flush"}))
                    sent += 1
            finally:
                sender_done.set()

        send_task = asyncio.create_task(sender(), name="deepgram-tts-sender")
        recv: Optional[asyncio.Task] = None
        completed = False
        try:
            while True:
                if sender_done.is_set() and flushed >= sent:
                    break
                recv = recv or asyncio.ensure_future(ws.recv())
                waiter = asyncio.ensure_future(sender_done.wait())
                done, _ = await asyncio.wait({recv, waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                if recv not in done:
                    continue
                raw, recv = recv.result(), None
                if isinstance(raw, bytes):
                    if raw:
                        yield AudioChunk(raw, self.sample_rate)
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "Flushed":
                    flushed += 1
                elif msg.get("type") == "Warning":
                    log.warning("deepgram tts: %s", msg.get("description"))
            await send_task  # surface sender errors
            completed = True
        finally:
            if recv is not None:
                recv.cancel()
            if not send_task.done():
                send_task.cancel()
            if not completed:
                # Interrupted: buffered audio on this socket belongs to the old turn.
                ws_, self._ws = self._ws, None
                if ws_ is not None:
                    asyncio.ensure_future(ws_.close())
