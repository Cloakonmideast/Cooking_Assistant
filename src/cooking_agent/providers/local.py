"""Offline fallbacks: Vosk STT and pyttsx3 TTS.

Quality and latency are well below the cloud providers, but the agent keeps
working without STT/TTS keys or network (the LLM still needs Gemini).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

from .base import STT_SAMPLE_RATE, AudioChunk, SttEvent

log = logging.getLogger(__name__)


class VoskSTT:
    """Streaming Kaldi recogniser. ``AcceptWaveform`` returns True when Vosk's
    own endpointer decides the utterance ended, which we surface as ``final``.
    Recognition runs in a worker thread so the event loop never blocks."""

    name = "vosk"

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or os.environ.get("VOSK_MODEL_PATH")
        if not path or not Path(path).is_dir():
            raise RuntimeError(
                "VOSK_MODEL_PATH must point to an unpacked Vosk model directory "
                "(e.g. vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models)"
            )
        import vosk

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(path)

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        import vosk

        rec = vosk.KaldiRecognizer(self._model, STT_SAMPLE_RATE)
        rec.SetWords(True)
        last_partial = ""
        async for frame in audio:
            ended = await asyncio.to_thread(rec.AcceptWaveform, frame)
            if ended:
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text:
                    words = res.get("result") or []
                    yield SttEvent("final", text, audio_end_s=words[-1]["end"] if words else None)
                last_partial = ""
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial and partial != last_partial:
                    if not last_partial:
                        yield SttEvent("speech_started")
                    yield SttEvent("partial", partial)
                    last_partial = partial


class Pyttsx3TTS:
    """Offline TTS via the OS speech engine (SAPI5 / NSSpeechSynthesizer / eSpeak).

    pyttsx3 has no streaming API, so each segment is rendered to a temporary
    WAV in a worker thread and yielded as one chunk. Because segments are
    sentence-sized the pipeline still overlaps: sentence 1 plays while sentence
    2 is being synthesised.
    """

    name = "pyttsx3"

    def __init__(self, rate: int = 170, voice: Optional[str] = None):
        self.rate = rate
        self.voice = voice

    def _render(self, text: str) -> AudioChunk:
        import pyttsx3

        from ..audio import read_wav

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        if self.voice:
            engine.setProperty("voice", self.voice)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "seg.wav"
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            engine.stop()
            return read_wav(out)

    async def stream(self, segments: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        async for seg in segments:
            chunk = await asyncio.to_thread(self._render, seg)
            if chunk.pcm:
                yield chunk
