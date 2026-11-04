"""Audio plumbing: PCM helpers, mic/WAV sources, playback sinks and the mic gate.

Everything here exchanges 16-bit little-endian mono PCM. ``sounddevice`` is
imported lazily so tests and the benchmark run on machines without PortAudio.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import threading
import time
import wave
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Protocol

import numpy as np

from .providers.base import FRAME_BYTES, FRAME_MS, FRAME_SAMPLES, STT_SAMPLE_RATE, AudioChunk

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers
def pcm_rms(pcm: bytes) -> float:
    """RMS of int16 PCM, normalised to 0..1."""
    if not pcm:
        return 0.0
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def dbfs(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 0 else -120.0


def silence(n_bytes: int) -> bytes:
    return b"\x00" * n_bytes


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler; adequate for speech, not for music."""
    if src_rate == dst_rate or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(round(len(x) * dst_rate / src_rate))
    if n_out <= 0:
        return b""
    t_out = np.linspace(0, len(x) - 1, n_out)
    y = np.interp(t_out, np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def read_wav(path: str | Path) -> AudioChunk:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, ch = w.getframerate(), w.getnchannels()
        data = w.readframes(w.getnframes())
    if ch != 1:
        arr = np.frombuffer(data, dtype=np.int16).reshape(-1, ch).mean(axis=1)
        data = arr.astype(np.int16).tobytes()
    return AudioChunk(data, rate)


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def speech_bounds(pcm: bytes, rate: int, threshold_dbfs: float = -40.0, frame_ms: int = 20) -> tuple[float, float]:
    """(start_s, end_s) of the region above ``threshold_dbfs``. Used to find
    where speech ends inside a WAV fixture."""
    step = rate * frame_ms // 1000 * 2
    active = [i for i in range(0, len(pcm), step) if dbfs(pcm_rms(pcm[i : i + step])) > threshold_dbfs]
    if not active:
        return 0.0, 0.0
    return active[0] / 2 / rate, (active[-1] + step) / 2 / rate


def frames_of(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> list[bytes]:
    out = [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    if out and len(out[-1]) < frame_bytes:
        out[-1] = out[-1] + silence(frame_bytes - len(out[-1]))
    return out


# ------------------------------------------------------------------ sources
class WavSource:
    """Feeds a WAV file as 20 ms frames, optionally paced in real time, followed
    by trailing silence so the STT endpointer can fire. Records the wall-clock
    moment the speech in the file ends (``t_speech_end``), which is what the
    latency benchmark measures from."""

    def __init__(self, path: str | Path, realtime: bool = True, trailing_silence_s: float = 2.0,
                 hold_open: bool = False):
        chunk = read_wav(path)
        self.pcm = resample(chunk.pcm, chunk.sample_rate, STT_SAMPLE_RATE)
        _, self.speech_end_offset = speech_bounds(self.pcm, STT_SAMPLE_RATE)
        self.realtime = realtime
        self.trailing_silence_s = trailing_silence_s
        self.hold_open = hold_open  # keep yielding silence until .close() (lets the agent finish its turn)
        self.t_start: Optional[float] = None
        self._closed = asyncio.Event()

    @property
    def t_speech_end(self) -> Optional[float]:
        return None if self.t_start is None else self.t_start + self.speech_end_offset

    def close(self) -> None:
        self._closed.set()

    async def frames(self) -> AsyncIterator[bytes]:
        frames = frames_of(self.pcm) + [silence(FRAME_BYTES)] * int(self.trailing_silence_s * 1000 / FRAME_MS)
        self.t_start = time.perf_counter()
        for i, f in enumerate(frames):
            if self._closed.is_set():
                return
            if self.realtime:
                delay = self.t_start + i * FRAME_MS / 1000 - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield f
        while self.hold_open and not self._closed.is_set():
            await asyncio.sleep(FRAME_MS / 1000)
            yield silence(FRAME_BYTES)


class MicSource:
    """Live microphone via sounddevice. The PortAudio callback runs on its own
    thread; frames are handed to the event loop with call_soon_threadsafe."""

    def __init__(self, device: Optional[int | str] = None):
        self.device = device

    async def frames(self) -> AsyncIterator[bytes]:
        import sounddevice as sd  # lazy: needs PortAudio

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)

        def put(data: bytes) -> None:
            if q.full():  # consumer stalled; drop oldest rather than grow unbounded
                q.get_nowait()
            q.put_nowait(data)

        def callback(indata, frames, t, status):  # noqa: ARG001 - PortAudio thread
            if status:
                log.debug("mic status: %s", status)
            loop.call_soon_threadsafe(put, bytes(indata))

        with sd.RawInputStream(samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16",
                               blocksize=FRAME_SAMPLES, device=self.device, callback=callback):
            while True:
                yield await q.get()


# ------------------------------------------------------------------ sinks
class AudioSink(Protocol):
    @property
    def is_playing(self) -> bool: ...
    @property
    def last_played_at(self) -> float: ...
    @property
    def reference_rms(self) -> float: ...
    async def play(self, chunk: AudioChunk) -> None: ...
    async def drain(self) -> None: ...
    def stop(self) -> None: ...


class SimulatedSink:
    """Pretends to play audio in real time (or instantly) without a device.
    Used by tests and the benchmark; records everything it was asked to play."""

    def __init__(self, realtime: bool = False, speed: float = 1.0, max_buffer_s: float = 0.5):
        self.realtime = realtime
        self.speed = speed
        self.max_buffer_s = max_buffer_s  # backpressure, like SoundDeviceSink
        self.chunks: list[AudioChunk] = []
        self.stopped = 0
        self._play_until = 0.0
        self._last = 0.0
        self._ref = 0.0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def is_playing(self) -> bool:
        return time.perf_counter() < self._play_until

    @property
    def last_played_at(self) -> float:
        return min(self._play_until, time.perf_counter()) if self._play_until else self._last

    @property
    def reference_rms(self) -> float:
        return self._ref if self.is_playing else 0.0

    async def play(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)
        self._ref = pcm_rms(chunk.pcm)
        if self.realtime:
            start = max(time.perf_counter(), self._play_until)
            self._play_until = start + chunk.duration_s / self.speed
            while self._play_until - time.perf_counter() > self.max_buffer_s / self.speed:
                await asyncio.sleep(0.01)
        self._last = time.perf_counter()

    async def drain(self) -> None:
        while self.is_playing:
            await asyncio.sleep(min(0.02, self._play_until - time.perf_counter() + 0.001))

    def stop(self) -> None:
        self.stopped += 1
        self._play_until = 0.0
        self._last = time.perf_counter()


class SoundDeviceSink:
    """Speaker output. Chunks are appended to a buffer that the PortAudio
    callback drains; ``stop()`` empties it immediately (barge-in)."""

    def __init__(self, device: Optional[int | str] = None, max_buffer_s: float = 2.0):
        self.device = device
        self.max_buffer_s = max_buffer_s
        self._buf = bytearray()
        self._lock = threading.Lock()  # shared with the PortAudio callback thread
        self._rate: Optional[int] = None
        self._stream = None
        self._last_played = 0.0
        self._ref = 0.0

    def _ensure_stream(self, rate: int) -> None:
        import sounddevice as sd

        if self._stream is not None and self._rate == rate:
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._rate = rate

        def callback(outdata, frames, t, status):  # noqa: ARG001 - PortAudio thread
            n = frames * 2
            with self._lock:
                chunk = bytes(self._buf[:n])
                del self._buf[:n]
            if chunk:
                self._last_played = time.perf_counter()
                self._ref = pcm_rms(chunk)
            outdata[: len(chunk)] = chunk
            outdata[len(chunk):] = silence(n - len(chunk))

        self._stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16",
                                          device=self.device, callback=callback, blocksize=0)
        self._stream.start()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    @property
    def last_played_at(self) -> float:
        return self._last_played

    @property
    def reference_rms(self) -> float:
        return self._ref if self.is_playing else 0.0

    async def play(self, chunk: AudioChunk) -> None:
        self._ensure_stream(chunk.sample_rate)
        with self._lock:
            self._buf.extend(chunk.pcm)
        # Backpressure: don't let synthesis run far ahead of playback, so stop() is cheap.
        while True:
            with self._lock:
                buffered_s = len(self._buf) / 2 / (self._rate or 1)
            if buffered_s <= self.max_buffer_s:
                return
            await asyncio.sleep(0.05)

    async def drain(self) -> None:
        while self.is_playing:
            await asyncio.sleep(0.02)

    def stop(self) -> None:
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ------------------------------------------------------------------ mic gate
class MicGate:
    """Half-duplex gating with energy-based barge-in.

    While the sink is playing (and for ``tail_ms`` afterwards, to cover room
    echo and output latency) mic frames are replaced by silence before they
    reach the STT, so the assistant cannot transcribe itself. Silence (rather
    than dropping frames) keeps the STT stream's clock continuous.

    Barge-in: during playback each frame's level is compared with both an
    absolute floor and the level of the audio being played. If the mic stays
    louder than ``ratio`` x playback level for ``min_ms``, we assume the user
    is talking over the assistant: ``on_barge_in`` is called and the last
    ``preroll_ms`` of held-back mic audio is released to the STT so the first
    words of the interruption are not lost.

    This is not acoustic echo cancellation. See the README for limits.
    """

    def __init__(self, sink: AudioSink, on_barge_in: Callable[[], None], *, enabled: bool = True,
                 threshold_dbfs: float = -35.0, ratio: float = 1.5, min_ms: int = 200,
                 preroll_ms: int = 400, tail_ms: int = 250):
        self.sink = sink
        self.on_barge_in = on_barge_in
        self.enabled = enabled
        self.threshold_rms = 10 ** (threshold_dbfs / 20)
        self.ratio = ratio
        self.min_ms = min_ms
        self.tail_s = tail_ms / 1000
        self._preroll: collections.deque[bytes] = collections.deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._hot_ms = 0
        self.barge_ins = 0
        self.gated_frames = 0

    def _gating(self) -> bool:
        return self.sink.is_playing or (time.perf_counter() - self.sink.last_played_at) < self.tail_s

    def filter(self, frame: bytes) -> bytes:
        if not self._gating():
            self._hot_ms = 0
            self._preroll.clear()
            return frame
        self._preroll.append(frame)
        if self.enabled and self.sink.is_playing:
            rms = pcm_rms(frame)
            if rms > self.threshold_rms and rms > self.ratio * self.sink.reference_rms:
                self._hot_ms += FRAME_MS
            else:
                self._hot_ms = max(0, self._hot_ms - FRAME_MS)
            if self._hot_ms >= self.min_ms:
                self.barge_ins += 1
                self._hot_ms = 0
                released = b"".join(self._preroll)
                self._preroll.clear()
                self.on_barge_in()
                return released
        self.gated_frames += 1
        return silence(len(frame))

    async def process(self, frames: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for f in frames:
            yield self.filter(f)
