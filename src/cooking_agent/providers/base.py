"""Provider interfaces.

The agent only talks to these three small protocols, so any STT, LLM or TTS
vendor can be swapped by writing one adapter. Audio is always 16-bit mono PCM.

Message history is kept in a provider-neutral form (``Message``). An adapter
may stash vendor-specific data in ``Message.raw`` (Gemini uses it to keep the
thought signatures it requires on function-call turns); other adapters ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

STT_SAMPLE_RATE = 16_000  # Hz, what the mic pump captures and every STT adapter accepts
FRAME_MS = 20
FRAME_SAMPLES = STT_SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2


# ---------------------------------------------------------------- audio / STT
@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes  # int16 little-endian mono
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


@dataclass(frozen=True)
class SttEvent:
    kind: Literal["speech_started", "partial", "final"]
    text: str = ""
    # For finals: end of the last word, in seconds of audio since the stream
    # started, if the provider reports it. Lets the latency benchmark measure
    # from the true end of speech instead of from the transcript arrival.
    audio_end_s: Optional[float] = None


@runtime_checkable
class STTProvider(Protocol):
    name: str

    def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        """Consume 16 kHz PCM16 frames, yield transcription events.

        A ``final`` event means "the user finished an utterance" (endpointed),
        not merely "this segment's text is stable".
        """
        ...


# ---------------------------------------------------------------- LLM
@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: Optional[str] = None


@dataclass
class ToolResult:
    name: str
    result: dict[str, Any]
    id: Optional[str] = None


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    raw: Any = None  # adapter-private payload (e.g. Gemini Content with thought signatures)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (type: object)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class TurnComplete:
    """Last event of a stream; carries the assistant Message to append to history."""

    message: Message


LLMEvent = TextDelta | ToolCallEvent | TurnComplete


@dataclass(frozen=True)
class MediaPart:
    """A file sent to the LLM alongside the prompt: image, PDF, audio or video."""

    data: bytes
    mime_type: str

M = TypeVar("M", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def stream(
        self, system: str, history: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        """Stream one model round. Must end with exactly one ``TurnComplete``."""
        ...

    async def generate_structured(self, prompt: str, schema: Type[M],
                                  media: Optional[list[MediaPart]] = None) -> M:
        """One-shot generation constrained to ``schema``; returns a validated
        model. ``media`` (images, PDFs, audio) is sent before the prompt."""
        ...


# ---------------------------------------------------------------- TTS
@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def stream(self, segments: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """Consume text segments as they are produced and yield audio as soon as
        it is available. Cancelling the consumer must stop synthesis."""
        ...
