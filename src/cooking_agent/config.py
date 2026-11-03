"""Environment-driven provider selection.

``auto`` picks the cloud provider when its key is set and falls back to the
offline one otherwise, so the agent degrades instead of crashing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .providers.base import LLMProvider, STTProvider, TTSProvider

log = logging.getLogger(__name__)


@dataclass
class Settings:
    stt: str = "auto"  # auto | deepgram | vosk
    tts: str = "auto"  # auto | deepgram | pyttsx3
    llm: str = "gemini"
    gemini_model: Optional[str] = None
    gemini_recipe_model: Optional[str] = None
    gemini_thinking_level: Optional[str] = None
    deepgram_stt_model: str = "nova-3"
    deepgram_tts_model: str = "aura-2-thalia-en"
    deepgram_endpointing_ms: int = 300
    wake_word: Optional[str] = None
    barge_in: bool = True
    metrics_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get

        def flag(name: str, default: bool) -> bool:
            v = e(name)
            return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            stt=e("STT_PROVIDER", "auto").lower(),
            tts=e("TTS_PROVIDER", "auto").lower(),
            llm=e("LLM_PROVIDER", "gemini").lower(),
            gemini_model=e("GEMINI_MODEL") or None,
            gemini_recipe_model=e("GEMINI_RECIPE_MODEL") or None,
            gemini_thinking_level=e("GEMINI_THINKING_LEVEL") or None,
            deepgram_stt_model=e("DEEPGRAM_STT_MODEL", "nova-3"),
            deepgram_tts_model=e("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
            deepgram_endpointing_ms=int(e("DEEPGRAM_ENDPOINTING_MS", "300")),
            wake_word=(e("WAKE_WORD") or None),
            barge_in=flag("BARGE_IN", True),
            metrics_path=e("METRICS_LOG") or None,
        )


def make_llm(s: Settings) -> LLMProvider:
    if s.llm == "gemini":
        from .providers.gemini import GeminiLLM

        return GeminiLLM(model=s.gemini_model, recipe_model=s.gemini_recipe_model,
                         thinking_level=s.gemini_thinking_level)
    raise ValueError(f"unknown LLM_PROVIDER {s.llm!r}")


def make_stt(s: Settings) -> STTProvider:
    choice = s.stt
    if choice == "auto":
        choice = "deepgram" if os.environ.get("DEEPGRAM_API_KEY") else "vosk"
    if choice == "deepgram":
        from .providers.deepgram import DeepgramSTT

        return DeepgramSTT(model=s.deepgram_stt_model, endpointing_ms=s.deepgram_endpointing_ms)
    if choice == "vosk":
        from .providers.local import VoskSTT

        return VoskSTT()
    raise ValueError(f"unknown STT_PROVIDER {s.stt!r}")


def make_tts(s: Settings) -> TTSProvider:
    choice = s.tts
    if choice == "auto":
        choice = "deepgram" if os.environ.get("DEEPGRAM_API_KEY") else "pyttsx3"
    if choice == "deepgram":
        from .providers.deepgram import DeepgramTTS

        return DeepgramTTS(model=s.deepgram_tts_model)
    if choice == "pyttsx3":
        from .providers.local import Pyttsx3TTS

        return Pyttsx3TTS()
    raise ValueError(f"unknown TTS_PROVIDER {s.tts!r}")
