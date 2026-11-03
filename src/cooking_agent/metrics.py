"""Per-turn latency instrumentation.

Timestamps are ``time.perf_counter()`` seconds. Reported durations:

* ``stt_endpoint_ms``  - end of user speech -> final transcript. Only known when
  the caller knows when speech actually ended (the benchmark does, from the
  WAV fixture; live mic runs do not, so it is omitted there).
* ``llm_first_token_ms`` - final transcript -> first LLM event (text or tool call).
* ``first_audio_ms``  - final transcript -> first TTS audio chunk ready to play.
* ``e2e_first_audio_ms`` - end of user speech -> first audio (when speech end is known).

"Final transcript" is the moment the STT endpointer declared the utterance
over; that is the earliest point the agent can act on.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("cooking_agent.metrics")


def now() -> float:
    return time.perf_counter()


@dataclass
class TurnMetrics:
    turn_id: int
    user_text: str
    t_speech_end: Optional[float] = None
    t_transcript: float = field(default_factory=now)
    t_first_llm_event: Optional[float] = None
    t_first_audio: Optional[float] = None
    t_done: Optional[float] = None
    first_audio_was_filler: bool = False
    llm_rounds: int = 0
    tool_calls: list[str] = field(default_factory=list)
    interrupted: bool = False
    providers: dict[str, str] = field(default_factory=dict)

    def mark_llm_event(self) -> None:
        if self.t_first_llm_event is None:
            self.t_first_llm_event = now()

    def mark_audio(self, filler: bool = False) -> None:
        if self.t_first_audio is None:
            self.t_first_audio = now()
            self.first_audio_was_filler = filler

    @staticmethod
    def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else round((b - a) * 1000, 1)

    def summary(self) -> dict:
        return {
            "turn": self.turn_id,
            "user_text": self.user_text,
            "stt_endpoint_ms": self._ms(self.t_speech_end, self.t_transcript),
            "llm_first_token_ms": self._ms(self.t_transcript, self.t_first_llm_event),
            "first_audio_ms": self._ms(self.t_transcript, self.t_first_audio),
            "e2e_first_audio_ms": self._ms(self.t_speech_end, self.t_first_audio),
            "turn_total_ms": self._ms(self.t_transcript, self.t_done),
            "first_audio_was_filler": self.first_audio_was_filler,
            "llm_rounds": self.llm_rounds,
            "tool_calls": self.tool_calls,
            "interrupted": self.interrupted,
            "providers": self.providers,
        }


class MetricsSink:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.turns: list[dict] = []

    def record(self, m: TurnMetrics) -> dict:
        s = m.summary()
        self.turns.append(s)
        log.info(
            "turn %d: llm_first_token=%sms first_audio=%sms%s total=%sms tools=%s%s",
            m.turn_id, s["llm_first_token_ms"], s["first_audio_ms"],
            " (filler)" if s["first_audio_was_filler"] else "",
            s["turn_total_ms"], ",".join(s["tool_calls"]) or "-",
            " INTERRUPTED" if s["interrupted"] else "",
        )
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**asdict(m) | s, "ts": time.time()}, default=str) + "\n")
        return s


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation), p in [0, 100]."""
    if not values:
        return math.nan
    xs = sorted(values)
    k = max(0, math.ceil(p / 100 * len(xs)) - 1)
    return xs[k]
