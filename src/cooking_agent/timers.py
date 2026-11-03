"""Concurrent named cooking timers on the asyncio event loop.

Every timer is one asyncio task sleeping until its deadline. When it fires it
calls ``on_fire(name)``; the agent uses that to queue a spoken announcement,
which the speaker inserts at the next sentence boundary even if the assistant
is in the middle of a reply. All mutation happens on the event loop thread, so
no locks are needed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

FireCallback = Callable[[str], Awaitable[None] | None]


class TimerError(ValueError):
    """Raised for invalid timer operations; the message is safe to speak."""


def normalize_name(name: str) -> str:
    name = " ".join(name.lower().strip().split())
    if name.endswith(" timer"):
        name = name[: -len(" timer")].strip()
    return name


@dataclass
class _Timer:
    name: str
    duration_s: float
    deadline: float  # loop.time() based
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class TimerManager:
    def __init__(self, on_fire: FireCallback, clock: Callable[[], float] | None = None):
        self._on_fire = on_fire
        self._timers: dict[str, _Timer] = {}
        self._finished: list[str] = []  # names of timers that fired, most recent last
        self._clock = clock or time.monotonic

    # -- commands ---------------------------------------------------------
    def set(self, name: str, seconds: float) -> dict:
        key = normalize_name(name)
        if not key:
            raise TimerError("A timer needs a name, like 'rice' or 'pasta'.")
        if seconds <= 0:
            raise TimerError("Timer duration must be positive.")
        if seconds > 24 * 3600:
            raise TimerError("Timers longer than 24 hours aren't supported.")
        replaced = key in self._timers
        if replaced:
            self._cancel(key)
        timer = _Timer(name=key, duration_s=seconds, deadline=self._clock() + seconds)
        timer.task = asyncio.get_running_loop().create_task(
            self._run(timer), name=f"timer:{key}"
        )
        self._timers[key] = timer
        return {"name": key, "seconds": seconds, "replaced_existing": replaced}

    def cancel(self, name: str) -> dict:
        key = normalize_name(name)
        if key not in self._timers:
            active = sorted(self._timers)
            raise TimerError(
                f"There's no running timer called {key}."
                + (f" Running timers: {', '.join(active)}." if active else " No timers are running.")
            )
        self._cancel(key)
        return {"cancelled": key}

    def cancel_all(self) -> list[str]:
        names = list(self._timers)
        for key in names:
            self._cancel(key)
        return names

    def list(self) -> list[dict]:
        now = self._clock()
        return [
            {"name": t.name, "remaining_seconds": max(0, round(t.deadline - now))}
            for t in sorted(self._timers.values(), key=lambda t: t.deadline)
        ]

    def recently_finished(self) -> list[str]:
        return list(self._finished[-5:])

    @property
    def active_names(self) -> list[str]:
        return sorted(self._timers)

    async def aclose(self) -> None:
        tasks = [t.task for t in self._timers.values() if t.task]
        self.cancel_all()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- internals --------------------------------------------------------
    def _cancel(self, key: str) -> None:
        timer = self._timers.pop(key)
        if timer.task and not timer.task.done():
            timer.task.cancel()

    async def _run(self, timer: _Timer) -> None:
        await asyncio.sleep(timer.duration_s)
        # Only remove ourselves if we are still the registered timer for this name
        # (a replacement may have been registered while we were being cancelled).
        if self._timers.get(timer.name) is timer:
            del self._timers[timer.name]
        self._finished.append(timer.name)
        result = self._on_fire(timer.name)
        if asyncio.iscoroutine(result):
            await result
