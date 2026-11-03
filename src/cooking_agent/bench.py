"""End-to-end latency benchmark over recorded WAV fixtures.

Each fixture is streamed through the full pipeline exactly as a mic would be
(20 ms frames, paced in real time, then trailing silence so the endpointer can
fire): MicGate -> STT -> LLM + tools -> segmenter -> TTS. Playback goes to a
simulated speaker, so the numbers measure the agent and its providers, not the
sound card.

Per fixture and run we record the ``TurnMetrics`` summary (see metrics.py)
plus whether the first tool the LLM called matches ``expect_tool`` in the
manifest. The report gives p50 / p90 / max per metric across all turns.

``--fake`` swaps in deterministic offline providers (no keys, no network). The
numbers are then only the agent's own overhead, which is useful to check the
harness and to catch regressions in the pipeline itself.

Usage::

    cooking-agent bench                      # providers from the environment
    cooking-agent bench --runs 3 --out bench/results.json
    cooking-agent bench --fake --no-realtime # offline smoke test, a few seconds
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from .agent import AgentConfig, CookingAgent
from .audio import SimulatedSink, WavSource
from .metrics import MetricsSink, TurnMetrics, percentile
from .providers.base import FRAME_MS, STTProvider, SttEvent, ToolCall
from .recipe import Recipe

log = logging.getLogger(__name__)

DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "bench" / "fixtures"
METRIC_KEYS = ("stt_endpoint_ms", "llm_first_token_ms", "first_audio_ms", "e2e_first_audio_ms", "turn_total_ms")

# Loaded for every fixture with needs_recipe, positioned on step 2 with a
# running timer, so next/previous/repeat/list_timers all have something to do.
BENCH_RECIPE = Recipe.model_validate({
    "title": "Jeera Rice",
    "servings": 2,
    "ingredients": [
        {"name": "basmati rice", "quantity": 1, "unit": "cup", "note": "rinsed"},
        {"name": "cumin seeds", "quantity": 1, "unit": "tsp"},
        {"name": "ghee", "quantity": 1.5, "unit": "tbsp"},
        {"name": "water", "quantity": 2, "unit": "cups"},
        {"name": "salt", "quantity": None},
    ],
    "steps": [
        {"instruction": "Soak the rice for 20 minutes, then drain.", "timer_minutes": 20},
        {"instruction": "Heat the ghee in a pot and add the cumin seeds."},
        {"instruction": "Add the rice, water and salt, cover and cook for 12 minutes.", "timer_minutes": 12},
        {"instruction": "Rest for 5 minutes, then fluff with a fork.", "timer_minutes": 5},
    ],
})


@dataclass
class Fixture:
    file: Path
    text: str
    needs_recipe: bool
    expect_tool: Optional[str]

    @property
    def name(self) -> str:
        return self.file.stem


def load_manifest(fixtures_dir: Path) -> list[Fixture]:
    data = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    return [
        Fixture(fixtures_dir / f["file"], f["text"], bool(f.get("needs_recipe", True)), f.get("expect_tool"))
        for f in data["fixtures"]
    ]


# ------------------------------------------------------------------ fake providers
class FixtureSTT:
    """Offline STT for ``--fake``: emits the fixture's known transcript once
    ``endpoint_ms`` of audio has passed after the speech in the WAV ends. Driven
    by frame count, so it behaves the same with or without real-time pacing."""

    name = "fixture-stt"

    def __init__(self, text: str, speech_end_s: float, endpoint_ms: int = 300):
        self.text = text
        self.fire_after_frames = int((speech_end_s * 1000 + endpoint_ms) / FRAME_MS)

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[SttEvent]:
        n = 0
        async for _ in audio:
            n += 1
            if n == 1:
                yield SttEvent("speech_started")
            if n == self.fire_after_frames:
                yield SttEvent("final", self.text)


def fake_llm_script() -> dict[str, list]:
    """Keyword -> first-round output for the ScriptedLLM, covering the manifest."""
    return {
        "recipe": ["Sure, let me find one.", ToolCall("get_recipe", {"dish": "jeera rice"})],
        "next": [ToolCall("next_step", {})],
        "repeat": [ToolCall("repeat_step", {})],
        "back": [ToolCall("previous_step", {})],
        "ingredients": [ToolCall("list_ingredients", {})],
        "how long": [ToolCall("list_timers", {})],
        "timer": [ToolCall("set_timer", {"name": "rice", "minutes": 12})],
        "people": [ToolCall("scale_servings", {"servings": 6})],
        "ghee": [ToolCall("substitute_ingredient", {"ingredient": "ghee", "replacement": "butter",
                                                    "quantity": 1.5, "unit": "tbsp"})],
        "done": ["The grains should be tender and the water fully absorbed. "
                 "Tilt the pot; if you see water, give it two more minutes."],
    }


# ------------------------------------------------------------------ one turn
async def run_fixture(fx: Fixture, *, stt: STTProvider, llm, tts, realtime: bool = True,
                      timeout_s: float = 45.0) -> dict:
    """Stream one fixture through a fresh agent and return its turn summary."""
    src = WavSource(fx.file, realtime=realtime, trailing_silence_s=2.0, hold_open=True)
    sink = SimulatedSink(realtime=False)  # measure the agent, not playback
    agent = CookingAgent(stt, llm, tts, sink, AgentConfig(barge_in=False), MetricsSink())
    if fx.needs_recipe:
        agent.session.load_recipe(BENCH_RECIPE.model_copy(deep=True))
        agent.session.next_step()
        agent.session.next_step()  # on step 2: previous/repeat/next all valid
        agent.timers.set("rice", 10 * 60)

    done: asyncio.Future[TurnMetrics] = asyncio.get_running_loop().create_future()

    def on_turn_end(m: TurnMetrics) -> None:
        if not done.done():
            done.set_result(m)
        src.close()
        agent.stop()

    agent.on_turn_end = on_turn_end

    def speech_end(ev: SttEvent) -> Optional[float]:
        # Only meaningful when frames are paced like a real mic.
        return src.t_speech_end if realtime else None

    runner = asyncio.create_task(agent.run(src.frames(), speech_end_fn=speech_end))
    t0 = time.perf_counter()
    error: Optional[str] = None
    try:
        m = await asyncio.wait_for(asyncio.shield(done), timeout_s)
        summary = m.summary()
    except asyncio.TimeoutError:
        error = f"no turn completed within {timeout_s:.0f}s (STT never produced a final?)"
        summary = {k: None for k in METRIC_KEYS} | {"tool_calls": [], "user_text": ""}
    finally:
        src.close()
        agent.stop()
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(runner, 10)
        if not runner.done():
            runner.cancel()
            with contextlib.suppress(BaseException):
                await runner

    first_tool = summary["tool_calls"][0] if summary.get("tool_calls") else None
    return {
        "fixture": fx.name,
        "expected_text": fx.text,
        "expect_tool": fx.expect_tool,
        "first_tool": first_tool,
        "tool_ok": first_tool == fx.expect_tool,
        "error": error,
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        **summary,
    }


# ------------------------------------------------------------------ whole benchmark
def _make_providers(fake: bool, fx: Fixture, speech_end_s: float):
    if fake:
        from .providers.fake import ScriptedLLM, ToneTTS

        return (FixtureSTT(fx.text, speech_end_s), ScriptedLLM(script=fake_llm_script(), recipe=BENCH_RECIPE),
                ToneTTS())
    from .config import Settings, make_llm, make_stt, make_tts

    s = Settings.from_env()
    return make_stt(s), make_llm(s), make_tts(s)


async def run_benchmark(fixtures_dir: Path = DEFAULT_FIXTURES, *, runs: int = 1, fake: bool = False,
                        realtime: bool = True, only: Optional[list[str]] = None,
                        timeout_s: float = 45.0, progress=print) -> dict:
    fixtures = load_manifest(fixtures_dir)
    if only:
        fixtures = [f for f in fixtures if f.name in only]
        if not fixtures:
            raise ValueError(f"no fixtures named {only}")
    results: list[dict] = []
    providers: dict[str, str] = {}
    for run in range(1, runs + 1):
        for fx in fixtures:
            stt, llm, tts = _make_providers(fake, fx, WavSource(fx.file, realtime=False).speech_end_offset)
            try:
                if hasattr(tts, "warmup"):
                    with contextlib.suppress(Exception):
                        await tts.warmup()  # connection setup is not per-turn latency
                r = await run_fixture(fx, stt=stt, llm=llm, tts=tts, realtime=realtime, timeout_s=timeout_s)
            finally:
                if hasattr(tts, "aclose"):
                    with contextlib.suppress(Exception):
                        await tts.aclose()
            r["run"] = run
            providers = r.get("providers") or providers
            results.append(r)
            progress(_line(r))
    return {"providers": providers, "realtime": realtime, "fake": fake, "runs": runs,
            "turns": results, "summary": summarize(results)}


def summarize(results: list[dict]) -> dict:
    out: dict = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        out[key] = ({"n": len(vals), "p50": percentile(vals, 50), "p90": percentile(vals, 90), "max": max(vals)}
                    if vals else {"n": 0})
    scored = [r for r in results if not r.get("error")]
    out["tool_accuracy"] = round(sum(r["tool_ok"] for r in scored) / len(scored), 3) if scored else None
    out["errors"] = sum(1 for r in results if r.get("error"))
    out["filler_first"] = sum(1 for r in results if r.get("first_audio_was_filler"))
    return out


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.0f}"


def _line(r: dict) -> str:
    if r.get("error"):
        return f"  {r['fixture']:<15} ERROR {r['error']}"
    mark = "ok " if r["tool_ok"] else "MISS"
    return (f"  {r['fixture']:<15} e2e {_fmt(r['e2e_first_audio_ms']):>5} ms | "
            f"llm {_fmt(r['llm_first_token_ms']):>5} | audio {_fmt(r['first_audio_ms']):>5} | "
            f"tool {mark} {r['first_tool'] or '-'}  heard {r['user_text']!r}")


def format_report(report: dict) -> str:
    p = report["providers"]
    lines = [
        f"providers: stt={p.get('stt')} llm={p.get('llm')} tts={p.get('tts')}  "
        f"(runs={report['runs']}, realtime={report['realtime']})",
        f"{'metric':<22}{'n':>4}{'p50':>9}{'p90':>9}{'max':>9}",
    ]
    s = report["summary"]
    for key in METRIC_KEYS:
        m = s[key]
        if m["n"]:
            lines.append(f"{key:<22}{m['n']:>4}{m['p50']:>9.0f}{m['p90']:>9.0f}{m['max']:>9.0f}")
        else:
            lines.append(f"{key:<22}{0:>4}{'-':>9}{'-':>9}{'-':>9}")
    acc = s["tool_accuracy"]
    lines.append(f"tool routing accuracy: {'-' if acc is None else f'{acc:.0%}'}   "
                 f"errors: {s['errors']}   turns that started with a filler: {s['filler_first']}")
    return "\n".join(lines)
