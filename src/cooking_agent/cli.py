"""Command line entry point: ``cooking-agent <command>``.

    run       live voice agent (mic + speakers)
    chat      type instead of talking; real LLM and tools, replies printed (and spoken with --speak)
    demo      fully offline scripted session with fake providers (no keys, no audio hardware)
    recipes   manage the recipe book: add a recipe from a web page or file, list, show, remove
    prefs     show or edit your saved preferences (diet, allergies, dislikes, pantry) and learned habits
    bench     end-to-end latency benchmark over recorded WAV fixtures
    diagnose  check audio devices, mic level, provider keys and connectivity
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time

from dotenv import load_dotenv

from .agent import AgentConfig, CookingAgent
from .audio import MicSource, SimulatedSink, SoundDeviceSink, dbfs, pcm_rms
from .config import Settings, make_llm, make_stt, make_tts, open_preferences, open_recipe_store
from .metrics import MetricsSink


def _print_text(who: str, text: str) -> None:
    prefix = "you" if who == "user" else "assistant"
    print(f"[{prefix}] {text}", flush=True)


# ------------------------------------------------------------------ run
async def _run(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    if args.wake_word is not None:
        s.wake_word = args.wake_word or None
    if args.no_barge_in:
        s.barge_in = False
    stt, llm, tts = make_stt(s), make_llm(s), make_tts(s)
    sink = SoundDeviceSink(device=args.output_device)
    store = open_recipe_store(s)
    agent = CookingAgent(
        stt, llm, tts, sink,
        AgentConfig(wake_word=s.wake_word, barge_in=s.barge_in),
        MetricsSink(s.metrics_path), on_text=_print_text, recipes=store, preferences=open_preferences(s),
    )
    if hasattr(tts, "warmup"):
        with contextlib.suppress(Exception):
            await tts.warmup()
    print(f"providers: stt={stt.name} llm={llm.name} tts={tts.name}; "
          f"wake word: {s.wake_word or '(none, always listening)'}; barge-in: {s.barge_in}; "
          f"recipe book: {len(store)} recipes")
    print("Speak now. Say something like 'find me a recipe for fried rice'. Ctrl+C to quit.")
    try:
        await agent.run(MicSource(device=args.input_device).frames())
    finally:
        sink.close()
        if hasattr(tts, "aclose"):
            await tts.aclose()
    return 0


# ------------------------------------------------------------------ chat
async def _chat(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    llm = make_llm(s)
    if args.speak:
        tts, sink = make_tts(s), SoundDeviceSink()
    else:
        from .providers.fake import ToneTTS

        tts, sink = ToneTTS(ms_per_char=0), SimulatedSink()
    from .providers.fake import QueueSTT

    agent = CookingAgent(QueueSTT(), llm, tts, sink, AgentConfig(barge_in=False),
                         MetricsSink(s.metrics_path), on_text=_print_text, recipes=open_recipe_store(s),
                         preferences=open_preferences(s))
    loop = asyncio.get_running_loop()
    timers_task = asyncio.create_task(agent._announcer())  # timer announcements in text mode
    print(f"text mode (llm={llm.name}). Type a message, empty line or Ctrl+D to quit.")
    try:
        while not agent.stopped:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line.strip():
                break
            await agent.submit_text(line.strip())
            last = agent.metrics.turns[-1]
            print(f"   (llm first token {last['llm_first_token_ms']} ms, "
                  f"tools: {', '.join(last['tool_calls']) or 'none'})")
    finally:
        timers_task.cancel()
        await agent.timers.aclose()
    return 0


# ------------------------------------------------------------------ demo
async def _demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    await run_demo()
    return 0


# ------------------------------------------------------------------ recipes
async def _recipes(args: argparse.Namespace) -> int:
    from .recipe_import import RecipeImportError, recipe_from_source

    store = open_recipe_store(Settings.from_env())
    if args.action == "add":
        try:
            recipe = await asyncio.to_thread(recipe_from_source, args.source)
        except (RecipeImportError, ValueError) as e:  # ValueError covers pydantic validation errors
            print(f"could not import: {e}")
            return 1
        if args.servings:
            recipe = recipe.model_copy(update={"servings": args.servings})
        if args.title:
            recipe = recipe.model_copy(update={"title": args.title})
        rid = store.add(recipe, source=args.source)
        _print_recipe(rid, recipe)
        print(f"saved as #{rid}. Check the servings above: quantities scale from it.")
    elif args.action == "list":
        rows = store.search(args.query) if args.query else store.all()
        if not rows:
            print("recipe book is empty" if not args.query else "no matches")
        for r in rows:
            print(f"#{r.id:<4} {r.title}  (serves {r.recipe.servings}; {r.source or 'unknown source'})")
    elif args.action == "show":
        r = store.get(args.id)
        if r is None:
            print(f"no recipe #{args.id}")
            return 1
        _print_recipe(r.id, r.recipe)
    elif args.action == "remove":
        if not store.remove(args.id):
            print(f"no recipe #{args.id}")
            return 1
        print(f"removed #{args.id}")
    return 0


def _print_recipe(rid: int, recipe) -> None:
    from .recipe import format_minutes

    print(f"#{rid} {recipe.title} (serves {recipe.servings})")
    print("ingredients:")
    for ing in recipe.ingredients:
        print(f"  - {ing.describe()}")
    print("steps:")
    for i, step in enumerate(recipe.steps, 1):
        timer = f"  [timer: {format_minutes(step.timer_minutes)}]" if step.timer_minutes else ""
        print(f"  {i}. {step.instruction}{timer}")


# ------------------------------------------------------------------ prefs
async def _prefs(args: argparse.Namespace) -> int:
    prefs = open_preferences(Settings.from_env())
    if args.action == "add":
        new = prefs.remember(args.category, " ".join(args.value))
        print("saved" if new else "already known")
    elif args.action == "remove":
        removed = prefs.forget(" ".join(args.value))
        print("removed: " + ", ".join(r["value"] for r in removed) if removed else "no matching preference")
        return 0 if removed else 1
    elif args.action == "forget-history":
        prefs.forget_history()
        print("forgot learned habits (recipe requests and ingredient counts); stated preferences kept")
    stated = prefs.stated()
    learned = prefs.learned()
    print("stated preferences:" if stated else "stated preferences: none")
    for r in stated:
        print(f"  {r['category']:<8} {r['value']}")
    print("learned from history:" if learned else "learned from history: nothing yet")
    for k, v in learned.items():
        print(f"  {k.replace('_', ' ')}: {', '.join(v) if isinstance(v, list) else v}")
    return 0


# ------------------------------------------------------------------ bench
async def _bench(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .bench import DEFAULT_FIXTURES, format_report, run_benchmark

    report = await run_benchmark(Path(args.fixtures) if args.fixtures else DEFAULT_FIXTURES,
                                 runs=args.runs, fake=args.fake, realtime=not args.no_realtime,
                                 only=args.only or None, timeout_s=args.timeout)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if report["summary"]["errors"] else 0


# ------------------------------------------------------------------ diagnose
async def _diagnose(args: argparse.Namespace) -> int:
    ok = True

    def report(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {name}: {detail}")

    for key in ("GEMINI_API_KEY", "DEEPGRAM_API_KEY", "VOSK_MODEL_PATH"):
        print(f"[info] {key}: {'set' if os.environ.get(key) else 'not set'}")

    try:
        import sounddevice as sd

        print(sd.query_devices())
        report("audio", True, f"default devices {sd.default.device}")
    except Exception as e:  # PortAudio missing, no devices, ...
        report("audio", False, f"{type(e).__name__}: {e}")
        sd = None

    if sd is not None and args.mic_seconds > 0:
        print(f"Recording {args.mic_seconds}s from the mic; please speak...")
        levels = []
        with contextlib.suppress(Exception):
            async for frame in _take(MicSource(device=args.input_device).frames(), args.mic_seconds):
                levels.append(dbfs(pcm_rms(frame)))
        if levels:
            peak = max(levels)
            report("mic level", peak > -45, f"peak {peak:.1f} dBFS, median {sorted(levels)[len(levels)//2]:.1f} dBFS "
                   "(speech should peak above about -35)")
        else:
            report("mic level", False, "no frames captured")

    s = Settings.from_env()
    for kind, factory in (("stt", make_stt), ("tts", make_tts)):
        try:
            p = factory(s)
            report(kind, True, f"provider {p.name}")
        except Exception as e:
            report(kind, False, str(e))
    try:
        llm = make_llm(s)
        t0 = time.perf_counter()
        from .providers.base import Message

        text = ""
        async for ev in llm.stream("Reply with one word.", [Message("user", "Say ready.")], []):
            if hasattr(ev, "text") and isinstance(getattr(ev, "text"), str):
                text += ev.text
        report("llm", True, f"{llm.name} replied {text.strip()!r} in {1000*(time.perf_counter()-t0):.0f} ms")
    except Exception as e:
        report("llm", False, f"{type(e).__name__}: {e}")
    return 0 if ok else 1


async def _take(frames, seconds: float):
    end = time.perf_counter() + seconds
    async for f in frames:
        yield f
        if time.perf_counter() >= end:
            break


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="cooking-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="live voice agent")
    r.add_argument("--wake-word", default=None, help="require a wake word ('' disables; default from WAKE_WORD)")
    r.add_argument("--no-barge-in", action="store_true")
    r.add_argument("--input-device", default=None)
    r.add_argument("--output-device", default=None)

    c = sub.add_parser("chat", help="text mode with the real LLM and tools")
    c.add_argument("--speak", action="store_true", help="also speak replies through the configured TTS")

    sub.add_parser("demo", help="offline scripted demo with fake providers")

    rc = sub.add_parser("recipes", help="manage the saved recipe book")
    rsub = rc.add_subparsers(dest="action", required=True)
    ra = rsub.add_parser("add", help="import a recipe from a web page or a file (.md/.txt/.json)")
    ra.add_argument("source", help="URL or file path")
    ra.add_argument("--servings", type=int, default=None, help="override the servings the source states")
    ra.add_argument("--title", default=None, help="override the title")
    rl = rsub.add_parser("list", help="list saved recipes")
    rl.add_argument("query", nargs="?", default=None)
    rs = rsub.add_parser("show", help="print one recipe")
    rs.add_argument("id", type=int)
    rr = rsub.add_parser("remove", help="delete one recipe")
    rr.add_argument("id", type=int)

    pf = sub.add_parser("prefs", help="show or edit what the assistant knows about you")
    psub = pf.add_subparsers(dest="action")
    pa = psub.add_parser("add", help="add a stated preference")
    pa.add_argument("category", choices=["diet", "allergy", "dislike", "like", "pantry", "other"])
    pa.add_argument("value", nargs="+")
    pr = psub.add_parser("remove", help="remove stated preferences matching a phrase")
    pr.add_argument("value", nargs="+")
    psub.add_parser("forget-history", help="clear learned habits")

    b = sub.add_parser("bench", help="end-to-end latency benchmark over WAV fixtures")
    b.add_argument("--runs", type=int, default=1, help="passes over the fixture set")
    b.add_argument("--fake", action="store_true", help="offline fake providers (harness check, no keys)")
    b.add_argument("--no-realtime", action="store_true", help="feed audio as fast as possible (fake mode only)")
    b.add_argument("--only", nargs="*", help="fixture names to run, e.g. next_step set_timer")
    b.add_argument("--fixtures", default=None, help="directory with manifest.json (default bench/fixtures)")
    b.add_argument("--timeout", type=float, default=45.0, help="seconds per fixture")
    b.add_argument("--out", default=None, help="write the full JSON report here")

    d = sub.add_parser("diagnose", help="check devices, mic level and providers")
    d.add_argument("--mic-seconds", type=float, default=3.0, help="0 to skip the mic test")
    d.add_argument("--input-device", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "websockets", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    handler = {"run": _run, "chat": _chat, "demo": _demo, "bench": _bench, "recipes": _recipes, "prefs": _prefs,
               "diagnose": _diagnose}[args.cmd]
    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nbye")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
