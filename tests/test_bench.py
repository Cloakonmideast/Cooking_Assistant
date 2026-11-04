"""The latency benchmark harness, run offline with fake providers."""

import json

from cooking_agent.bench import DEFAULT_FIXTURES, format_report, load_manifest, run_benchmark, summarize
from cooking_agent.cli import main


def test_manifest_fixtures_exist_and_have_expectations():
    raw = json.loads((DEFAULT_FIXTURES / "manifest.json").read_text())["fixtures"]
    fixtures = load_manifest(DEFAULT_FIXTURES)
    assert len(fixtures) == len(raw) == 10
    for entry, fx in zip(raw, fixtures):
        assert fx.file.is_file(), fx.file
        assert "expect_tool" in entry, entry["file"]


async def test_fake_benchmark_routes_every_fixture_correctly():
    report = await run_benchmark(fake=True, realtime=False, progress=lambda _: None)
    s = report["summary"]
    assert s["errors"] == 0
    assert s["tool_accuracy"] == 1.0
    assert s["llm_first_token_ms"]["n"] == 10
    assert s["first_audio_ms"]["n"] == 10
    # Without real-time pacing the end of speech is unknown, so e2e is not reported.
    assert s["e2e_first_audio_ms"]["n"] == 0
    assert "tool routing accuracy: 100%" in format_report(report)


async def test_realtime_benchmark_measures_from_end_of_speech():
    report = await run_benchmark(fake=True, realtime=True, only=["next_step"], progress=lambda _: None)
    turn = report["turns"][0]
    # FixtureSTT endpoints 300 ms after speech ends (minus frame rounding).
    assert 200 <= turn["stt_endpoint_ms"] <= 450
    assert turn["e2e_first_audio_ms"] >= turn["stt_endpoint_ms"]


def test_summarize_counts_errors_and_skips_them_for_accuracy():
    rows = [
        {"error": None, "tool_ok": True, "llm_first_token_ms": 100.0},
        {"error": None, "tool_ok": False, "llm_first_token_ms": 300.0},
        {"error": "timeout", "tool_ok": False, "llm_first_token_ms": None},
    ]
    s = summarize(rows)
    assert s["errors"] == 1
    assert s["tool_accuracy"] == 0.5
    assert s["llm_first_token_ms"] == {"n": 2, "p50": 100.0, "p90": 300.0, "max": 300.0}


def test_cli_bench_writes_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(["bench", "--fake", "--no-realtime", "--only", "set_timer", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["turns"][0]["first_tool"] == "set_timer"
    assert "llm_first_token_ms" in capsys.readouterr().out
