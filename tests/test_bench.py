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
    # Instant commands (no LLM) answer some fixtures; the rest reach the LLM.
    instant = sum(1 for t in report["turns"] if t.get("llm_rounds") == 0)
    assert instant >= 1
    assert s["llm_first_token_ms"]["n"] == 10 - instant
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


# ------------------------------------------------------------ measuring the model itself
async def test_no_instant_sends_matched_fixture_to_llm():
    from cooking_agent.bench import run_benchmark

    with_instant = await run_benchmark(fake=True, realtime=False, only=["next_step"], progress=lambda _: None)
    assert with_instant["turns"][0]["llm_rounds"] == 0  # answered by an instant command
    without = await run_benchmark(fake=True, realtime=False, only=["next_step"], instant=False,
                                  progress=lambda _: None)
    assert without["turns"][0]["llm_rounds"] >= 1 and without["turns"][0]["tool_ok"]


def test_text_stt_flag_uses_fixture_stt(monkeypatch):
    from cooking_agent.bench import FixtureSTT, _make_providers, load_manifest

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TTS_PROVIDER", "pyttsx3")
    fx = load_manifest(DEFAULT_FIXTURES)[0]
    stt, llm, _ = _make_providers(False, fx, 1.0, text_stt=True)
    assert isinstance(stt, FixtureSTT) and llm.name == "ollama"


# ------------------------------------------------------------ diet test (diet-eval)
from cooking_agent.diet_eval import DIET_CASES, run_diet_eval  # noqa: E402
from cooking_agent.recipe import Recipe  # noqa: E402


def _dish(*names):
    return Recipe.model_validate({"title": "Test", "servings": 2,
                                  "ingredients": [{"name": n, "quantity": 1} for n in names],
                                  "steps": [{"instruction": "Cook."}]})


class ChickenFirstLLM:
    """Writes chicken first; fixes it when sent back (or never, if stubborn)."""

    name = "fake"

    def __init__(self, stubborn=False):
        self.stubborn = stubborn

    async def generate_structured(self, prompt, schema, media=None):
        if "Your previous version included" in prompt and not self.stubborn:
            return _dish("rice", "salt")
        return _dish("chicken", "rice")


def test_diet_cases_never_name_a_forbidden_ingredient():
    from cooking_agent.dietary import find_conflicts

    assert len(DIET_CASES) == 10
    for case in DIET_CASES:
        if case.diet:
            assert not find_conflicts([case.dish], diets=[case.diet]), case


async def test_diet_eval_scores_cases():
    report = await run_diet_eval(ChickenFirstLLM(), progress=lambda _: None)
    s = report["summary"]
    diet_cases = sum(1 for c in DIET_CASES if c.diet)
    assert s["total"] == 10 and s["passed"] == 10 and s["errors"] == 0
    assert s["rewrites"] == diet_cases  # every diet case needed one rewrite


async def test_diet_eval_counts_failures():
    report = await run_diet_eval(ChickenFirstLLM(stubborn=True), progress=lambda _: None)
    assert report["summary"]["passed"] == 10 - sum(1 for c in DIET_CASES if c.diet)
    failed = [c for c in report["cases"] if not c["passed"]]
    assert all("chicken" in " ".join(c["warnings"]) for c in failed)
