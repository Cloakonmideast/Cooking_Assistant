"""Automatic diet test: does the LLM's recipe keep to a cook's diet, allergy or
health condition, after the app's own safety check?

Each case asks for a dish that normally breaks the rule but doesn't name the
forbidden ingredient (a vegetarian asking for pad thai, which usually has
fish sauce and egg), so the rule must hold. The recipe goes through the
normal ``get_recipe`` path: profile in the prompt, the code check, one
rewrite if needed. A case passes when the final recipe is valid and has no
conflict left. Used to choose a local model (``cooking-agent diet-eval``).
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .preferences import PreferenceStore
from .providers.base import ToolCall
from .recipe_store import RecipeStore
from .state import CookingSession
from .timers import TimerManager
from .tools import ToolBox


@dataclass(frozen=True)
class DietCase:
    dish: str
    diet: Optional[str] = None
    allergy: Optional[str] = None
    condition: Optional[str] = None

    @property
    def label(self) -> str:
        rule = self.diet or (f"{self.allergy} allergy" if self.allergy else self.condition)
        return f"{rule}: {self.dish}"


DIET_CASES: list[DietCase] = [
    DietCase("pad thai", diet="vegetarian"),
    DietCase("caesar salad", diet="vegetarian"),
    DietCase("ramen", diet="vegetarian"),
    DietCase("kheer", diet="vegan"),
    DietCase("chocolate cake", diet="vegan"),
    DietCase("dal tadka", diet="jain"),
    DietCase("pav bhaji", diet="jain"),
    DietCase("paneer butter masala", condition="lactose intolerance"),
    DietCase("pasta", condition="celiac disease"),
    DietCase("satay", allergy="peanuts"),
]


class _Counting:
    """Counts the LLM's recipe writes, so a rewrite can be seen."""

    def __init__(self, llm):
        self.llm, self.calls = llm, 0
        self.name = getattr(llm, "name", "llm")

    async def generate_structured(self, prompt, schema, media=None):
        self.calls += 1
        return await self.llm.generate_structured(prompt, schema)


async def _run_case(llm, case: DietCase, folder: Path) -> dict:
    db = folder / f"case-{abs(hash(case.label))}.db"
    prefs = PreferenceStore(db)
    for category, value in (("diet", case.diet), ("allergy", case.allergy), ("condition", case.condition)):
        if value:
            prefs.remember(category, value)
    prefs.mark_setup_done()
    counting = _Counting(llm)
    store = RecipeStore(db)
    box = ToolBox(CookingSession(), TimerManager(on_fire=lambda name: None), counting, store=store, prefs=prefs)
    t0 = time.perf_counter()
    try:
        out = (await box.dispatch(ToolCall("get_recipe", {"dish": case.dish, "servings": 2}))).result
    finally:
        store.close()  # Windows can't delete the temporary folder while the files are open
        prefs.close()
    recipe = box.session.recipe
    warnings = list(out.get("warnings", []))
    valid = "error" not in out and recipe is not None
    return {
        **asdict(case),
        "label": case.label,
        "valid": valid,
        "error": out.get("error"),
        "warnings": warnings,
        "rewrites": max(0, counting.calls - 1),
        "passed": valid and not warnings,
        "seconds": round(time.perf_counter() - t0, 1),
        "title": recipe.title if recipe else None,
        "ingredients": [i.describe() for i in recipe.ingredients] if recipe else [],
        "steps": [s.instruction for s in recipe.steps] if recipe else [],
    }


async def run_diet_eval(llm, cases: list[DietCase] = DIET_CASES, progress=print) -> dict:
    results: list[dict] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for case in cases:
            r = await _run_case(llm, case, Path(tmp))
            results.append(r)
            progress(_line(r))
    summary = {
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "errors": sum(not r["valid"] for r in results),
        "rewrites": sum(r["rewrites"] for r in results),
        "model": getattr(llm, "model", getattr(llm, "name", "?")),
    }
    return {"cases": results, "summary": summary}


def _line(r: dict) -> str:
    status = "PASS" if r["passed"] else ("ERROR" if not r["valid"] else "FAIL")
    extra = r["error"] or "; ".join(r["warnings"])
    rewrite = f" (rewritten {r['rewrites']}x)" if r["rewrites"] else ""
    return f"  {status:<5} {r['label']:<42}{rewrite} {extra}".rstrip()


def format_diet_report(report: dict) -> str:
    s = report["summary"]
    return (f"diet test ({s['model']}): {s['passed']}/{s['total']} passed, "
            f"{s['errors']} errors, {s['rewrites']} rewrites needed")
