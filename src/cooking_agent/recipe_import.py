"""Turn recipes from outside into validated ``Recipe`` objects, without an LLM.

* Web pages: almost every recipe site embeds the recipe as schema.org
  ``Recipe`` JSON-LD for search engines. We read that block: title, yield,
  ingredient lines and instructions, exactly as published.
* Text files: a simple Markdown layout the cook can write by hand::

      # Mum's Dal
      Serves: 4

      ## Ingredients
      - 1 cup toor dal, rinsed
      - 1/2 tsp turmeric
      - Salt to taste

      ## Steps
      1. Pressure cook the dal with 3 cups of water for 15 minutes.
      2. ...

* JSON files in the app's own ``Recipe`` schema.

Ingredient lines are parsed into quantity / unit / name / note with
``parse_ingredient_line`` so servings scaling works; a line it cannot parse
keeps its text and is simply not scaled. Step timers come from the step text.
"""

from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

from .recipe import Recipe, Step, parse_ingredient_line

DEFAULT_SERVINGS = 4
USER_AGENT = "Mozilla/5.0 (compatible; cooking-agent/0.3; personal recipe import)"


class RecipeImportError(ValueError):
    """The source could not be turned into a recipe; the message says why."""


# ------------------------------------------------------------------ helpers
def _clean(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split()).strip()


def parse_servings(value: Any) -> Optional[int]:
    """schema.org recipeYield: 4, '4', '4 servings', ['4', '4 servings'], 'Serves 4-6'."""
    if isinstance(value, list):
        for v in value:
            n = parse_servings(v)
            if n:
                return n
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if 1 <= n <= 50 else None
    m = re.search(r"\d+", str(value or ""))
    if m and 1 <= int(m.group()) <= 50:
        return int(m.group())
    return None


def build_recipe(title: str, servings: Optional[int], ingredients: Iterable[str],
                 steps: Iterable[str]) -> Recipe:
    ings = [parse_ingredient_line(line) for line in ingredients if _clean(line)]
    sts = [Step(instruction=_clean(s)) for s in steps if _clean(s)]
    if not ings:
        raise RecipeImportError("no ingredients found")
    if not sts:
        raise RecipeImportError("no steps found")
    return Recipe(title=_clean(title) or "Untitled recipe", servings=servings or DEFAULT_SERVINGS,
                  ingredients=ings, steps=sts)


# ------------------------------------------------------------------ web pages
_LD_JSON = re.compile(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def _walk(node: Any) -> Iterable[dict]:
    if isinstance(node, list):
        for n in node:
            yield from _walk(n)
    elif isinstance(node, dict):
        yield node
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                yield from _walk(node[key])


def _is_recipe(node: dict) -> bool:
    t = node.get("@type")
    return t == "Recipe" or (isinstance(t, list) and "Recipe" in t)


def _instructions(value: Any) -> list[str]:
    """recipeInstructions comes as a string, a list of strings, HowToStep
    objects, or HowToSection objects that contain steps."""
    if isinstance(value, str):
        text = html.unescape(value)
        parts = re.split(r"(?:<br\s*/?>|</p>|</li>|\n)+", text, flags=re.IGNORECASE)
        return [p for p in (_clean(x) for x in parts) if p]
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            out += _instructions(item)
    elif isinstance(value, dict):
        if "itemListElement" in value:
            out += _instructions(value["itemListElement"])
        else:
            text = value.get("text") or value.get("name")
            if text:
                out.append(_clean(text))
    return out


def recipe_from_html(page: str) -> Recipe:
    for block in _LD_JSON.findall(page):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for node in _walk(data):
            if _is_recipe(node):
                return build_recipe(
                    node.get("name", ""),
                    parse_servings(node.get("recipeYield")),
                    [_clean(x) for x in node.get("recipeIngredient") or node.get("ingredients") or []],
                    _instructions(node.get("recipeInstructions")),
                )
    raise RecipeImportError("this page has no machine-readable recipe (schema.org Recipe) in it")


def fetch(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except Exception as e:  # HTTPError (403 from bot blockers), URLError, timeouts
        raise RecipeImportError(f"could not download {url}: {e}") from e


def recipe_from_url(url: str) -> Recipe:
    return recipe_from_html(fetch(url))


# ------------------------------------------------------------------ text / files
_HEADING = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")
_SERVES = re.compile(r"^\s*(?:serves|servings|yield|makes)\s*[:\-]?\s*(.+)$", re.IGNORECASE)
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def recipe_from_text(text: str) -> Recipe:
    title, servings = "", None
    section: Optional[str] = None
    ingredients: list[str] = []
    steps: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        h = _HEADING.match(line)
        if h:
            name = h.group(1).lower()
            if not title and not raw.lstrip().startswith("##"):
                title = h.group(1)
            elif "ingredient" in name:
                section = "ingredients"
            elif any(k in name for k in ("step", "method", "instruction", "direction", "preparation")):
                section = "steps"
            continue
        s = _SERVES.match(line)
        if s and section is None:
            servings = parse_servings(s.group(1))
            continue
        item = _LIST_MARK.sub("", line)
        if section == "ingredients":
            ingredients.append(item)
        elif section == "steps":
            steps.append(item)
        elif not title:
            title = line
    return build_recipe(title, servings, ingredients, steps)


def recipe_from_file(path: str | Path) -> Recipe:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return Recipe.model_validate_json(text)
    return recipe_from_text(text)


def recipe_from_source(source: str) -> Recipe:
    """A URL or a file path."""
    if re.match(r"^https?://", source, re.IGNORECASE):
        return recipe_from_url(source)
    if not Path(source).is_file():
        raise RecipeImportError(f"{source} is neither a web address nor a file")
    return recipe_from_file(source)
