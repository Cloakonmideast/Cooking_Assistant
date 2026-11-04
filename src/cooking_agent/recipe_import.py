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

import asyncio
import html
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, PrivateAttr

from .recipe import Recipe, Step, parse_ingredient_line

log = logging.getLogger(__name__)

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


# ================================================================== any source
# Captions, transcripts, pages without structured data, photos and PDFs have
# no fixed layout, so the LLM finds the recipe in them. It is only allowed to
# copy: every line it returns is checked against the source text, word for
# word, and sent back once if it was reworded.

class ExtractedRecipe(BaseModel):
    found: bool = Field(description="False if the source contains no recipe.")
    title: str = Field(default="", description="The recipe's name as given in the source.")
    servings: Optional[int] = Field(default=None, ge=1, le=50,
                                    description="Servings if the source states them, else null.")
    ingredients: list[str] = Field(default_factory=list,
                                   description="Each ingredient line copied exactly as written or said.")
    steps: list[str] = Field(default_factory=list, description="Each step copied exactly as written or said.")
    _reworded: list[str] = PrivateAttr(default_factory=list)  # lines that failed the word-for-word check


EXTRACT_PROMPT = """\
Below is {what}. Find the recipe in it and copy it out EXACTLY.

Rules:
- Copy every ingredient line and every step word for word from the source. Do not rephrase, \
summarise, translate, convert units, change numbers or add anything that isn't there.
- You may only: split the text into separate ingredient lines and steps (one action or \
sentence per step is fine), leave out things that are not part of the recipe (greetings, \
hashtags, emojis, "link in bio", "like and subscribe", sponsor messages, filler words like \
"um" or "you know"), and drop list numbering or bullets.
- If the creator only says the ingredients while cooking, list each ingredient with the \
amount in the words they used.
- Use the title given in the source; if none, name the dish briefly.
- If there is no recipe, set found to false.
{feedback}
{source}"""


@dataclass
class ImportResult:
    recipe: Recipe
    how: str  # what was read, e.g. "the post's caption" or "the spoken audio (transcribed)"
    warnings: list[str] = field(default_factory=list)


_TOKEN = re.compile(r"[a-z0-9]+(?:[/.][0-9]+)?")


def _tokens(text: str) -> list[str]:
    from .recipe import normalize_fractions

    return _TOKEN.findall(normalize_fractions(text).lower().replace("’", "'"))


def is_verbatim(line: str, source_tokens: list[str]) -> bool:
    """True if ``line``'s words appear in the source in the same order, allowing
    a few skipped source words (filler, emojis, bullets) but no new words."""
    words = _tokens(line)
    if not words:
        return True
    allow = max(3, len(words) // 3)
    first = words[0]
    for i, tok in enumerate(source_tokens):
        if tok != first:
            continue
        j, k, skipped = 1, i + 1, 0
        while j < len(words) and k < len(source_tokens):
            if source_tokens[k] == words[j]:
                j += 1
            else:
                skipped += 1
                if skipped > allow:
                    break
            k += 1
        if j == len(words):
            return True
    return False


def not_verbatim(extracted: ExtractedRecipe, source: str) -> list[str]:
    toks = _tokens(source)
    return [line for line in extracted.ingredients + extracted.steps if not is_verbatim(line, toks)]


async def extract_recipe(llm, source_text: str, what: str, media: Optional[list] = None) -> Optional[ExtractedRecipe]:
    """Ask the LLM to copy the recipe out of ``source_text`` (or out of ``media``
    for photos/PDFs). Returns None if there's no usable recipe. For text
    sources, reworded lines are sent back once; ``extracted.reworded`` lists
    any that are still not word for word."""
    if llm is None:
        raise RecipeImportError("reading this kind of source needs the LLM: set GEMINI_API_KEY")
    source_block = f"SOURCE:\n{source_text}" if source_text else ""
    prompt = EXTRACT_PROMPT.format(what=what, feedback="", source=source_block)
    out = await llm.generate_structured(prompt, ExtractedRecipe, media=media)
    reworded: list[str] = []
    if source_text and out.found:
        reworded = not_verbatim(out, source_text)
        if reworded:
            log.info("extractor reworded %d line(s); asking again", len(reworded))
            feedback = ("\nYour previous answer reworded these lines; copy them exactly as in the source "
                        "instead:\n" + "\n".join(f"- {r}" for r in reworded) + "\n")
            retry = await llm.generate_structured(
                EXTRACT_PROMPT.format(what=what, feedback=feedback, source=source_block), ExtractedRecipe, media=media)
            if retry.found:
                out, reworded = retry, not_verbatim(retry, source_text)
    if not (out.found and out.ingredients and out.steps):
        return None
    out._reworded = reworded
    return out


def _result(ex: ExtractedRecipe, how: str, fallback_title: str = "") -> ImportResult:
    recipe = build_recipe(ex.title or fallback_title, ex.servings, ex.ingredients, ex.steps)
    warnings = []
    if ex.servings is None:
        warnings.append(f"the source doesn't say how many it serves; assumed {DEFAULT_SERVINGS} "
                        "(use --servings to set it, since scaling starts from this)")
    reworded = ex._reworded
    if reworded:
        warnings.append("these lines are not word for word from the source, please check them: "
                        + " | ".join(reworded))
    return ImportResult(recipe, how, warnings)


def page_text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|li|h[1-6]|div|tr)>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:60_000]


async def import_recipe(source: str, *, llm=None, transcriber=None, fetcher=None) -> ImportResult:
    """Import from a web page, a social media/video link, or a local file
    (Markdown/text, JSON, HTML, photo/screenshot, PDF, audio or video)."""
    from . import media_import as mi

    if re.match(r"^https?://", source, re.IGNORECASE):
        if mi.is_media_url(source):
            return await _from_media_url(source, llm, transcriber, fetcher or mi.YtDlpFetcher())
        page = await asyncio.to_thread(fetch, source)
        return await _from_html(page, llm, "the web page")

    path = Path(source).expanduser()
    if not path.is_file():
        raise RecipeImportError(f"{source} is neither a web address nor a file")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return ImportResult(recipe_from_file(path), "the JSON file")
    if suffix in (".html", ".htm"):
        return await _from_html(path.read_text(encoding="utf-8", errors="replace"), llm, "the saved web page")
    if suffix in mi.IMAGE_SUFFIXES or suffix == ".pdf":
        what = "a PDF" if suffix == ".pdf" else "a photo or screenshot"
        ex = await extract_recipe(llm, "", f"{what} of a recipe (attached)",
                                  media=[mi.MediaPart(path.read_bytes(), mi.mime_for(path))])
        if ex is None:
            raise RecipeImportError(f"no recipe found in {path.name}")
        res = _result(ex, f"the {'PDF' if suffix == '.pdf' else 'image'}", path.stem)
        res.warnings.append("read from an image/PDF, so the wording couldn't be checked automatically; "
                            "check it with `cooking-agent recipes show`")
        return res
    if suffix in mi.AUDIO_VIDEO_SUFFIXES:
        if transcriber is None:
            raise RecipeImportError("transcribing needs DEEPGRAM_API_KEY or GEMINI_API_KEY")
        transcript = await transcriber.transcribe(path)
        ex = await extract_recipe(llm, transcript, "a transcript of a cooking video/recording")
        if ex is None:
            raise RecipeImportError(f"no recipe found in what is said in {path.name}")
        return _result(ex, f"the spoken audio of {path.name} (transcribed)", path.stem)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return ImportResult(recipe_from_text(text), "the text file")
    except RecipeImportError:  # not in our layout (e.g. a pasted caption): let the LLM find it
        ex = await extract_recipe(llm, text, "a text file")
        if ex is None:
            raise RecipeImportError(f"no recipe found in {path.name}")
        return _result(ex, "the text file", path.stem)


async def _from_html(page: str, llm, where: str) -> ImportResult:
    try:
        return ImportResult(recipe_from_html(page), f"{where} (its published recipe data)")
    except RecipeImportError:
        text = page_text(page)
        ex = await extract_recipe(llm, text, f"the text of {where}")
        if ex is None:
            raise RecipeImportError(f"no recipe found on {where}")
        return _result(ex, f"the text of {where}")


async def _from_media_url(url: str, llm, transcriber, fetcher) -> ImportResult:
    from . import media_import as mi

    info = await asyncio.to_thread(fetcher.info, url)
    caption = info.caption.strip()
    if caption:  # many creators write the full recipe in the caption/description
        ex = await extract_recipe(llm, caption, "the caption/description of a cooking post")
        if ex is not None:
            return _result(ex, "the post's caption/description", info.title)
    spoken, how = info.subtitles, "the video's subtitles"
    if not spoken:
        if transcriber is None:
            raise RecipeImportError("the caption has no full recipe and transcribing the audio needs "
                                    "DEEPGRAM_API_KEY or GEMINI_API_KEY")
        spoken, how = await mi.transcribe_url(url, fetcher, transcriber), "the spoken audio (transcribed)"
    combined = (f"CAPTION:\n{caption}\n\n" if caption else "") + f"WHAT IS SAID IN THE VIDEO:\n{spoken}"
    ex = await extract_recipe(llm, combined, "a cooking video's caption and what is said in it")
    if ex is None:
        raise RecipeImportError("no recipe found in the caption or in what is said in the video")
    return _result(ex, f"{'the caption and ' if caption else ''}{how}", info.title)
