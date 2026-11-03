"""Recipe book: parsing, web/file import, the SQLite store, and the tools that use it."""

import json

import pytest

from cooking_agent.cli import main
from cooking_agent.providers.base import ToolCall
from cooking_agent.providers.fake import ScriptedLLM
from cooking_agent.recipe import format_minutes, parse_duration_minutes, parse_ingredient_line
from cooking_agent.recipe_import import RecipeImportError, recipe_from_html, recipe_from_text
from cooking_agent.recipe_store import RecipeStore
from cooking_agent.state import CookingSession
from cooking_agent.timers import TimerManager
from cooking_agent.tools import ToolBox


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,minutes", [
    ("Soak the rice for 20 minutes, then drain.", 20),
    ("Simmer for 10-12 minutes", 10),
    ("Cook for 1 to 2 hours", 60),
    ("Bake 1 hour 15 minutes", 75),
    ("Bake for 1½ hours", 90),
    ("Cook for an hour and a half", 90),
    ("Let it rest for half an hour", 30),
    ("Stir fry for three minutes", 3),
    ("Toast the cumin for 30 seconds", 0.5),
    ("Heat the oil in a pan.", None),
    ("Cut into 2 inch pieces.", None),
])
def test_parse_duration(text, minutes):
    assert parse_duration_minutes(text) == minutes


def test_format_minutes():
    assert [format_minutes(m) for m in (0.5, 1, 12, 1.5, 60, 90)] == [
        "30 seconds", "1 minute", "12 minutes", "1 and 1/2 minutes", "1 hour", "1 hour and 30 minutes"]


@pytest.mark.parametrize("line,expected", [
    ("1½ cups basmati rice, rinsed", (1.5, "cups", "basmati rice", "rinsed")),
    ("2 cloves garlic, minced", (2, "cloves", "garlic", "minced")),
    ("1 (14 oz) can chopped tomatoes", (1, "can", "chopped tomatoes", "14 oz")),
    ("2-3 tbsp ghee", (2, "tbsp", "ghee", None)),
    ("1/2 tsp. turmeric", (0.5, "tsp", "turmeric", None)),
    ("200g paneer", (200, "g", "paneer", None)),
    ("2 cups of flour", (2, "cups", "flour", None)),
    ("a pinch of salt", (1, "pinch", "salt", None)),
    ("3 eggs", (3, None, "eggs", None)),
    ("1 large onion (finely chopped)", (1, None, "large onion", "finely chopped")),
    ("Salt to taste", (None, None, "Salt to taste", None)),
])
def test_parse_ingredient_line(line, expected):
    i = parse_ingredient_line(line)
    assert (i.quantity, i.unit, i.name, i.note) == expected


def test_scaled_units_read_naturally():
    assert parse_ingredient_line("1 cup rice").describe(2) == "2 cups rice"
    assert parse_ingredient_line("2 cups rice").describe(0.5) == "1 cup rice"
    assert parse_ingredient_line("1 can tomatoes").describe(3) == "3 cans tomatoes"


# ------------------------------------------------------------------ import
PAGE = """<html><head>
<script type="application/ld+json">{"@context": "https://schema.org", "@type": "WebSite", "name": "x"}</script>
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Organization", "name": "Some Site"},
  {"@type": ["Recipe", "NewsArticle"], "name": "Easy Pancakes &amp; Syrup",
   "recipeYield": ["8", "8 pancakes"],
   "recipeIngredient": ["100g plain flour", "2 large eggs", "300ml milk", "1 tbsp sunflower oil, plus extra"],
   "recipeInstructions": [
     {"@type": "HowToSection", "name": "Batter", "itemListElement": [
       {"@type": "HowToStep", "text": "Whisk the flour, eggs and milk to a <b>smooth</b> batter."},
       {"@type": "HowToStep", "text": "Rest the batter for 30 minutes."}]},
     {"@type": "HowToStep", "text": "Cook each pancake for 1 minute on each side."}]}
]}
</script></head><body>...</body></html>"""


def test_recipe_from_html_reads_schema_org_json_ld():
    r = recipe_from_html(PAGE)
    assert r.title == "Easy Pancakes & Syrup"
    assert r.servings == 8
    assert [i.describe() for i in r.ingredients] == [
        "100 g plain flour", "2 large eggs", "300 ml milk", "1 tbsp sunflower oil, plus extra"]
    assert [s.instruction for s in r.steps] == [
        "Whisk the flour, eggs and milk to a smooth batter.",
        "Rest the batter for 30 minutes.",
        "Cook each pancake for 1 minute on each side.",
    ]
    assert [s.timer_minutes for s in r.steps] == [None, 30, 1]


def test_recipe_from_html_string_instructions_and_missing_yield():
    page = ('<script type="application/ld+json">{"@type": "Recipe", "name": "Dal", '
            '"recipeIngredient": ["1 cup toor dal"], '
            '"recipeInstructions": "<p>Rinse the dal.</p><p>Boil for 20 minutes.</p>"}</script>')
    r = recipe_from_html(page)
    assert r.servings == 4  # default when the page doesn't say
    assert [s.instruction for s in r.steps] == ["Rinse the dal.", "Boil for 20 minutes."]


def test_recipe_from_html_without_recipe_is_a_clear_error():
    with pytest.raises(RecipeImportError, match="no machine-readable recipe"):
        recipe_from_html("<html><body>just a blog post</body></html>")


MARKDOWN = """# Mum's Dal
Serves: 4

## Ingredients
- 1 cup toor dal, rinsed
- 1/2 tsp turmeric
- Salt to taste

## Method
1. Pressure cook the dal with 3 cups of water for 15 minutes.
2. Temper cumin in ghee and pour over.
"""


def test_recipe_from_text():
    r = recipe_from_text(MARKDOWN)
    assert (r.title, r.servings, len(r.ingredients), len(r.steps)) == ("Mum's Dal", 4, 3, 2)
    assert r.steps[0].timer_minutes == 15
    assert r.ingredients[2].quantity is None


# ------------------------------------------------------------------ store
def test_store_search_is_fuzzy_per_word_and_persistent(tmp_path, recipe):
    db = tmp_path / "book.db"
    store = RecipeStore(db)
    rid = store.add(recipe, source="test")
    dal = store.add(recipe_from_text(MARKDOWN), source="mum")
    assert store.find("jeera rice recipe").id == rid
    assert store.find("jeera rise").id == rid  # STT misspelling
    assert store.find("dal").id == dal
    assert store.find("mums dal").id == dal
    assert store.find("chicken curry") is None
    chicken = store.add(recipe.model_copy(update={"title": "Chicken Curry"}))
    assert store.find("chickpea curry") is None  # near in spelling, not in meaning
    assert store.find("chicken curry").id == chicken
    biryani = store.add(recipe.model_copy(update={"title": "Veg Biryani"}))
    assert store.find("biriyani").id == biryani
    store.close()
    again = RecipeStore(db)
    assert len(again) == 4 and again.get(rid).recipe == recipe
    assert again.remove(rid) and not again.remove(rid)


# ------------------------------------------------------------------ tools
@pytest.fixture
def book(tmp_path, recipe):
    store = RecipeStore(tmp_path / "book.db")
    store.add(recipe_from_text(MARKDOWN), source="mum")
    tb = ToolBox(CookingSession(), TimerManager(on_fire=lambda n: None), ScriptedLLM(recipe=recipe), store=store)
    return tb


async def call(tb, name, /, **args):
    return (await tb.dispatch(ToolCall(name, args))).result


async def test_get_recipe_prefers_the_recipe_book_and_scales(book):
    out = await call(book, "get_recipe", dish="dal", servings=8)
    assert out["title"] == "Mum's Dal" and "recipe book" in out["source"]
    assert out["servings"] == 8 and out["original_servings"] == 4
    assert out["ingredients"][0] == "2 cups toor dal, rinsed"
    assert book.session.snapshot()["recipe_source"] == "recipe book"


async def test_get_recipe_falls_back_to_generating(book):
    out = await call(book, "get_recipe", dish="jeera rice")
    assert out["title"] == "Jeera Rice" and "not from the recipe book" in out["source"]
    # a constraint the saved title doesn't show -> generate rather than ignore it
    out = await call(book, "get_recipe", dish="dal", constraints="vegan")
    assert out["title"] == "Jeera Rice"


async def test_generated_recipe_can_be_saved_once(book):
    await call(book, "get_recipe", dish="jeera rice", servings=2)
    assert (await call(book, "save_recipe"))["saved"] == "Jeera Rice"
    assert "already" in (await call(book, "save_recipe"))["error"]
    titles = [r["title"] for r in (await call(book, "list_saved_recipes"))["recipes"]]
    assert titles == ["Jeera Rice", "Mum's Dal"]


async def test_dictating_a_recipe_by_voice(book):
    assert "error" in await call(book, "add_to_dictated_recipe", steps=["x"])  # nothing started
    await call(book, "start_recipe_dictation", title="Grandma's Kheer")
    out = await call(book, "add_to_dictated_recipe", ingredients=["1 litre milk", "1/4 cup rice, washed"])
    assert out["added_ingredients"] == ["1 litre milk", "1/4 cup rice, washed"]
    await call(book, "add_to_dictated_recipe", steps=["Boil the milk.", "Add the rice and simmer for 40 minutes.",
                                                      "Oops wrong step."])
    assert (await call(book, "remove_from_dictated_recipe", step_number=3))["removed_step"] == "Oops wrong step."
    assert "serves" in (await call(book, "save_dictated_recipe"))["error"]  # servings missing
    await call(book, "add_to_dictated_recipe", servings=4)
    assert book.session.snapshot()["dictating_recipe"] == {
        "title": "Grandma's Kheer", "servings": 4, "ingredients": 2, "steps": 2}
    saved = await call(book, "save_dictated_recipe")
    assert saved["saved"] == "Grandma's Kheer" and book.session.draft is None
    # and it can be cooked, scaled, with the dictated time as a step timer
    out = await call(book, "get_recipe", dish="kheer", servings=8)
    assert out["ingredients"] == ["2 litres milk", "1/2 cup rice, washed"]
    await call(book, "next_step")
    assert (await call(book, "next_step"))["suggested_timer_minutes"] == 40


async def test_start_step_timer(book):
    await call(book, "get_recipe", dish="dal")
    step = await call(book, "next_step")
    assert step["suggested_timer_minutes"] == 15
    out = await call(book, "start_step_timer", name="dal")
    assert out["name"] == "dal" and out["seconds"] == 900 and out["duration"] == "15 minutes"
    # already started: repeating the step doesn't offer it again, and there's nothing to start
    assert "suggested_timer_minutes" not in await call(book, "repeat_step")
    assert "error" in await call(book, "start_step_timer")
    await book.timers.aclose()


# ------------------------------------------------------------------ CLI
def test_cli_recipes_add_list_show_remove(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECIPE_DB", str(tmp_path / "book.db"))
    f = tmp_path / "dal.md"
    f.write_text(MARKDOWN)
    assert main(["recipes", "add", str(f), "--servings", "2"]) == 0
    out = capsys.readouterr().out
    assert "#1 Mum's Dal (serves 2)" in out and "[timer: 15 minutes]" in out
    assert main(["recipes", "list", "dal"]) == 0
    assert "#1    Mum's Dal" in capsys.readouterr().out
    assert main(["recipes", "remove", "1"]) == 0
    assert main(["recipes", "show", "1"]) == 1
    assert main(["recipes", "add", str(tmp_path / "missing.md")]) == 1
    assert "neither a web address nor a file" in capsys.readouterr().out


def test_cli_recipes_add_json(tmp_path, monkeypatch, recipe):
    monkeypatch.setenv("RECIPE_DB", str(tmp_path / "book.db"))
    f = tmp_path / "r.json"
    f.write_text(json.dumps(recipe.model_dump()))
    assert main(["recipes", "add", str(f)]) == 0
    assert RecipeStore(tmp_path / "book.db").find("jeera rice").recipe == recipe
