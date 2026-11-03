import json

import pytest
from pydantic import ValidationError

from cooking_agent.recipe import Ingredient, Recipe, format_quantity


def test_valid_recipe_round_trips(recipe):
    again = Recipe.model_validate_json(recipe.model_dump_json())
    assert again == recipe


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(steps=[]),
    lambda d: d.update(ingredients=[]),
    lambda d: d.update(servings=0),
    lambda d: d.update(title=""),
    lambda d: d["ingredients"].append({"name": "   "}),
    lambda d: d["ingredients"].append({"name": "x", "quantity": -1}),
    lambda d: d["steps"].append({"instruction": "wait", "timer_minutes": 0}),
    lambda d: d.pop("steps"),
])
def test_invalid_recipes_rejected(recipe, mutate):
    data = recipe.model_dump()
    mutate(data)
    with pytest.raises(ValidationError):
        Recipe.model_validate(data)


def test_old_string_list_format_is_rejected():
    # The previous implementation's free-form shape must not silently pass.
    old = {"title": "x", "ingredients": ["rice"], "steps": ["cook"]}
    with pytest.raises(ValidationError):
        Recipe.model_validate(old)


def test_markdown_wrapped_json_is_not_accepted():
    payload = "```json\n" + json.dumps({"title": "x"}) + "\n```"
    with pytest.raises(ValidationError):
        Recipe.model_validate_json(payload)


def test_json_schema_is_usable_as_structured_output_schema():
    schema = Recipe.model_json_schema()
    assert set(schema["required"]) == {"title", "servings", "ingredients", "steps"}
    assert "$defs" in schema and "Ingredient" in schema["$defs"]


@pytest.mark.parametrize("q,expected", [
    (1, "1"), (1.5, "1 and 1/2"), (0.25, "1/4"), (0.333, "1/3"), (200.0, "200"), (2.75, "2 and 3/4"),
])
def test_format_quantity(q, expected):
    assert format_quantity(q) == expected


def test_ingredient_describe_scales():
    ing = Ingredient(name="ghee", quantity=1.5, unit="tbsp", note="melted")
    assert ing.describe() == "1 and 1/2 tbsp ghee, melted"
    assert ing.describe(2) == "3 tbsp ghee, melted"
    assert Ingredient(name="salt").describe(3) == "salt"
