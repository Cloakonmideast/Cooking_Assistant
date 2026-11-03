import pytest

from cooking_agent.recipe import Recipe


@pytest.fixture
def recipe() -> Recipe:
    return Recipe.model_validate({
        "title": "Jeera Rice",
        "servings": 2,
        "ingredients": [
            {"name": "basmati rice", "quantity": 1, "unit": "cup", "note": "rinsed"},
            {"name": "cumin seeds", "quantity": 1, "unit": "tsp"},
            {"name": "ghee", "quantity": 1.5, "unit": "tbsp"},
            {"name": "salt", "quantity": None},
        ],
        "steps": [
            {"instruction": "Soak the rice for 20 minutes.", "timer_minutes": 20},
            {"instruction": "Heat ghee and add cumin seeds."},
            {"instruction": "Add rice, water and salt; cook covered for 12 minutes.", "timer_minutes": 12},
        ],
    })
