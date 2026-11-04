import pytest

from cooking_agent.recipe import Ingredient
from cooking_agent.state import CookingPhase, CookingSession, InvalidTransition, TurnState, TurnStateMachine


# ------------------------------------------------------------ cooking session
def test_requires_recipe():
    s = CookingSession()
    for op in (s.next_step, s.previous_step, s.current_step, s.ingredient_lines):
        with pytest.raises(InvalidTransition, match="No recipe"):
            op()


def test_full_walkthrough(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    assert s.phase is CookingPhase.PREVIEW
    with pytest.raises(InvalidTransition, match="haven't started"):
        s.current_step()
    with pytest.raises(InvalidTransition, match="haven't started"):
        s.previous_step()

    first = s.next_step()
    assert s.phase is CookingPhase.COOKING
    assert first["step_number"] == 1 and first["suggested_timer_minutes"] == 20
    assert s.next_step()["step_number"] == 2
    assert s.previous_step()["step_number"] == 1
    with pytest.raises(InvalidTransition, match="first step"):
        s.previous_step()
    s.next_step()
    last = s.next_step()
    assert last["step_number"] == 3 and last["is_last_step"]

    assert s.next_step() == {"finished": True, "message": "That was the last step."}
    assert s.phase is CookingPhase.DONE
    with pytest.raises(InvalidTransition, match="already finished"):
        s.next_step()
    # going back from DONE reopens the last step
    assert s.previous_step()["step_number"] == 3
    assert s.phase is CookingPhase.COOKING


def test_loading_new_recipe_resets_progress(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    s.next_step()
    s.scale_servings(4)
    s.load_recipe(recipe.model_copy(deep=True))
    assert (s.phase, s.step_index, s.target_servings) == (CookingPhase.PREVIEW, -1, 2)


def test_scale_servings(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    out = s.scale_servings(6)
    assert out["factor"] == 3
    assert out["ingredients"][0] == "3 cups basmati rice, rinsed"
    assert out["ingredients"][2] == "4 and 1/2 tbsp ghee"
    assert out["ingredients"][3] == "salt"
    with pytest.raises(InvalidTransition):
        s.scale_servings(0)


def test_substitute_keeps_scaling_and_records(recipe):
    s = CookingSession()
    s.load_recipe(recipe)
    s.scale_servings(4)
    out = s.substitute("ghee", Ingredient(name="sunflower oil", quantity=1, unit="tbsp"))
    assert out["original"] == "ghee"
    assert out["line"] == "2 tbsp sunflower oil, instead of ghee"
    assert s.substitutions == [{"original": "ghee", "replacement": "sunflower oil"}]
    # substring matching: "rice" finds "basmati rice"
    assert s.substitute("rice", Ingredient(name="brown rice"))["original"] == "basmati rice"
    with pytest.raises(InvalidTransition, match="can't find"):
        s.substitute("chicken", Ingredient(name="tofu"))


def test_snapshot(recipe):
    s = CookingSession()
    assert s.snapshot() == {"phase": "no_recipe"}
    s.load_recipe(recipe)
    s.next_step()
    assert s.snapshot() == {"phase": "cooking", "recipe": "Jeera Rice", "servings": 2, "step": 1, "total_steps": 3,
                            "current_step_timer": "20 minutes"}


# ------------------------------------------------------------ turn state
def test_turn_state_happy_path():
    seen = []
    t = TurnStateMachine(on_change=lambda a, b: seen.append((a, b)))
    t.to(TurnState.THINKING)
    t.to(TurnState.SPEAKING)
    t.to(TurnState.THINKING)  # second LLM round after a tool call
    t.to(TurnState.SPEAKING)
    t.to(TurnState.LISTENING)
    t.to(TurnState.LISTENING)  # no-op
    assert len(seen) == 5
    t.to(TurnState.STOPPED)
    with pytest.raises(InvalidTransition):
        t.to(TurnState.LISTENING)


def test_turn_state_rejects_illegal():
    t = TurnStateMachine()
    t.to(TurnState.STOPPED)
    with pytest.raises(InvalidTransition):
        t.to(TurnState.THINKING)
