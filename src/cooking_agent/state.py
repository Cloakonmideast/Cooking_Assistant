"""Explicit state machines.

Two independent machines:

* ``CookingSession`` - what the cook is doing: no recipe -> previewing a
  recipe -> cooking step N -> done. Tool handlers are the only code that
  mutates it, and they all run on the single asyncio event loop.
* ``TurnStateMachine`` - what the agent is doing in the conversation:
  listening -> thinking -> speaking -> listening. Illegal transitions raise,
  so ordering bugs show up in tests instead of as odd behaviour at runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Optional

from .recipe import Ingredient, Recipe


class InvalidTransition(RuntimeError):
    """Raised when a state machine is asked to make an illegal move.

    For CookingSession the message is phrased so it can be returned to the LLM
    (and spoken) as-is.
    """


# --------------------------------------------------------------------------
# Cooking session
# --------------------------------------------------------------------------
class CookingPhase(str, enum.Enum):
    NO_RECIPE = "no_recipe"
    PREVIEW = "preview"  # recipe loaded, cooking not started
    COOKING = "cooking"
    DONE = "done"


@dataclass
class CookingSession:
    recipe: Optional[Recipe] = None
    phase: CookingPhase = CookingPhase.NO_RECIPE
    step_index: int = -1
    target_servings: Optional[int] = None
    substitutions: list[dict] = field(default_factory=list)

    # -- recipe lifecycle -------------------------------------------------
    def load_recipe(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.phase = CookingPhase.PREVIEW
        self.step_index = -1
        self.target_servings = recipe.servings
        self.substitutions = []

    def _require_recipe(self) -> Recipe:
        if self.recipe is None:
            raise InvalidTransition("No recipe is loaded yet. Ask for a recipe first.")
        return self.recipe

    @property
    def scale_factor(self) -> float:
        recipe = self._require_recipe()
        return (self.target_servings or recipe.servings) / recipe.servings

    # -- step navigation --------------------------------------------------
    def next_step(self) -> dict:
        recipe = self._require_recipe()
        if self.phase == CookingPhase.DONE:
            raise InvalidTransition("You've already finished the last step.")
        if self.step_index + 1 >= len(recipe.steps):
            self.phase = CookingPhase.DONE
            return {"finished": True, "message": "That was the last step."}
        self.step_index += 1
        self.phase = CookingPhase.COOKING
        return self.current_step()

    def previous_step(self) -> dict:
        self._require_recipe()
        if self.phase == CookingPhase.DONE:
            # Going back from "done" re-opens the last step.
            self.phase = CookingPhase.COOKING
            return self.current_step()
        if self.phase != CookingPhase.COOKING:
            raise InvalidTransition("We haven't started cooking yet.")
        if self.step_index == 0:
            raise InvalidTransition("You're already on the first step.")
        self.step_index -= 1
        return self.current_step()

    def current_step(self) -> dict:
        recipe = self._require_recipe()
        if self.phase not in (CookingPhase.COOKING, CookingPhase.DONE) or self.step_index < 0:
            raise InvalidTransition("We haven't started cooking yet. Say 'next step' to begin.")
        step = recipe.steps[self.step_index]
        out: dict = {
            "step_number": self.step_index + 1,
            "total_steps": len(recipe.steps),
            "instruction": step.instruction,
            "is_last_step": self.step_index == len(recipe.steps) - 1,
        }
        if step.timer_minutes:
            out["suggested_timer_minutes"] = step.timer_minutes
        return out

    # -- ingredients ------------------------------------------------------
    def ingredient_lines(self) -> list[str]:
        recipe = self._require_recipe()
        factor = self.scale_factor
        return [ing.describe(factor) for ing in recipe.ingredients]

    def scale_servings(self, servings: int) -> dict:
        recipe = self._require_recipe()
        if not 1 <= servings <= 50:
            raise InvalidTransition("Servings must be between 1 and 50.")
        self.target_servings = servings
        return {
            "original_servings": recipe.servings,
            "servings": servings,
            "factor": round(self.scale_factor, 3),
            "ingredients": self.ingredient_lines(),
        }

    def substitute(self, ingredient: str, replacement: Ingredient) -> dict:
        recipe = self._require_recipe()
        idx = self._find_ingredient(ingredient)
        if idx is None:
            names = ", ".join(i.name for i in recipe.ingredients)
            raise InvalidTransition(f"I can't find {ingredient} in this recipe. Ingredients are: {names}.")
        original = recipe.ingredients[idx]
        note = f"instead of {original.name}"
        if replacement.note:
            note = f"{replacement.note}; {note}"
        new = replacement.model_copy(update={"note": note})
        recipe.ingredients[idx] = new
        record = {"original": original.name, "replacement": new.name}
        self.substitutions.append(record)
        return {**record, "line": new.describe(self.scale_factor)}

    def _find_ingredient(self, query: str) -> Optional[int]:
        recipe = self._require_recipe()
        q = query.lower().strip()
        names = [i.name.lower() for i in recipe.ingredients]
        for i, n in enumerate(names):  # exact match first
            if n == q:
                return i
        for i, n in enumerate(names):  # then substring either way ("rice" ~ "basmati rice")
            if q in n or n in q:
                return i
        return None

    def snapshot(self) -> dict:
        """Compact state summary given to the LLM each turn."""
        if self.recipe is None:
            return {"phase": self.phase.value}
        return {
            "phase": self.phase.value,
            "recipe": self.recipe.title,
            "servings": self.target_servings,
            "step": (self.step_index + 1) if self.step_index >= 0 else None,
            "total_steps": len(self.recipe.steps),
        }


# --------------------------------------------------------------------------
# Conversation turn state
# --------------------------------------------------------------------------
class TurnState(str, enum.Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    STOPPED = "stopped"


_ALLOWED: dict[TurnState, set[TurnState]] = {
    TurnState.LISTENING: {TurnState.THINKING, TurnState.SPEAKING, TurnState.STOPPED},
    # THINKING -> LISTENING: reply finished with nothing to say, or was cancelled.
    TurnState.THINKING: {TurnState.SPEAKING, TurnState.LISTENING, TurnState.THINKING, TurnState.STOPPED},
    # SPEAKING -> THINKING: a second LLM round (after a tool call) while audio still plays.
    # SPEAKING -> LISTENING: playback finished or barge-in.
    TurnState.SPEAKING: {TurnState.LISTENING, TurnState.THINKING, TurnState.STOPPED},
    TurnState.STOPPED: set(),
}


class TurnStateMachine:
    def __init__(self, on_change: Callable[[TurnState, TurnState], None] | None = None):
        self.state = TurnState.LISTENING
        self._on_change = on_change
        self.history: list[TurnState] = [self.state]

    def to(self, new: TurnState) -> None:
        if new == self.state and new != TurnState.THINKING:
            return
        if new not in _ALLOWED[self.state]:
            raise InvalidTransition(f"illegal turn transition {self.state.value} -> {new.value}")
        old, self.state = self.state, new
        self.history.append(new)
        if self._on_change:
            self._on_change(old, new)
