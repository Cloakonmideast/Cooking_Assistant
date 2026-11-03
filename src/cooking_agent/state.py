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

from .recipe import Ingredient, Recipe, Step, format_minutes


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


class Awaiting(str, enum.Enum):
    """The yes/no question the assistant last asked after a step, so a bare
    "yes" from the cook can be routed to the right tool."""

    TIMER = "start_timer"  # "Do you want me to start a timer for 12 minutes?"
    NEXT_STEP = "next_step"  # "Are you ready for the next step?"


@dataclass
class RecipeDraft:
    """A recipe the cook is dictating by voice, built up across turns."""

    title: str
    servings: Optional[int] = None
    ingredients: list[Ingredient] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "title": self.title,
            "servings": self.servings,
            "ingredients": [i.describe() for i in self.ingredients],
            "steps": [s.instruction for s in self.steps],
        }

    def to_recipe(self) -> Recipe:
        missing = [what for what, ok in (("how many people it serves", self.servings),
                                         ("any ingredients", self.ingredients),
                                         ("any steps", self.steps)) if not ok]
        if missing:
            raise InvalidTransition(f"The recipe doesn't have {' or '.join(missing)} yet.")
        return Recipe(title=self.title, servings=self.servings, ingredients=list(self.ingredients),
                      steps=list(self.steps))


@dataclass
class CookingSession:
    recipe: Optional[Recipe] = None
    phase: CookingPhase = CookingPhase.NO_RECIPE
    step_index: int = -1
    target_servings: Optional[int] = None
    substitutions: list[dict] = field(default_factory=list)
    recipe_source: Optional[str] = None  # 'recipe book' or 'generated'
    # Step timer the cook has been offered for the current step: {"minutes", "step_number"}.
    timer_offer: Optional[dict] = None
    awaiting: Optional[Awaiting] = None
    timed_steps: set[int] = field(default_factory=set)  # step numbers whose timer was started
    draft: Optional[RecipeDraft] = None
    setup_pending: bool = False  # first-run question about allergies/conditions/diet is open

    # -- recipe lifecycle -------------------------------------------------
    def load_recipe(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.phase = CookingPhase.PREVIEW
        self.step_index = -1
        self.target_servings = recipe.servings
        self.substitutions = []
        self.timer_offer = None
        self.awaiting = None
        self.timed_steps = set()

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
            self.timer_offer = None
            self.awaiting = None
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
        if step.timer_minutes and (self.step_index + 1) not in self.timed_steps:
            out["suggested_timer_minutes"] = step.timer_minutes
            self.timer_offer = {"minutes": step.timer_minutes, "step_number": self.step_index + 1}
        else:
            self.timer_offer = None
        return out

    def take_timer_offer(self) -> dict:
        """The current step's timer, for start_step_timer. Raises if there is none."""
        self._require_recipe()
        if self.phase != CookingPhase.COOKING or not self.timer_offer:
            raise InvalidTransition("The current step doesn't have a time to set a timer for. "
                                    "You can still ask for a timer with a number of minutes.")
        offer, self.timer_offer = self.timer_offer, None
        self.timed_steps.add(offer["step_number"])
        return offer

    @property
    def on_last_step(self) -> bool:
        return self.recipe is not None and self.step_index == len(self.recipe.steps) - 1

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
        out: dict = {"phase": self.phase.value}
        if self.recipe is not None:
            out.update({
                "recipe": self.recipe.title,
                "servings": self.target_servings,
                "step": (self.step_index + 1) if self.step_index >= 0 else None,
                "total_steps": len(self.recipe.steps),
            })
            if self.recipe_source:
                out["recipe_source"] = self.recipe_source
        if self.setup_pending:
            out["first_time_setup"] = "in progress: the app asked about allergies, conditions and diet"
        if self.awaiting:
            out["awaiting_answer_to"] = self.awaiting.value
        if self.timer_offer:
            out["current_step_timer"] = format_minutes(self.timer_offer["minutes"])
        if self.draft is not None:
            d = self.draft
            out["dictating_recipe"] = {"title": d.title, "servings": d.servings,
                                       "ingredients": len(d.ingredients), "steps": len(d.steps)}
        return out


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
