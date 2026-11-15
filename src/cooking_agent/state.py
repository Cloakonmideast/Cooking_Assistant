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

import re

from .recipe import Ingredient, Recipe, Step, format_minutes, parse_ingredient_line

# Words that don't identify an ingredient on their own ("cumin seeds" is found by "cumin").
_FILLER_WORDS = {"the", "a", "an", "some", "of", "any", "my", "this", "that", "few", "bit", "little"}
_GENERIC_WORDS = {"seed", "powder", "leaf", "leave", "oil", "sauce", "paste", "nut", "fresh", "ground", "whole",
                  "chopped", "dried", "green", "red", "white", "black", "large", "small"}
# "double the garlic", "less salt": multipliers for change_amount.
AMOUNT_WORDS = {"double": 2.0, "twice": 2.0, "triple": 3.0, "half": 0.5, "halve": 0.5,
                "more": 1.5, "extra": 1.5, "less": 0.5}


def _word_set(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _FILLER_WORDS:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith(("ches", "shes", "oes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return out


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

    # -- changes asked for mid-recipe (change_recipe) -----------------------
    def find_ingredient_words(self, query: str) -> Optional[int]:
        """Whole-word match: "cumin" finds "cumin seeds", "ice" doesn't find "rice"."""
        recipe = self._require_recipe()
        q = _word_set(query)
        if not q:
            return None
        names = [_word_set(i.name) for i in recipe.ingredients]
        for i, n in enumerate(names):
            if n == q:
                return i
        for i, n in enumerate(names):
            if set(q) <= set(n) or (n and set(n) <= set(q)):
                return i
        return None

    def _require_ingredient(self, query: str) -> int:
        idx = self.find_ingredient_words(query)
        if idx is None:
            names = ", ".join(i.name for i in self._require_recipe().ingredients)
            raise InvalidTransition(f"I can't find {query} in this recipe. Ingredients are: {names}.")
        return idx

    def steps_mentioning(self, name: str) -> list[dict]:
        """Steps from the current one on that mention the ingredient, so the
        assistant can rewrite them (steps are read out word for word)."""
        recipe = self._require_recipe()
        keys = {w for w in _word_set(name) if w not in _GENERIC_WORDS} or set(_word_set(name))
        first = self.step_index if self.phase is CookingPhase.COOKING else 0
        out = []
        for n, step in enumerate(recipe.steps[first:], start=first + 1):
            if keys & set(_word_set(step.instruction)):
                out.append({"step_number": n, "instruction": step.instruction})
        return out

    def remove_ingredient(self, query: str) -> dict:
        recipe = self._require_recipe()
        idx = self._require_ingredient(query)
        if len(recipe.ingredients) == 1:
            raise InvalidTransition("That's the only ingredient left, so I can't take it out.")
        removed = recipe.ingredients.pop(idx)
        return {"removed": removed.describe(self.scale_factor), "ingredient": removed.name,
                "steps_to_update": self.steps_mentioning(removed.name)}

    def add_ingredient(self, line: str) -> dict:
        """``line`` is said at the current servings ("1 cup peas" for 4)."""
        recipe = self._require_recipe()
        ing = parse_ingredient_line(line)
        if ing.quantity is not None:
            ing = ing.model_copy(update={"quantity": ing.quantity / self.scale_factor})
        ing = ing.model_copy(update={"original": None})
        recipe.ingredients.append(ing)
        return {"added": ing.describe(self.scale_factor), "ingredient": ing.name}

    def change_amount(self, query: str, amount: str) -> dict:
        """``amount``: "2 tbsp" (at the current servings), or double/half/more/less."""
        recipe = self._require_recipe()
        idx = self._require_ingredient(query)
        ing = recipe.ingredients[idx]
        factor = next((AMOUNT_WORDS[w] for w in _word_set(amount) if w in AMOUNT_WORDS), None)
        if factor is not None:
            if ing.quantity is None:
                note = "a little more than usual" if factor > 1 else "a little less than usual"
                new = ing.model_copy(update={"note": note, "original": None})
            else:
                new = ing.model_copy(update={"quantity": ing.quantity * factor, "original": None})
            more = factor > 1
        else:
            parsed = parse_ingredient_line(f"{amount} {ing.name}")
            if parsed.quantity is None:
                raise InvalidTransition(f"Say an amount for {ing.name}, like '2 tablespoons', 'double' or 'half'.")
            new = ing.model_copy(update={"quantity": parsed.quantity / self.scale_factor,
                                         "unit": parsed.unit or ing.unit, "original": None})
            more = ing.quantity is None or new.quantity > ing.quantity
        recipe.ingredients[idx] = new
        return {"ingredient": new.name, "was": ing.describe(self.scale_factor),
                "line": new.describe(self.scale_factor), "more": more}

    def change_step(self, step_number: Optional[int], text: str) -> dict:
        recipe = self._require_recipe()
        if step_number is None:
            if self.phase is not CookingPhase.COOKING:
                raise InvalidTransition("Tell me which step number to change.")
            step_number = self.step_index + 1
        if not 1 <= step_number <= len(recipe.steps):
            raise InvalidTransition(f"There are only {len(recipe.steps)} steps.")
        if not text.strip():
            raise InvalidTransition("Tell me what the step should say.")
        step = Step(instruction=text.strip())  # re-reads any time in the new wording
        recipe.steps[step_number - 1] = step
        self.timed_steps.discard(step_number)
        out: dict = {"step_number": step_number, "instruction": step.instruction}
        if self.phase is CookingPhase.COOKING and step_number == self.step_index + 1:
            if step.timer_minutes:
                self.timer_offer = {"minutes": step.timer_minutes, "step_number": step_number}
                out["suggested_timer_minutes"] = step.timer_minutes
            else:
                self.timer_offer = None
        return out

    def recipe_text(self) -> str:
        """The whole recipe as the cook currently has it, for answering
        questions about the dish."""
        recipe = self._require_recipe()
        lines = [f"{recipe.title}, for {self.target_servings}.",
                 "Ingredients: " + "; ".join(self.ingredient_lines()), "Steps:"]
        current = self.step_index if self.phase is CookingPhase.COOKING else None
        for n, step in enumerate(recipe.steps):
            mark = "[current step] " if n == current else ""
            lines.append(f"{n + 1}. {mark}{step.instruction}")
        return "\n".join(lines)

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
                "recipe_text": self.recipe_text(),
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
