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

from .recipe import Ingredient, Recipe, Step, format_minutes, ingredient_words, parse_ingredient_line

# Words that don't identify an ingredient on their own ("cumin seeds" is found by "cumin").
_GENERIC_WORDS = {"seed", "powder", "leaf", "oil", "sauce", "paste", "nut", "fresh", "ground", "whole",
                  "chopped", "dried", "green", "red", "white", "black", "large", "small"}
# "double the garlic", "less salt": multipliers for change_amount.
_SMALL_NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4"}
AMOUNT_WORDS = {"double": 2.0, "twice": 2.0, "triple": 3.0, "half": 0.5, "halve": 0.5,
                "more": 1.5, "extra": 1.5, "less": 0.5}


_word_set = ingredient_words


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
    # "Kaju Curry is built around cashews ... I'd go with almonds. Shall I, or keep the cashews?"
    DISH_CHOICE = "keep_or_change_the_main_ingredient"


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
    removed: list[str] = field(default_factory=list)  # ingredients the cook took out mid-recipe
    # The dish is built around something the cook usually leaves out or swaps (kept):
    # {"ingredient", "use_instead", "similar_dish", "pick", "question"}; "pick" is what a "yes" does.
    dish_choice: Optional[dict] = None

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
        self.removed = []
        self.dish_choice = None

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
            "instruction": self.step_text(self.step_index),
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
    def find_ingredient_candidates(self, query: str) -> list[int]:
        """Whole-word matches, plurals and spellings folded: "cumin" finds "cumin
        seeds", "chilli" finds "green chillies", "ice" doesn't find "rice". An
        exact name wins; otherwise every ingredient that fits is returned."""
        recipe = self._require_recipe()
        q = _word_set(query)
        if not q:
            return []
        names = [_word_set(i.name) for i in recipe.ingredients]
        exact = [i for i, n in enumerate(names) if n == q]
        if exact:
            return exact[:1]
        within = [i for i, n in enumerate(names) if set(q) <= set(n)]
        if within:
            return within
        # "salt and pepper": every ingredient sharing a telling word, so the cook is asked which
        telling = set(q) - _GENERIC_WORDS
        return [i for i, n in enumerate(names) if n and (set(n) <= set(q) or telling & set(n))]

    def find_ingredient_words(self, query: str) -> Optional[int]:
        """The one ingredient the words name, or None (none, or more than one)."""
        found = self.find_ingredient_candidates(query)
        return found[0] if len(found) == 1 else None

    def _require_ingredient(self, query: str) -> int:
        recipe = self._require_recipe()
        found = self.find_ingredient_candidates(query)
        if not found:
            names = ", ".join(i.name for i in recipe.ingredients)
            raise InvalidTransition(f"I can't find {query} in this recipe. Ingredients are: {names}.")
        if len(found) > 1:
            raise InvalidTransition(f"Which one: {', '.join(recipe.ingredients[i].name for i in found)}?")
        return found[0]

    def _mention_keys(self, name: str) -> set[str]:
        recipe = self._require_recipe()
        words = _word_set(name)
        keys = {w for w in words if w not in _GENERIC_WORDS} or set(words)
        # "olive oil" is "the oil" in a step when no other ingredient is an oil
        if words and sum(1 for i in recipe.ingredients if _word_set(i.name)[-1:] == words[-1:]) <= 1:
            keys.add(words[-1])
        return keys

    def steps_mentioning(self, name: str) -> list[dict]:
        """Steps from the current one on that mention the ingredient."""
        recipe = self._require_recipe()
        keys = self._mention_keys(name)
        first = self.step_index if self.phase is CookingPhase.COOKING else 0
        return [{"step_number": n, "instruction": step.instruction}
                for n, step in enumerate(recipe.steps[first:], start=first + 1)
                if keys & set(_word_set(step.instruction))]

    def step_text(self, index: int) -> str:
        """A step as it is read out: its own words, plus a note for any
        ingredient the cook took out or swapped since ("Leave out the mint
        leaves.", "Use butter instead of ghee."). Steps are read word for
        word, so this is what keeps them true after a change."""
        recipe = self._require_recipe()
        text = recipe.steps[index].instruction
        words = set(_word_set(text))
        notes = [f"Leave out the {name}." for name in self.removed if self._mention_keys(name) & words]
        notes += [f"Use {sub['replacement']} instead of {sub['original']}." for sub in self.substitutions
                  if self._mention_keys(sub["original"]) & words]
        return " ".join([text, *notes])

    def remove_ingredient(self, query: str) -> dict:
        recipe = self._require_recipe()
        idx = self._require_ingredient(query)
        if len(recipe.ingredients) == 1:
            raise InvalidTransition("That's the only ingredient left, so I can't take it out.")
        removed = recipe.ingredients.pop(idx)
        self.removed.append(removed.name)
        return {"removed": removed.describe(self.scale_factor), "ingredient": removed.name}

    def add_ingredient(self, line: str) -> dict:
        """``line`` is said at the current servings ("1 cup peas" for 4)."""
        recipe = self._require_recipe()
        ing = parse_ingredient_line(line)
        if ing.quantity is not None:
            ing = ing.model_copy(update={"quantity": ing.quantity / self.scale_factor})
        ing = ing.model_copy(update={"original": None})
        recipe.ingredients.append(ing)
        added = set(_word_set(ing.name))
        self.removed = [r for r in self.removed if not (self._mention_keys(r) & added)]  # put back
        return {"added": ing.describe(self.scale_factor), "ingredient": ing.name}

    def change_amount(self, query: str, amount: str) -> dict:
        """``amount`` at the current servings: "2 tbsp", "one and a half cups",
        "half a cup less", "2 more cloves" (added to the amount), or a bare
        double / twice as much / half / more / less. ``more`` in the result is
        whether the amount went up (None if the unit changed)."""
        recipe = self._require_recipe()
        idx = self._require_ingredient(query)
        ing = recipe.ingredients[idx]
        text = f" {amount.lower().strip()} "
        text = re.sub(r"(?<![a-z])(one|two|three|four|\d+) and a half(?![a-z])",
                      lambda m: f"{_SMALL_NUMBERS.get(m.group(1), m.group(1))} 1/2", text)
        text = re.sub(r"(?<![a-z])half(?: an?)?(?= (?!as |the |of |it |that |this )[a-z])", "1/2", text)
        text = re.sub(r"(?<![a-z])a quarter(?: of)?(?: an?)?(?![a-z])", "1/4", text)
        words = set(re.findall(r"[a-z]+", text))
        direction = 1 if words & {"more", "extra"} else -1 if words & {"less", "fewer"} else 0
        number_text = re.sub(r"(?<![a-z])(?:more|less|extra|fewer|than|usual|of it)(?![a-z])", " ", text).strip()
        parsed = parse_ingredient_line(f"{number_text} {ing.name}") if number_text else None
        unit_changed = False
        if parsed is not None and parsed.quantity is not None:
            if parsed.quantity == 0:
                raise InvalidTransition(f"To take the {ing.name} out completely, just say leave it out.")
            said = parsed.quantity / self.scale_factor  # stored at the recipe's own servings
            same_unit = parsed.unit is None or ing.unit is None or _word_set(parsed.unit) == _word_set(ing.unit)
            unit_changed = not same_unit
            if direction and ing.quantity is not None and same_unit:
                quantity = ing.quantity + direction * said
                if quantity <= 0:
                    raise InvalidTransition(f"That would leave no {ing.name}. To take it out, say leave it out.")
            else:
                quantity = said
            new = ing.model_copy(update={"quantity": quantity, "unit": parsed.unit or ing.unit, "original": None})
            more = None if unit_changed else (ing.quantity is None or quantity > ing.quantity)
        else:
            factor = next((AMOUNT_WORDS[w] for w in ("double", "twice", "triple", "half", "halve") if w in words),
                          None) or next((AMOUNT_WORDS[w] for w in ("more", "extra", "less") if w in words), None)
            if factor is None:
                raise InvalidTransition(f"Say an amount for {ing.name}, like '2 tablespoons', 'double' or 'half'.")
            if ing.quantity is None:
                note = "a little more than usual" if factor > 1 else "a little less than usual"
                new = ing.model_copy(update={"note": note, "original": None})
            else:
                new = ing.model_copy(update={"quantity": ing.quantity * factor, "original": None})
            more = factor > 1
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
            lines.append(f"{n + 1}. {mark}{self.step_text(n)}")
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
