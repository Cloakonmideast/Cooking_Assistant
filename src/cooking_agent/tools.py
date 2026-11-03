"""LLM tools and their dispatcher.

Each tool is a pydantic argument model + an async handler. The JSON schema
sent to the LLM is generated from the argument model, and the LLM's arguments
are validated against the same model before the handler runs, so the schema
the model sees and the checks we apply cannot drift apart.

Handlers never raise to the caller: domain errors (no recipe loaded, unknown
timer, ...) and validation errors come back as ``{"error": "..."}`` so the LLM
can explain the problem to the user in its own words.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from .providers.base import LLMProvider, ToolCall, ToolResult, ToolSpec
from .recipe import Ingredient, Recipe
from .state import CookingSession, InvalidTransition
from .timers import TimerError, TimerManager

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ arg models
class NoArgs(BaseModel):
    pass


class GetRecipeArgs(BaseModel):
    dish: str = Field(min_length=1, description="The dish to cook, e.g. 'vegetable biryani'.")
    servings: Optional[int] = Field(default=None, ge=1, le=50, description="Number of servings, if the user said.")
    constraints: Optional[str] = Field(
        default=None, description="Dietary needs or preferences, e.g. 'vegan', 'no nuts', 'under 30 minutes'."
    )


class SetTimerArgs(BaseModel):
    name: str = Field(min_length=1, description="Short label for the timer, e.g. 'rice', 'pasta', 'oven'.")
    minutes: float = Field(gt=0, le=24 * 60, description="Duration in minutes (may be fractional, 0.5 = 30 seconds).")


class TimerNameArgs(BaseModel):
    name: str = Field(min_length=1, description="Label of the timer to cancel.")


class ScaleArgs(BaseModel):
    servings: int = Field(ge=1, le=50, description="New number of servings.")


class SubstituteArgs(BaseModel):
    ingredient: str = Field(min_length=1, description="Ingredient in the current recipe to replace.")
    replacement: str = Field(min_length=1, description="What to use instead.")
    quantity: Optional[float] = Field(default=None, ge=0, description="Amount of the replacement for the recipe's original servings.")
    unit: Optional[str] = Field(default=None, description="Unit for the replacement amount.")
    note: Optional[str] = Field(default=None, description="Short usage note, e.g. 'add with the onions'.")


# ------------------------------------------------------------------ registry
Handler = Callable[[Any], Awaitable[dict]]


@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Handler

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


RECIPE_PROMPT = (
    "Write a home-cook recipe for: {dish}.\n"
    "Servings: {servings}.\n"
    "{constraints}"
    "It will be read aloud one step at a time, so keep each step to one or two short "
    "sentences with a single action, use plain words (no symbols or abbreviations other "
    "than common units), and set timer_minutes on steps that involve waiting."
)


class ToolBox:
    """Owns the tools and executes calls against the cooking session and timers."""

    def __init__(self, session: CookingSession, timers: TimerManager, llm: LLMProvider):
        self.session = session
        self.timers = timers
        self.llm = llm
        self.end_requested = False
        self._tools: dict[str, Tool] = {}
        for tool in self._build():
            self._tools[tool.name] = tool

    # -- public ------------------------------------------------------------
    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, {"error": f"Unknown tool {call.name}."}, call.id)
        try:
            args = tool.args_model.model_validate(call.args or {})
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'args'}: {err['msg']}" for err in e.errors())
            return ToolResult(call.name, {"error": f"Invalid arguments: {problems}"}, call.id)
        try:
            result = await tool.handler(args)
        except (InvalidTransition, TimerError) as e:
            result = {"error": str(e)}
        except Exception as e:  # provider failure inside get_recipe, etc.
            log.exception("tool %s failed", call.name)
            result = {"error": f"{call.name} failed: {type(e).__name__}"}
        log.info("tool %s(%s) -> %s", call.name, call.args, result)
        return ToolResult(call.name, result, call.id)

    # -- handlers ----------------------------------------------------------
    def _build(self) -> list[Tool]:
        s = self.session
        return [
            Tool("get_recipe", "Find a recipe for a dish and load it. Replaces any current recipe.",
                 GetRecipeArgs, self._get_recipe),
            Tool("next_step", "Advance to the next cooking step (starts cooking if not started) and return it.",
                 NoArgs, self._wrap(lambda _: s.next_step())),
            Tool("previous_step", "Go back one step and return it.",
                 NoArgs, self._wrap(lambda _: s.previous_step())),
            Tool("repeat_step", "Return the current step again.",
                 NoArgs, self._wrap(lambda _: s.current_step())),
            Tool("list_ingredients", "List the ingredients of the current recipe, scaled to the current servings.",
                 NoArgs, self._wrap(lambda _: {"servings": s.target_servings, "ingredients": s.ingredient_lines()})),
            Tool("set_timer", "Start a named countdown timer. Several timers can run at once. "
                 "Reusing a name restarts that timer.",
                 SetTimerArgs, self._wrap(lambda a: self.timers.set(a.name, a.minutes * 60))),
            Tool("cancel_timer", "Cancel a running timer by name.",
                 TimerNameArgs, self._wrap(lambda a: self.timers.cancel(a.name))),
            Tool("list_timers", "List running timers and the seconds left on each.",
                 NoArgs, self._wrap(lambda _: {"timers": self.timers.list()})),
            Tool("scale_servings", "Rescale the current recipe's ingredient quantities to a number of servings.",
                 ScaleArgs, self._wrap(lambda a: s.scale_servings(a.servings))),
            Tool("substitute_ingredient", "Replace an ingredient in the current recipe with a substitute you choose.",
                 SubstituteArgs, self._wrap(self._substitute)),
            Tool("end_session", "End the conversation. Only when the user clearly wants to quit the assistant "
                 "(not when they want to stop a timer or stop talking).",
                 NoArgs, self._end_session),
        ]

    @staticmethod
    def _wrap(fn: Callable[[Any], dict]) -> Handler:
        async def handler(args: Any) -> dict:
            return fn(args)

        return handler

    async def _get_recipe(self, a: GetRecipeArgs) -> dict:
        prompt = RECIPE_PROMPT.format(
            dish=a.dish,
            servings=a.servings or "4",
            constraints=f"Constraints: {a.constraints}.\n" if a.constraints else "",
        )
        recipe = await self.llm.generate_structured(prompt, Recipe)
        self.session.load_recipe(recipe)
        return {
            "title": recipe.title,
            "servings": recipe.servings,
            "ingredient_count": len(recipe.ingredients),
            "step_count": len(recipe.steps),
            "ingredients": self.session.ingredient_lines(),
        }

    def _substitute(self, a: SubstituteArgs) -> dict:
        repl = Ingredient(name=a.replacement, quantity=a.quantity, unit=a.unit, note=a.note)
        return self.session.substitute(a.ingredient, repl)

    async def _end_session(self, _: NoArgs) -> dict:
        self.end_requested = True
        return {"ok": True}
