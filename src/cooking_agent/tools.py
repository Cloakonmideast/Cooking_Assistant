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
from typing import Any, Awaitable, Callable, Literal, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from .providers.base import LLMProvider, ToolCall, ToolResult, ToolSpec
from .recipe import Ingredient, Recipe, Step, format_minutes, parse_ingredient_line
from .preferences import CATEGORIES, PreferenceStore, describe_for_recipe
from .recipe_store import RecipeStore
from .state import Awaiting, CookingPhase, CookingSession, InvalidTransition, RecipeDraft
from .timers import TimerError, TimerManager

READY_QUESTION = "Are you ready for the next step?"
STEP_TOOLS = {"next_step", "previous_step", "repeat_step"}

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


class StepTimerArgs(BaseModel):
    name: Optional[str] = Field(default=None, description="Short label, e.g. 'rice'. Defaults to 'step N'.")


class SearchArgs(BaseModel):
    query: Optional[str] = Field(default=None, description="Words to look for in recipe titles; omit to list all.")


class StartDraftArgs(BaseModel):
    title: str = Field(min_length=1, description="Name of the recipe, e.g. \"Mum's dal\".")
    servings: Optional[int] = Field(default=None, ge=1, le=50, description="How many people it serves, if said.")


class DraftAddArgs(BaseModel):
    ingredients: list[str] = Field(default_factory=list, description=(
        "Ingredient lines in the order said, written with digits like '2 cups basmati rice, rinsed' or "
        "'salt to taste'."))
    steps: list[str] = Field(default_factory=list, description="Steps in the order said, one action each, "
                             "in the cook's words (lightly cleaned up). Keep any times, e.g. 'for 10 minutes'.")
    servings: Optional[int] = Field(default=None, ge=1, le=50, description="Set or change how many it serves.")


class DraftRemoveArgs(BaseModel):
    ingredient_number: Optional[int] = Field(default=None, ge=1, description="1-based ingredient to remove.")
    step_number: Optional[int] = Field(default=None, ge=1, description="1-based step to remove.")


class RememberArgs(BaseModel):
    category: Literal[CATEGORIES] = Field(description=(  # type: ignore[valid-type]
        "diet (vegetarian, vegan, ...), allergy, dislike (ingredient or food they avoid), like (flavours, "
        "cuisines, spice level), pantry (ingredients they usually have), other."))
    value: str = Field(min_length=1, description="Short phrase, e.g. 'vegetarian', 'peanuts', 'mushrooms', "
                       "'spicy food', 'paneer'.")


class ForgetArgs(BaseModel):
    value: str = Field(min_length=1, description="The preference to forget, e.g. 'mushrooms'.")


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
    "{profile}"
    "It will be read aloud one step at a time, so keep each step to one or two short "
    "sentences with a single action, use plain words (no symbols or abbreviations other "
    "than common units), and set timer_minutes on steps that involve waiting."
)


class ToolBox:
    """Owns the tools and executes calls against the cooking session and timers."""

    def __init__(self, session: CookingSession, timers: TimerManager, llm: LLMProvider,
                 store: Optional[RecipeStore] = None, prefs: Optional[PreferenceStore] = None):
        self.session = session
        self.timers = timers
        self.llm = llm
        self.store = store
        self.prefs = prefs
        self._request_id: Optional[int] = None  # get_recipe request behind the loaded recipe
        self._cook_recorded = True  # nothing loaded yet, nothing to record
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

    def follow_up(self, results: list[ToolResult]) -> Optional[str]:
        """The question to ask once the reply to this turn has been spoken, and
        what a bare "yes" will mean next turn (``session.awaiting``).

        After a step: offer its timer if it states a time, otherwise ask if the
        cook is ready for the next step. After the timer offer is answered
        (started, or declined / moved on): ask if they are ready. Nothing after
        the last step, or while the cook is doing something else.
        """
        s = self.session
        ok = [r for r in results if "error" not in r.result]
        step = next((r.result for r in reversed(ok) if r.name in STEP_TOOLS), None)
        if step is not None:
            if "instruction" not in step:  # next_step past the end: finished
                s.awaiting = None
                return None
            if s.timer_offer:
                s.awaiting = Awaiting.TIMER
                return f"Do you want me to start a timer for {format_minutes(s.timer_offer['minutes'])}?"
            return self._ask_ready()
        timer_answered = any(r.name in ("start_step_timer", "set_timer") for r in ok)
        if s.awaiting is Awaiting.TIMER and (timer_answered or not results):
            if s.timer_offer and timer_answered:  # cook chose their own timer instead
                s.timed_steps.add(s.timer_offer["step_number"])
                s.timer_offer = None
            return self._ask_ready()
        if any(r.name == "start_step_timer" for r in ok):
            return self._ask_ready()
        return None

    def _ask_ready(self) -> Optional[str]:
        s = self.session
        if s.phase is not CookingPhase.COOKING or s.on_last_step:
            s.awaiting = None
            return None
        s.awaiting = Awaiting.NEXT_STEP
        return READY_QUESTION

    # -- handlers ----------------------------------------------------------
    def _build(self) -> list[Tool]:
        s = self.session
        return [
            Tool("get_recipe", "Load a recipe for a dish: from the cook's saved recipe book if it has one, "
                 "otherwise a newly written one. Replaces any current recipe. Pass servings if the cook said "
                 "how many people.", GetRecipeArgs, self._get_recipe),
            Tool("list_saved_recipes", "List or search the recipes saved in the cook's recipe book.",
                 SearchArgs, self._list_saved),
            Tool("save_recipe", "Save the currently loaded recipe to the recipe book (e.g. a generated one "
                 "the cook liked).", NoArgs, self._save_current),
            Tool("start_recipe_dictation", "Start recording a recipe the cook will dictate, to save in the "
                 "recipe book.", StartDraftArgs, self._wrap(self._start_draft)),
            Tool("add_to_dictated_recipe", "Add ingredients and/or steps the cook just dictated, in order.",
                 DraftAddArgs, self._wrap(self._draft_add)),
            Tool("remove_from_dictated_recipe", "Remove a wrongly dictated ingredient or step by number.",
                 DraftRemoveArgs, self._wrap(self._draft_remove)),
            Tool("read_dictated_recipe", "Return everything dictated so far, to read back to the cook.",
                 NoArgs, self._wrap(lambda _: self._draft().summary())),
            Tool("save_dictated_recipe", "Save the dictated recipe to the recipe book. Only after the cook "
                 "has said they are finished and want it saved.", NoArgs, self._save_draft),
            Tool("discard_dictated_recipe", "Throw away the recipe being dictated.",
                 NoArgs, self._wrap(self._discard_draft)),
            Tool("start_step_timer", "Start a timer for the time the current step states (after the cook "
                 "says yes to the timer offer).", StepTimerArgs, self._wrap(self._start_step_timer)),
            Tool("next_step", "Advance to the next cooking step (starts cooking if not started) and return it.",
                 NoArgs, self._wrap(self._next_step)),
            Tool("remember_preference", "Remember a lasting preference of the cook for future recipes "
                 "(e.g. 'I'm vegetarian', 'I'm allergic to peanuts', 'I hate mushrooms', 'I always have paneer'). "
                 "Not for one-off requests like 'no onions today'.", RememberArgs, self._wrap(self._remember)),
            Tool("forget_preference", "Forget a preference the cook no longer has.",
                 ForgetArgs, self._wrap(self._forget)),
            Tool("get_preferences", "What is known about the cook: stated preferences and learned habits.",
                 NoArgs, self._wrap(lambda _: self._profile() or {"note": "Nothing known yet."})),
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
        request_id = self.prefs.record_request(a.dish, a.constraints, a.servings) if self.prefs else None
        servings, servings_learned = a.servings, False
        if servings is None and self.prefs and (usual := self.prefs.default_servings()):
            servings, servings_learned = usual, True
        saved = self._find_saved(a)
        if saved is not None:
            self._loaded(saved.recipe, "recipe book", request_id)
            if servings and servings != saved.recipe.servings:
                self.session.scale_servings(servings)
            out = self._recipe_summary(saved.recipe) | {"source": "your recipe book"}
        else:
            profile = self._profile()
            prompt = RECIPE_PROMPT.format(
                dish=a.dish,
                servings=servings or 4,
                constraints=f"Constraints: {a.constraints}.\n" if a.constraints else "",
                profile=describe_for_recipe(profile),
            )
            recipe = await self.llm.generate_structured(prompt, Recipe)
            self._loaded(recipe, "generated", request_id)
            if servings and servings != recipe.servings:  # the model ignored the requested servings
                self.session.scale_servings(servings)
            out = self._recipe_summary(recipe) | {"source": "newly written, not from the recipe book"}
            applied = self._applied(profile)
            if applied:
                out["tailored_to"] = applied
        if servings_learned:
            out["servings_note"] = f"made for {servings}, the number you usually cook for"
        return out

    def _loaded(self, recipe: Recipe, source: str, request_id: Optional[int]) -> None:
        self.session.load_recipe(recipe)
        self.session.recipe_source = source
        self._request_id, self._cook_recorded = request_id, False

    # -- preferences -------------------------------------------------------
    def _profile(self) -> dict:
        return self.prefs.profile() if self.prefs else {}

    @staticmethod
    def _applied(profile: dict) -> list[str]:
        """Short notes on what shaped a generated recipe, for the assistant to mention."""
        out = [f"no {a} (allergy)" for a in profile.get("allergy", [])]
        out += profile.get("diet", [])
        out += [f"no {d}" for d in profile.get("dislike", [])]
        learned = profile.get("learned_from_history", {})
        out += [f"{d} (you usually ask for it)" for d in learned.get("usually_asks_for", [])]
        return out

    def _remember(self, a: RememberArgs) -> dict:
        if self.prefs is None:
            raise InvalidTransition("I can't store preferences right now.")
        new = self.prefs.remember(a.category, a.value)
        return {"remembered": a.value, "category": a.category, "already_known": not new}

    def _forget(self, a: ForgetArgs) -> dict:
        if self.prefs is None:
            raise InvalidTransition("I can't store preferences right now.")
        removed = self.prefs.forget(a.value)
        if not removed:
            known = ", ".join(r["value"] for r in self.prefs.stated()) or "nothing yet"
            raise InvalidTransition(f"I don't have a preference about {a.value}. I know: {known}.")
        return {"forgot": [r["value"] for r in removed]}

    def _next_step(self, _: NoArgs) -> dict:
        out = self.session.next_step()
        if not self._cook_recorded and out.get("step_number") == 1 and self.session.recipe:
            # They are actually cooking it: that is what "usually cooks with" learns from.
            self._cook_recorded = True
            if self.prefs:
                r = self.session.recipe
                self.prefs.record_cooked(self._request_id, r.title, [i.name for i in r.ingredients])
        return out

    def _recipe_summary(self, recipe: Recipe) -> dict:
        return {
            "title": recipe.title,
            "servings": self.session.target_servings,
            "original_servings": recipe.servings,
            "ingredient_count": len(recipe.ingredients),
            "step_count": len(recipe.steps),
            "ingredients": self.session.ingredient_lines(),
        }

    def _find_saved(self, a: GetRecipeArgs):
        if self.store is None:
            return None
        saved = self.store.find(a.dish)
        if saved is None:
            return None
        # A dietary constraint must be visible in the saved title ("vegan dal"),
        # otherwise write a recipe that is guaranteed to respect it.
        if a.constraints and not self.store.search(f"{saved.title} {a.constraints}"):
            return None
        return saved

    async def _list_saved(self, a: SearchArgs) -> dict:
        if self.store is None:
            return {"error": "The recipe book is not available."}
        found = self.store.search(a.query) if a.query else self.store.all()
        return {"count": len(found), "recipes": [
            {"title": r.title, "servings": r.recipe.servings} for r in found[:20]]}

    async def _save_current(self, _: NoArgs) -> dict:
        if self.store is None:
            return {"error": "The recipe book is not available."}
        if self.session.recipe is None:
            return {"error": "There's no recipe loaded to save."}
        recipe = self.session.recipe
        if self.session.recipe_source == "recipe book":
            return {"error": f"{recipe.title} is already in the recipe book."}
        rid = self.store.add(recipe.model_copy(deep=True), source="generated")
        self.session.recipe_source = "recipe book"
        return {"saved": recipe.title, "id": rid}

    # -- step timer --------------------------------------------------------
    def _start_step_timer(self, a: StepTimerArgs) -> dict:
        offer = self.session.take_timer_offer()
        name = a.name or f"step {offer['step_number']}"
        out = self.timers.set(name, offer["minutes"] * 60)
        return out | {"duration": format_minutes(offer["minutes"])}

    # -- dictation ---------------------------------------------------------
    def _draft(self) -> RecipeDraft:
        if self.session.draft is None:
            raise InvalidTransition("No recipe is being dictated. Say you want to save your own recipe first.")
        return self.session.draft

    def _start_draft(self, a: StartDraftArgs) -> dict:
        if self.store is None:
            raise InvalidTransition("The recipe book is not available, so I can't save recipes.")
        replaced = self.session.draft.title if self.session.draft else None
        self.session.draft = RecipeDraft(title=a.title.strip(), servings=a.servings)
        out = {"started": a.title, "servings": a.servings}
        if replaced:
            out["discarded_unsaved_draft"] = replaced
        return out

    def _draft_add(self, a: DraftAddArgs) -> dict:
        d = self._draft()
        if a.servings:
            d.servings = a.servings
        added_ings = [parse_ingredient_line(line) for line in a.ingredients if line.strip()]
        d.ingredients += added_ings
        d.steps += [Step(instruction=t.strip()) for t in a.steps if t.strip()]
        return {"added_ingredients": [i.describe() for i in added_ings], "added_steps": len(a.steps),
                "total_ingredients": len(d.ingredients), "total_steps": len(d.steps), "servings": d.servings}

    def _draft_remove(self, a: DraftRemoveArgs) -> dict:
        d = self._draft()
        out: dict = {}
        if a.ingredient_number is not None:
            if a.ingredient_number > len(d.ingredients):
                raise InvalidTransition(f"There are only {len(d.ingredients)} ingredients so far.")
            out["removed_ingredient"] = d.ingredients.pop(a.ingredient_number - 1).describe()
        if a.step_number is not None:
            if a.step_number > len(d.steps):
                raise InvalidTransition(f"There are only {len(d.steps)} steps so far.")
            out["removed_step"] = d.steps.pop(a.step_number - 1).instruction
        if not out:
            raise InvalidTransition("Tell me which ingredient or step number to remove.")
        return out

    async def _save_draft(self, _: NoArgs) -> dict:
        try:
            recipe = self._draft().to_recipe()
        except InvalidTransition as e:
            return {"error": str(e)}
        assert self.store is not None
        rid = self.store.add(recipe, source="dictated")
        self.session.draft = None
        return {"saved": recipe.title, "id": rid, "servings": recipe.servings,
                "ingredient_count": len(recipe.ingredients), "step_count": len(recipe.steps)}

    def _discard_draft(self, _: NoArgs) -> dict:
        title = self._draft().title
        self.session.draft = None
        return {"discarded": title}

    def _substitute(self, a: SubstituteArgs) -> dict:
        repl = Ingredient(name=a.replacement, quantity=a.quantity, unit=a.unit, note=a.note)
        return self.session.substitute(a.ingredient, repl)

    async def _end_session(self, _: NoArgs) -> dict:
        self.end_requested = True
        return {"ok": True}
