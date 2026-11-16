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
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional, Type

from pydantic import BaseModel, Field, ValidationError, model_validator

from .providers.base import LLMProvider, ToolCall, ToolResult, ToolSpec
from .recipe import Ingredient, Recipe, Step, format_minutes, ingredient_words, parse_ingredient_line
from .dietary import DIETS, find_conflicts, requested_words
from .preferences import (CATEGORIES, PreferenceStore, describe_for_recipe, diets_in, habit_key,
                          prefer_eggetarian, usual_changes)
from .dishes import built_around, main_words, name_glossary
from .recipe_store import RecipeStore
from .state import Awaiting, CookingPhase, CookingSession, InvalidTransition, RecipeDraft
from .timers import TimerError, TimerManager
from .utterance import asks_permission, is_question

READY_QUESTION = "Are you ready for the next step?"
STEP_TOOLS = {"next_step", "previous_step", "repeat_step"}

OPTIONS_PROMPT = (
    "A cook asked for {title}, a dish built around {ingredient}, but they usually leave {ingredient} out. "
    "Suggest one ingredient they could use in place of {ingredient} in this dish, and one similar dish "
    "that doesn't need {ingredient}. Then pick the one you'd recommend. A few words each, no explanation.\n"
    "{profile}"
)


class BetterOptions(BaseModel):
    """Alternatives when a dish is built around something the cook usually leaves out."""

    # required, with no "empty if none": given a default of "", qwen3 returned "" for both (3 of 3)
    use_instead: str = Field(description="One ingredient to use in its place in this dish, e.g. 'almonds'.")
    similar_dish: str = Field(description="A similar dish that doesn't need it, e.g. 'paneer butter masala'.")
    pick: Literal["use_instead", "similar_dish"]

# Tools offered to the LLM per phase (ToolBox.specs_for).
ALWAYS_TOOLS = {"get_recipe", "list_saved_recipes", "start_recipe_dictation", "set_timer", "cancel_timer",
                "list_timers", "remember_preference", "forget_preference", "get_preferences", "end_session"}
SETUP_TOOLS = {"finish_setup"}
DICTATION_TOOLS = {"add_to_dictated_recipe", "remove_from_dictated_recipe", "read_dictated_recipe",
                   "save_dictated_recipe", "discard_dictated_recipe"}
RECIPE_TOOLS = STEP_TOOLS | {"start_step_timer", "list_ingredients", "scale_servings", "substitute_ingredient",
                             "change_recipe", "save_recipe"}

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
        "allergy (a food they react to), condition (health condition or intolerance that affects food, e.g. "
        "lactose intolerance, celiac disease, diabetes, high blood pressure), diet (vegetarian, vegan, "
        "eggetarian = vegetarian who eats eggs, jain, ...), "
        "dislike (ingredient or food they avoid), like (flavours, cuisines, spice level), pantry (ingredients "
        "they usually have), other."))
    value: str = Field(min_length=1, description="Short phrase, e.g. 'peanuts', 'lactose intolerance', "
                       "'vegetarian', 'mushrooms', 'spicy food', 'paneer'.")


class FinishSetupArgs(BaseModel):
    skipped: bool = Field(default=False, description="True if the cook chose to skip the setup question.")


class ForgetArgs(BaseModel):
    value: str = Field(min_length=1, description="The preference to forget, e.g. 'mushrooms'.")


class SubstituteArgs(BaseModel):
    ingredient: str = Field(min_length=1, description="Ingredient in the current recipe to replace.")
    replacement: str = Field(min_length=1, description="What to use instead.")
    quantity: Optional[float] = Field(default=None, ge=0, description="Amount of the replacement for the recipe's original servings.")
    unit: Optional[str] = Field(default=None, description="Unit for the replacement amount.")
    note: Optional[str] = Field(default=None, description="Short usage note, e.g. 'add with the onions'.")

    @model_validator(mode="before")
    @classmethod
    def _amount_with_unit(cls, data):
        # Local models often send quantity "2 tbsp": split it rather than fail the swap.
        q = data.get("quantity") if isinstance(data, dict) else None
        if isinstance(q, str):
            try:
                data = {**data, "quantity": float(q)}
            except ValueError:
                parsed = parse_ingredient_line(f"{q} x")
                data = {**data, "quantity": parsed.quantity}
                if parsed.quantity is not None and parsed.unit and not data.get("unit"):
                    data["unit"] = parsed.unit
        return data


class ChangeRecipeArgs(BaseModel):
    action: Literal["add_ingredient", "remove_ingredient", "change_amount", "change_step"] = Field(description=(
        "remove_ingredient: leave something out. add_ingredient: put something extra in. change_amount: more or "
        "less of an ingredient. change_step: new wording for a step."))
    ingredient: Optional[str] = Field(default=None, description="The ingredient, e.g. 'cashews', 'green peas'.")
    amount: Optional[str] = Field(default=None, description=(
        "For add_ingredient and change_amount: an amount for the servings being cooked now, e.g. '1 cup', "
        "'2 tbsp', or for change_amount 'double', 'half', 'more', 'less'."))
    step_number: Optional[int] = Field(default=None, ge=1, description="For change_step; defaults to the current step.")
    text: Optional[str] = Field(default=None, description="For change_step: the whole new step, one or two short "
                                "sentences, keeping any time, e.g. 'Fry the onions for 5 minutes.'")
    reason: Optional[str] = Field(default=None, description="Why, in the cook's words, e.g. 'less spicy'.")


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
    "{glossary}"
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
        self._request = ""  # what the cook asked for when the recipe was loaded (dish + constraints)
        self._cook_recorded = True  # nothing loaded yet, nothing to record
        self.end_requested = False
        self.utterance = ""  # what the cook said this turn (set by the agent)
        self._tools: dict[str, Tool] = {}
        for tool in self._build():
            self._tools[tool.name] = tool

    # -- public ------------------------------------------------------------
    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def specs_for(self, session: CookingSession) -> list[ToolSpec]:
        """Only the tools that fit the moment: fewer to read means a faster,
        more accurate choice from a small local model. Registry order is kept
        so the prompt prefix stays the same from turn to turn."""
        offered = set(ALWAYS_TOOLS)
        if session.setup_pending:
            offered |= SETUP_TOOLS
        if session.draft is not None:
            offered |= DICTATION_TOOLS
        if session.recipe is not None:
            offered |= RECIPE_TOOLS
        return [t.spec() for name, t in self._tools.items() if name in offered]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        args = call.args or {}
        valid = ("add_ingredient", "remove_ingredient", "change_amount", "change_step")
        if call.name == "change_recipe" and args.get("replacement") and args.get("action") not in valid:
            # qwen3 sometimes sends a swap as change_recipe with an invented action
            # ("substitute_ingredient", "change_ingredient"): it's a swap
            rest = {k: v for k, v in args.items() if k not in ("action", "reason", "amount", "step_number", "text")}
            result = await self.dispatch(ToolCall("substitute_ingredient", rest, call.id))
            return ToolResult(call.name, result.result, call.id)
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
        if s.dish_choice is not None:
            if any(r.name == "get_recipe" for r in ok):
                s.awaiting = Awaiting.DISH_CHOICE
                return s.dish_choice["question"]
            s.dish_choice = None  # answered, or the cook moved on to something else
            if s.awaiting is Awaiting.DISH_CHOICE:
                s.awaiting = None
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
        if s.awaiting is Awaiting.TIMER and not results and s.timer_offer and is_question(self.utterance):
            # "why do we soak it?" isn't an answer to the timer offer: ask it again
            return f"Do you want me to start a timer for {format_minutes(s.timer_offer['minutes'])}?"
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
            Tool("forget_preference", "Forget a preference the cook no longer has, or a change they usually make "
                 "(e.g. 'stop swapping ghee for butter' -> 'butter').",
                 ForgetArgs, self._wrap(self._forget)),
            Tool("finish_setup", "Finish the first-time setup after saving what the cook said about allergies, "
                 "conditions and diet (or after they said none, or asked to skip).",
                 FinishSetupArgs, self._wrap(self._finish_setup)),
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
            Tool("change_recipe", "Change the loaded recipe when the cook asks to leave something out, add "
                 "something, use more or less of something, or do a step differently. Keeps the current step. "
                 "For swapping one ingredient for another use substitute_ingredient instead.",
                 ChangeRecipeArgs, self._wrap(self._change_recipe)),
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
        request = " ".join(filter(None, [a.dish, a.constraints]))  # what the cook asked for, for diet checks
        saved = self._find_saved(a)
        if saved is not None:
            self._loaded(saved.recipe, "recipe book", request_id, request)
            if servings and servings != saved.recipe.servings:
                self.session.scale_servings(servings)
            out = self._recipe_summary(saved.recipe) | {"source": "your recipe book"}
            warnings = self._warnings(saved.recipe, request) + self._learned_diet_warnings(saved.recipe, request)
            if warnings:
                out["warnings"] = warnings  # their own recipe: warn, never change it silently
            profile = self._profile()
            around = self._built_around([a.dish, saved.recipe.title], profile)
            if usual := self._usual_changes_for_saved(self._without(profile, around)):
                out["your_usual_changes"] = usual  # offered, made only if the cook says so
            if kept := await self._kept(around, profile):
                out["built_around"] = kept
        else:
            full_profile = self._profile()
            # "leave out cashews" isn't for a kaju curry: neither the writer nor the code applies it
            around = self._built_around([a.dish], full_profile)
            profile = self._without(full_profile, around)
            prompt = RECIPE_PROMPT.format(
                dish=a.dish,
                glossary=name_glossary(a.dish),
                servings=servings or 4,
                constraints=f"Constraints: {a.constraints}.\n" if a.constraints else "",
                profile=describe_for_recipe(profile),
            )
            recipe, warnings = await self._write_recipe(prompt, request)
            warnings += self._learned_diet_warnings(recipe, request)
            self._loaded(recipe, "generated", request_id, request)
            if servings and servings != recipe.servings:  # the model ignored the requested servings
                self.session.scale_servings(servings)
            habits = self._apply_usual_changes(profile, request)
            out = self._recipe_summary(self.session.recipe) | {"source": "newly written, not from the recipe book"}
            if warnings:
                out["warnings"] = warnings
            applied = self._applied(profile) + habits
            if applied:
                out["tailored_to"] = applied
            if kept := await self._kept(around, full_profile):
                out["built_around"] = kept
        if servings_learned:
            out["servings_note"] = f"made for {servings}, the number you usually cook for"
        return out

    async def _write_recipe(self, prompt: str, request: str = "") -> tuple[Recipe, list[str]]:
        """Have the LLM write a recipe; if it includes something the cook's
        allergies, conditions or diet rule out, send it back once to fix."""
        recipe = await self.llm.generate_structured(prompt, Recipe)
        conflicts = self._conflicts([i.name for i in recipe.ingredients], request)
        if not conflicts:
            return recipe, []
        log.warning("generated recipe conflicts with the cook's profile: %s", [c.message() for c in conflicts])
        retry = (f"{prompt}\nYour previous version included "
                 f"{'; '.join(c.message() for c in conflicts)}. This cook must not have these. "
                 "Rewrite the whole recipe without them, using safe substitutes.")
        recipe = await self.llm.generate_structured(retry, Recipe)
        conflicts = self._conflicts([i.name for i in recipe.ingredients], request)
        return recipe, [self._warning_text(c) for c in conflicts]

    def _diets(self, profile: dict, request: str) -> tuple[list[str], set[str]]:
        """Diets to check (stated ones plus any named in the request, e.g.
        "vegan, for four") and the forbidden ingredients the cook asked for by
        name ("chicken biryani, just this once"): for a diet, the request wins
        for those ingredients only."""
        wanted = prefer_eggetarian(diets_in(" ".join(profile.get("diet", []))) | diets_in(request))
        diets = sorted(d for d in wanted if d in DIETS)
        return diets, requested_words(request, diets)

    def _conflicts(self, names: list[str], request: str = ""):
        p = self._profile()
        diets, allowed = self._diets(p, request)
        if not (p.get("allergy") or p.get("condition") or diets):
            return []
        return find_conflicts(names, allergies=p.get("allergy", []), conditions=p.get("condition", []),
                              diets=diets, diet_allowed=allowed)

    @staticmethod
    def _warning_text(c) -> str:
        return f"contains {c.message()}"

    def _learned_diet_warnings(self, recipe: Recipe, request: str) -> list[str]:
        """A diet the cook usually asks for but never stated: a heads-up only,
        never a rewrite. Skipped when a stated or requested diet applies, and
        for ingredients the request names ("chicken biryani")."""
        p = self._profile()
        learned = [d for d in p.get("learned_from_history", {}).get("usually_asks_for", []) if d in DIETS]
        hard, _ = self._diets(p, request)
        if not learned or hard:
            return []
        soft = sorted(prefer_eggetarian(set(learned)))
        allowed = requested_words(request, soft)
        return [f"has {c.ingredient}, and you usually cook {c.reason.removesuffix(' diet')}"
                for c in find_conflicts([i.name for i in recipe.ingredients], diets=soft, diet_allowed=allowed)]

    @staticmethod
    def _built_around(dishes: list[str], profile: dict) -> list[dict]:
        """The cook's usual leave-outs and swaps of an ingredient the dish is
        named after ("leave out cashews" for a kaju curry)."""
        usual = usual_changes(profile)
        named = lambda x: any(d and built_around(d, x) for d in dishes)  # noqa: E731
        out = [{"ingredient": s["from"], "your_usual_change": f"swap {s['from']} for {s['to']}",
                "usual_swap": s["to"]} for s in usual["swaps"] if named(s["from"])]
        out += [{"ingredient": x, "your_usual_change": f"leave out {x}"} for x in usual["leaves_out"] if named(x)]
        return out

    @staticmethod
    def _without(profile: dict, around: list[dict]) -> dict:
        """The profile minus the usual changes in ``around``."""
        if not around:
            return profile
        skip = {n["ingredient"] for n in around}
        learned = dict(profile.get("learned_from_history", {}))
        learned["usually_swaps"] = [s for s in learned.get("usually_swaps", []) if s["from"] not in skip]
        learned["usually_leaves_out"] = [x for x in learned.get("usually_leaves_out", []) if x not in skip]
        return profile | {"learned_from_history": {k: v for k, v in learned.items() if v}}

    async def _kept(self, around: list[dict], profile: dict) -> list[dict]:
        """The usual changes not made because the dish is built around the
        ingredient (and the recipe has it). For the first one, the app asks the
        cook itself after the reply (``session.dish_choice``): the model was
        told to and didn't (qwen3 said "I kept the cashews" with no options in
        3 of 3, then acted on the answer wrongly)."""
        r = self.session.recipe
        kept = [n for n in around
                if any(main_words(n["ingredient"]) <= set(ingredient_words(i.name)) for i in r.ingredients)]
        if kept:
            self.session.dish_choice = await self._dish_choice(kept[0], profile)
        return [n | {"note": "kept on purpose; the app asked the cook whether to change it"} for n in kept]

    async def _dish_choice(self, note: dict, profile: dict) -> dict:
        # "Kaju Curry (Cashew Curry) - 2 Servings", as qwen3 titled it, is said as "Kaju Curry"
        name = re.sub(r"\s*\([^)]*\)|\s*[-–:]\s*(?:for\s+)?\d+\s+servings?\b.*$", "", self.session.recipe.title,
                      flags=re.IGNORECASE).strip() or self.session.recipe.title
        x = note["ingredient"]
        them = "them" if x.endswith("s") else "it"
        if "usual_swap" in note:  # the usual swap is the option
            to = note["usual_swap"]
            return {"name": name, "ingredient": x, "use_instead": to, "similar_dish": None, "pick": "use_instead",
                    "question": f"{name} is built around {x}, so I kept {them} in, though you usually use "
                                f"{to} instead. Shall I use {to} anyway, or keep the {x}?"}
        kept = f"{name} is built around {x}, so I kept {them} in, though you usually leave {them} out."
        use, dish, pick = await self._better_options(x, name, profile)
        if use and dish:
            mine = f"I'd go with {use}." if pick == "use_instead" else f"I'd make {dish} instead."
            offer = f"You could use {use} instead, or make {dish}. {mine} Shall I, or keep the {x}?"
        elif use:
            offer = f"I'd go with {use} instead. Shall I, or keep the {x}?"
        elif dish:
            offer = f"I'd make {dish} instead. Shall I, or keep the {x}?"
        else:
            offer = f"Shall I leave {them} out anyway, or keep {them}?"
        return {"name": name, "ingredient": x, "use_instead": use, "similar_dish": dish, "pick": pick,
                "question": f"{kept} {offer}"}

    async def _better_options(self, x: str, title: str, profile: dict
                              ) -> tuple[Optional[str], Optional[str], str]:
        """(something to use instead, a similar dish without it, which to pick);
        anything the cook can't have, or that is the same thing again, is dropped."""
        prompt = OPTIONS_PROMPT.format(title=title, ingredient=x, profile=describe_for_recipe(profile))
        try:
            o = await self.llm.generate_structured(prompt, BetterOptions)
        except Exception as e:  # the recipe is still fine: ask the plain question instead
            log.warning("no alternatives for %s in %s: %s", x, title, e)
            return None, None, "leave_out"
        use, dish = o.use_instead.strip().lower() or None, o.similar_dish.strip().lower() or None
        has = lambda name: any(main_words(name) <= set(ingredient_words(i.name))  # noqa: E731
                               for i in self.session.recipe.ingredients)
        if use and (self._conflicts([use]) or main_words(x) <= set(ingredient_words(use)) or has(use)):
            use = None  # can't have it, the same thing again, or already in the recipe (two coconut milk lines)
        if dish and (self._conflicts([dish]) or built_around(dish, x)):
            dish = None
        options = {"use_instead": use, "similar_dish": dish}
        pick = o.pick if options[o.pick] else next((k for k, v in options.items() if v), "leave_out")
        return use, dish, pick

    def _usual_changes_for_saved(self, profile: dict) -> list[str]:
        """The cook's usual changes that apply to the loaded recipe-book recipe."""
        usual = usual_changes(profile)
        s = self.session
        has = lambda name: s.find_ingredient_words(name) is not None  # noqa: E731
        return ([f"swap {x['from']} for {x['to']}" for x in usual["swaps"] if has(x["from"])]
                + [f"leave out {x}" for x in usual["leaves_out"] if has(x)]
                + [f"use more {x}" for x in usual["more"] if has(x)]
                + [f"use less {x}" for x in usual["less"] if has(x)])

    def _record(self, kind: str, original: str, replacement: Optional[str] = None) -> None:
        if self.prefs is not None:
            self.prefs.record_adjustment(kind, original, replacement, load_id=self._request_id)

    def _warnings(self, recipe: Recipe, request: str = "") -> list[str]:
        return [self._warning_text(c) for c in self._conflicts([i.name for i in recipe.ingredients], request)]

    def _finish_setup(self, a: FinishSetupArgs) -> dict:
        if self.prefs is None:
            raise InvalidTransition("I can't store preferences right now.")
        self.prefs.mark_setup_done(skipped=a.skipped)
        self.session.setup_pending = False
        return {"setup_done": True, "skipped": a.skipped, "profile": self.prefs.profile()}

    def _loaded(self, recipe: Recipe, source: str, request_id: Optional[int], request: str = "") -> None:
        self.session.load_recipe(recipe)
        self.session.recipe_source = source
        self._request_id, self._cook_recorded = request_id, False
        self._request = request

    # -- preferences -------------------------------------------------------
    def _profile(self) -> dict:
        return self.prefs.profile() if self.prefs else {}

    @staticmethod
    def _applied(profile: dict) -> list[str]:
        """Short notes on what shaped a generated recipe, for the assistant to mention."""
        out = [f"no {a} (allergy)" for a in profile.get("allergy", [])]
        out += [f"suitable for {c}" for c in profile.get("condition", [])]
        out += profile.get("diet", [])
        out += [f"no {d}" for d in profile.get("dislike", [])]
        learned = profile.get("learned_from_history", {})
        out += [f"{d} (you usually ask for it)" for d in learned.get("usually_asks_for", [])]
        return out

    def _apply_usual_changes(self, profile: dict, request: str) -> list[str]:
        """Make the cook's usual changes to a newly written recipe in code (the
        writer often ignores them) and return what was actually done, for the
        assistant to mention. Skips an ingredient the request names ("cumin
        rice") or that matches more than one line; never recorded as new
        evidence for the habit."""
        s = self.session
        usual = usual_changes(profile)
        named = set(ingredient_words(request))
        done: list[str] = []

        def target(name: str) -> Optional[int]:
            # the same ingredient only ("onion" -> "red onions"), never a near one
            # ("coconut milk" habit -> "milk", "butter" -> "peanut butter")
            if set(ingredient_words(name)) & named:
                return None
            key = habit_key(name)
            same = [i for i, ing in enumerate(s.recipe.ingredients) if habit_key(ing.name) == key]
            return same[0] if len(same) == 1 else None

        def attempt(action: Callable[[], None], note: str) -> None:
            try:
                action()
                done.append(note)
            except InvalidTransition as e:  # e.g. the last ingredient: leave the recipe as written
                log.info("usual change skipped (%s): %s", note, e)

        for x in usual["swaps"]:
            idx = target(x["from"])
            if idx is not None:
                old = s.recipe.ingredients[idx]
                attempt(lambda old=old, x=x: s.substitute(old.name, Ingredient(
                    name=x["to"], quantity=old.quantity, unit=old.unit)), f"{x['to']} instead of {x['from']} (your usual swap)")
        for x in usual["leaves_out"]:
            if (idx := target(x)) is not None:
                name = s.recipe.ingredients[idx].name
                attempt(lambda name=name: s.remove_ingredient(name), f"no {x} (you usually leave it out)")
        for kind, items in (("more", usual["more"]), ("less", usual["less"])):
            for x in items:
                if (idx := target(x)) is not None:
                    name = s.recipe.ingredients[idx].name
                    attempt(lambda name=name, kind=kind: s.change_amount(name, kind),
                            f"{kind} {x} (you usually like {kind})")
        for x in usual["adds"]:  # left to the writer (no amount or step to add it at); claimed if it's there
            if s.find_ingredient_words(x) is not None:
                done.append(f"with {x} (you usually add it)")
        return done

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
        out = self.session.substitute(a.ingredient, repl)
        self._record("swap", out["original"], a.replacement)
        if conflicts := self._conflicts([a.replacement], self._request):
            out["warnings"] = [self._warning_text(c) for c in conflicts]
        return out

    def _change_recipe(self, a: ChangeRecipeArgs) -> dict:
        s = self.session
        if asks_permission(self.utterance):
            # "could I leave out the garlic?" asks whether, not to: answer, offer, don't change or learn
            return {"not_changed": True, "note": "The cook only asked whether they could. Answer the question, "
                    "then offer to make the change; make it only if they say yes."}
        if a.action == "change_step":
            return s.change_step(a.step_number, a.text or "")
        if not a.ingredient:
            raise InvalidTransition("Tell me which ingredient to change.")
        if a.action == "remove_ingredient":
            out = s.remove_ingredient(a.ingredient)
            self._record("remove", out["ingredient"])
            return out
        if a.action == "add_ingredient":
            out = s.add_ingredient(f"{a.amount} {a.ingredient}" if a.amount else a.ingredient)
            self._record("add", out["ingredient"])
        else:
            if not a.amount:
                raise InvalidTransition(f"How much {a.ingredient}? For example '2 tablespoons', 'double' or 'half'.")
            out = s.change_amount(a.ingredient, a.amount)
            more = out.pop("more")
            if more is not None:  # a unit change ("2 tbsp" for "1 cup") says nothing about taste
                self._record("more" if more else "less", out["ingredient"])
            if more is False:
                return out
        if conflicts := self._conflicts([out["ingredient"]], self._request):
            out["warnings"] = [self._warning_text(c) for c in conflicts]
        return out

    async def _end_session(self, _: NoArgs) -> dict:
        self.end_requested = True
        return {"ok": True}
