"""Instant commands: everyday cooking phrases handled without the LLM.

``match_command`` decides whether an utterance is one of a small set of
commands ("next", "say that again", "set a rice timer for twelve minutes",
"what do I need", "cook my dal") and, if so, which tools to call. The agent
then runs those tools directly, so the reply is immediate and works with no
model and no internet.

Matching is deliberately strict: after light normalisation the *whole*
utterance must match a pattern, so "next time make it spicier" is left to the
LLM. A bare "yes"/"no" only counts when the agent says the app's own question
is still open (``awaiting``). While dictating only timer commands are matched,
and right after the first-run question "no", "yes", "next" and "that's all"
are left to the LLM, because there they answer the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .providers.base import ToolCall, ToolResult
from .recipe import format_minutes, ingredient_words, parse_duration_minutes
from .recipe_store import RecipeStore
from .state import Awaiting, CookingPhase, CookingSession
from .utterance import is_question

STEP_RESULT_TOOLS = {"next_step", "previous_step", "repeat_step"}
NO_REPLY = "No problem, take your time."
GOODBYE = "Goodbye, happy cooking!"


@dataclass
class CommandMatch:
    calls: list[ToolCall]
    reply: Optional[str] = None
    keep_open: bool = False  # "hold on": the app's question is still the open one


# ------------------------------------------------------------------ number words
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}


def words_to_digits(text: str) -> str:
    """'set a rice timer for twelve minutes' -> 'set a rice timer for 12 minutes'.

    Also 'twenty five' -> 25, 'one and a half minutes' -> '1.5 minutes',
    'a minute' -> '1 minute', 'half an hour' -> '30 minutes'."""
    t = re.sub(r"\bhalf an hour\b", "30 minutes", text)
    t = re.sub(r"\ban hour and a half\b", "90 minutes", t)
    t = re.sub(r"\bhalf a minute\b", "30 seconds", t)
    out: list[str] = []
    words = t.split()
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS:
            n = _TENS[w]
            if i + 1 < len(words) and words[i + 1] in _UNITS and 0 < _UNITS[words[i + 1]] < 10:
                n += _UNITS[words[i + 1]]
                i += 1
            out.append(str(n))
        elif w in _UNITS:
            out.append(str(_UNITS[w]))
        else:
            out.append(w)
        i += 1
    t = " ".join(out)
    t = re.sub(r"\b(\d+) and a half\b", r"\1.5", t)
    t = re.sub(r"\ban? (second|minute|hour|moment)\b", r"1 \1", t)
    return t


# ------------------------------------------------------------------ normalisation
_FILLERS = {"please", "um", "umm", "uh", "uhh", "hmm", "okay", "ok", "alright", "so", "well", "hey", "just"}
_PREFIX = re.compile(r"^(?:can you|could you|would you|will you|can we|i want you to|i need you to|"
                     r"i'd like you to|go ahead and)\s+")
_SUFFIX = re.compile(r"\s+(?:for me|thanks|thank you|thanks a lot)$")


def _normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep the dot in "1.5"
    t = re.sub(r"[^a-z0-9'. ]+", " ", t)
    t = words_to_digits(" ".join(t.split()))
    core = " ".join(w for w in t.split() if w not in _FILLERS)
    core = _SUFFIX.sub("", _PREFIX.sub("", core)).strip()
    return core or t  # a lone "okay" is an answer, not a filler


# ------------------------------------------------------------------ patterns
def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern)


# Ends with "what's next": the cook is done with the step. "What now" only alone or after a done
# word: "I burnt the onions, what now" is a problem, not a step forward.
_WHATS_NEXT = re.compile(r"(?:^|\s)(?:what's next|whats next|what is next|what do i do next|"
                         r"what's the next step|whats the next step)$"
                         r"|^(?:(?:ok|okay|right|alright|done|finished|all done|that's done|that's it|i'm done|"
                         r"im done|ready)\s+)*what (?:do i do )?now$")


_YES = _p(r"(?:(?:yes|yeah|yea|yep|yup|sure|ok|okay|alright|of course|go ahead|do it|please do|"
          r"sounds good)\s*)+")
_YES_TIMER = _p(r"(?:(?:yes|yeah|yep|yup|sure|ok|okay|alright|please)\s+)*(?:start it|start the timer)")
_WAIT = _p(r"(?:(?:not yet|wait|hold on|not now|later|give me 1 (?:minute|second|moment)|1 (?:minute|second|moment)|"
           r"in 1 minute)\s*)+")
_DECLINE = _p(r"(?:(?:no|nope|nah|no thanks|no thank you)\s*)+")
_NEXT = _p(r"next|next step|the next step|next 1|ready|i'm ready|im ready|i am ready|done|i'm done|im done|"
           r"i am done|continue|go on|carry on|move on|what's next|whats next|what next")
_BACK = _p(r"back|go back|go back 1(?: step)?|go back a step|previous|previous step|the previous step|"
           r"last step|the last step|step back|1 step back")
_REPEAT = _p(r"repeat|repeat that|repeat it|repeat the step|say that again|say it again|again|what was that|"
             r"what did you say|pardon|pardon me|come again|1 more time")
_DUR = r"\d+(?:\.\d+)? (?:seconds?|secs?|minutes?|mins?|hours?|hrs?)(?: and \d+(?:\.\d+)? (?:seconds?|secs?|minutes?|mins?))?"
_SET_TIMER = [
    _p(rf"(?:set|start|put on|make)(?: me| up)? (?:a |an |the )?(?:(?P<name>[a-z]+) )?timer (?:for|of) (?P<dur>{_DUR})"),
    _p(r"(?:set|start)(?: me)? (?:a |an )?(?P<dur>\d+(?:\.\d+)? (?:second|sec|minute|min|hour|hr)s?) "
       r"(?:(?P<name>[a-z]+) )?timer"),
    _p(rf"(?:a )?timer for (?P<dur>{_DUR})"),
]
_CANCEL_TIMER = _p(r"(?:stop|cancel|turn off|kill|end|clear) (?:the |my )?(?:(?P<name>[a-z]+) )?timer")
_LIST_TIMERS = _p(r"how long is left|how much time is left|how long left|how much time left|how much longer|"
                  r"time left|what timers|which timers|what timers are running|which timers are running|"
                  r"what timers do i have|list timers|list the timers|timers|check the timers?|"
                  r"how long (?:is left )?on the [a-z]+(?: timer)?")
_INGREDIENTS = _p(r"what do i need|what do we need|what will i need|ingredients|the ingredients|"
                  r"list (?:the )?ingredients|read (?:me )?(?:the )?ingredients|what are the ingredients|"
                  r"tell me the ingredients|what's in it|whats in it")
_SCALE = _p(r"(?:make|scale|cook|do|change|adjust)(?: it| this| the recipe)?(?: up| down)? for (?P<n>\d+)"
            r"(?: people| persons| servings| portions| of us)?")
_COOK = _p(r"(?:let's |lets |let us |i want to |i'd like to |id like to |i wanna )?(?:cook|make) "
           r"(?:my |our |the )?(?P<dish>.+?)(?: recipe)?")
_END = _p(r"bye|goodbye|good bye|bye bye|bye now|that's all bye|thats all bye|thanks bye|thank you bye|"
          r"stop the assistant|quit|exit")
# "that's all" only ends the session when nothing is going on; mid-recipe it
# usually means "that's all for this step" or answers a question.
_THATS_ALL = _p(r"that's all|thats all|that is all")
_GENERIC_TIMER_NAMES = {"new", "kitchen", "another", "cooking", "a", "an"}


def _call(tool: str, **args) -> CommandMatch:
    return CommandMatch([ToolCall(tool, args)])


# Answers to "Kaju Curry is built around cashews ... I'd go with almonds. Shall I, or keep the cashews?"
# (?P<what>...) must be "it", "them" or the asked-about ingredient (checked in _dish_choice).
_ANSWER_START = r"(?:(?:no|nah|nope|yes|yeah|yep|ok|okay|sure)\s+)?(?:just\s+)?"
_CLOSER = r"(?:\s+(?:then|please|today|this time|as (?:it is|they are|is)|that's fine|thats fine|it's fine|its fine))?"
_KEEP = _p(_ANSWER_START + r"(?:keep(?: (?:the )?(?P<what>[a-z' ]+?))?(?: in)?|leave (?:the )?(?P<what2>[a-z' ]+?) "
           r"(?:in|as (?:it is|they are|is))|(?:it's|its|that's|thats) fine(?: as it is)?)" + _CLOSER)
_LEAVE_OUT = _p(_ANSWER_START + r"(?:(?:leave|take) (?:the )?(?P<what>[a-z' ]+?) out|leave out (?:the )?"
                r"(?P<what2>[a-z' ]+?)|skip (?:the )?(?P<what3>[a-z' ]+?))(?: anyway)?" + _CLOSER)
# Words that carry no choice: "let's use the almonds please", "I'd go with that".
_CARRIER = {"use", "make", "go", "with", "let", "s", "do", "i", "ll", "d", "want", "have", "try", "instead",
            "just", "then", "option", "today", "please", "ok", "okay", "yes", "yeah", "sure", "is", "fine",
            "good", "we", "prefer", "rather", "go", "for", "can", "you", "one"}
_YOUR_PICK = _p(r"(?:(?:yes|yeah|yep|sure|ok|okay) )?"
                r"(?:(?:go with|do|make|use|let's do|lets do|let's go with|lets go with) )?"
                r"(?:that|that 1|it|your pick|your choice|the 1 you'd pick|what you'd pick|"  # "one" is "1" here
                r"whatever you'd pick|the 1 you picked|what you said|what you suggested)")
_CHOOSE = r"(?:(?:use|make|go with|do|let's do|lets do|let's go with|lets go with) )?"
_FIRST = _p(_CHOOSE + r"(?:the )?first(?: 1| option)?")
_SECOND = _p(_CHOOSE + r"(?:the )?second(?: 1| option)?")
_OTHER = _p(_CHOOSE + r"(?:the )?other(?: 1| option)?(?: instead)?")


def _about(m: re.Match, mine: set[str]) -> bool:
    """The answer is about the asked-about ingredient: "keep them", "keep the cashews",
    not "keep it nut free" or "leave the onions out" (Fable, A3)."""
    what = next((v for k, v in m.groupdict().items() if k.startswith("what") and v), None)
    return what is None or what in ("it", "them") or (bool(ingredient_words(what))
                                                       and set(ingredient_words(what)) <= mine)


def _dish_choice(t: str, choice: dict, session: CookingSession, question: bool) -> Optional[CommandMatch]:
    """The cook's answer to the app's built-around question, or None for the LLM.
    ``question``: it sounds like one, so only the exact answers are matched."""
    x, line = choice["ingredient"], choice.get("line") or choice["ingredient"]
    use, dish, pick = choice.get("use_instead"), choice.get("similar_dish"), choice["pick"]
    mine = set(ingredient_words(f"{x} {line}"))

    def do(kind: str) -> CommandMatch:
        if kind == "use_instead":
            # the same amount as what it replaces (the swap had dropped it: "butter, instead of ghee"; A6),
            # as stored, i.e. for the recipe's own servings, which is what the tool takes
            old = next((i for i in session.recipe.ingredients if i.name == line), None) if session.recipe else None
            amount = {"quantity": old.quantity, "unit": old.unit} if old and old.quantity is not None else {}
            return CommandMatch([ToolCall("substitute_ingredient", {"ingredient": line, "replacement": use} | amount)],
                                reply=f"Okay, {use} instead of {x}.")
        if kind == "similar_dish":
            servings = session.target_servings
            return CommandMatch([ToolCall("get_recipe", {"dish": dish} | ({"servings": servings} if servings else {}))])
        return CommandMatch([ToolCall("change_recipe", {"action": "remove_ingredient", "ingredient": line})],
                            reply=f"Okay, I've left out the {x}.")

    keep = CommandMatch([], reply=f"Okay, keeping the {x}.")
    if (m := _KEEP.fullmatch(t)) and _about(m, mine) or _DECLINE.fullmatch(t):
        return keep
    if (m := _LEAVE_OUT.fullmatch(t)) and _about(m, mine):
        return do("leave_out")
    if _YES.fullmatch(t) or _YOUR_PICK.fullmatch(t) or t == "do":  # "please do" is "do" once normalised
        return do(pick)
    listed = [k for k, v in (("use_instead", use), ("similar_dish", dish)) if v]  # in the order they were said
    if listed and _FIRST.fullmatch(t):
        return do(listed[0])
    if len(listed) == 2 and _SECOND.fullmatch(t):
        return do(listed[1])
    if len(listed) == 2 and _OTHER.fullmatch(t):
        return do(next(k for k in listed if k != pick))
    if question:
        return None
    # Naming one: every word left must be that option's, and at least one of them not shared with
    # the dish in hand or the other option. "I'm allergic to almonds" and "not almonds" (A1), or
    # "the curry" for a kaju curry vs a coconut curry (A2), go to the model.
    content = set(ingredient_words(t)) - _CARRIER
    if not content:
        return None
    name = set(ingredient_words(choice.get("name", "")))
    words = {k: set(ingredient_words(choice[k])) for k in listed}
    offered = set().union(*words.values())
    named = [k for k in listed
             if content <= words[k]
             and content & (words[k] - name - set().union(*(w for j, w in words.items() if j != k)))]
    if len(named) == 1:
        return do(named[0])
    # "just the kaju curry", "cashews", "I want the cashews today": the dish in hand, by a word no option has
    if not named and content <= name | mine and content & ((name | mine) - offered):
        return keep
    return None


def _minutes(dur: str) -> Optional[float]:
    """'5 minutes and 30 seconds' -> 5.5 (each part parsed, then added)."""
    parts = [parse_duration_minutes(p) for p in dur.split(" and ")]
    return sum(parts) if parts and all(parts) else None


def _timer_command(t: str, active_timers: list[str]) -> Optional[CommandMatch]:
    for pat in _SET_TIMER:
        mt = pat.fullmatch(t)
        if mt:
            minutes = _minutes(mt.group("dur"))
            if not minutes:
                return None
            name = mt.groupdict().get("name") or "kitchen"
            if name in _GENERIC_TIMER_NAMES:
                # an unnamed timer never silently replaces another one
                name, n = "kitchen", 2
                while name in active_timers:
                    name, n = f"kitchen {n}", n + 1
            return _call("set_timer", name=name, minutes=minutes)
    mt = _CANCEL_TIMER.fullmatch(t)
    if mt:
        name = mt.group("name")
        if name:
            return _call("cancel_timer", name=name)
        return _call("cancel_timer", name=active_timers[0]) if len(active_timers) == 1 else None
    if _LIST_TIMERS.fullmatch(t):
        return _call("list_timers")
    return None


def match_command(text: str, session: CookingSession, store: Optional[RecipeStore], active_timers: list[str],
                  awaiting: Optional[Awaiting], *, setup_open: bool = False) -> Optional[CommandMatch]:
    """The tools to run for ``text``, or None to let the LLM handle it."""
    t = _normalise(text)
    if not t:
        return None
    timer = _timer_command(t, active_timers)
    if timer is not None:
        return timer
    if session.draft is not None:
        return None  # while dictating, "next" and "that's all" belong to the recipe being dictated
    # Right after the first-run question ("any allergies? ... if not, just say none"),
    # yes/no/next/that's all are answers to it, not commands.
    answers_held = setup_open
    if awaiting is Awaiting.DISH_CHOICE and session.dish_choice and not answers_held:
        # before the question check: "do it", "what you suggested" start like questions (Fable, A7)
        choice = _dish_choice(t, session.dish_choice, session, question=is_question(text))
        if choice is not None:
            return choice
    if awaiting is not None and not answers_held:
        if awaiting is Awaiting.TIMER and (_YES.fullmatch(t) or _YES_TIMER.fullmatch(t)):
            return _call("start_step_timer")
        if awaiting is Awaiting.NEXT_STEP and _YES.fullmatch(t):
            return _call("next_step")
        if _WAIT.fullmatch(t):
            return CommandMatch([], reply=NO_REPLY, keep_open=True)
        if _DECLINE.fullmatch(t):
            # "no" to a timer offer declines it (the app moves on to "ready?");
            # "no" to "ready for the next step?" means not yet.
            return CommandMatch([], reply=NO_REPLY, keep_open=awaiting is Awaiting.NEXT_STEP)
    if session.recipe is not None:
        if session.phase is CookingPhase.COOKING and not answers_held and _WHATS_NEXT.search(t):
            return _call("next_step")  # "right, that's toasted, what's next" (qwen3 answered instead, 2/2)
        if _NEXT.fullmatch(t):
            return None if answers_held else _call("next_step")
        if _BACK.fullmatch(t):
            return _call("previous_step")
        if _REPEAT.fullmatch(t):
            return _call("repeat_step")
        if _INGREDIENTS.fullmatch(t):
            return _call("list_ingredients")
        mt = _SCALE.fullmatch(t)
        if mt:
            return _call("scale_servings", servings=int(mt.group("n")))
    mt = _COOK.fullmatch(t)
    if mt and store is not None and session.phase is not CookingPhase.COOKING:  # never swap mid-cook
        dish = mt.group("dish").strip()
        if dish and not dish.startswith(("it ", "a ", "an ", "some ")) and store.find(dish) is not None:
            return _call("get_recipe", dish=dish)
    if _END.fullmatch(t) and not answers_held:
        return _call("end_session")
    idle = session.recipe is None and not active_timers and not session.setup_pending
    if _THATS_ALL.fullmatch(t) and idle and not answers_held:
        return _call("end_session")
    return None


# ------------------------------------------------------------------ spoken replies
def _left(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return "1 second" if s == 1 else f"{s} seconds"
    m = int(round(s / 60))
    return "1 minute" if m == 1 else f"{m} minutes"


def _one(r: ToolResult) -> Optional[str]:
    res = r.result
    if "error" in res:
        return str(res["error"])
    if r.name == "set_timer":
        return f"Okay, {res['name']} timer set for {format_minutes(res['seconds'] / 60)}."
    if r.name == "start_step_timer":
        return f"Okay, timer started for {res.get('duration') or format_minutes(res['seconds'] / 60)}."
    if r.name == "cancel_timer":
        return f"{str(res['cancelled']).capitalize()} timer cancelled."
    if r.name == "list_timers":
        timers = res.get("timers") or []
        if not timers:
            return "No timers are running."
        return " ".join(f"{t['name'].capitalize()}: {_left(t['remaining_seconds'])} left." for t in timers)
    if r.name == "list_ingredients":
        return f"For {res['servings']}: " + ". ".join(res["ingredients"]) + "."
    if r.name == "scale_servings":
        return f"Okay, for {res['servings']}: " + ". ".join(res["ingredients"]) + "."
    if r.name == "get_recipe":
        if res.get("source") == "your recipe book":
            found = f"Found {res['title']} in your recipe book, for {res['servings']}."
            if usual := res.get("your_usual_changes"):  # never said on this path before (Fable, A11)
                found += f" You usually {'; '.join(usual)}; say if you'd like that."
            if not res.get("built_around"):  # otherwise the app's question about it comes next
                found += " Say ready to start."
        else:  # written just now, e.g. the similar dish the cook chose instead
            found = (f"Okay, here's {res['title']} for {res['servings']}: {res['ingredient_count']} ingredients "
                     f"and {res['step_count']} steps. Say ready to start.")
        warnings = res.get("warnings") or []
        return (f"Heads up: this recipe {'; '.join(warnings)}. " if warnings else "") + found
    if r.name == "end_session":
        return GOODBYE
    if r.name in STEP_RESULT_TOOLS:
        return res.get("message")  # "That was the last step."
    return None


def speak_result(results: list[ToolResult]) -> Optional[str]:
    """What to say after instant tool calls; None for a step that was read
    out, which the agent speaks itself word for word."""
    if any(r.name in STEP_RESULT_TOOLS and "instruction" in r.result for r in results):
        return None
    parts = [p for p in (_one(r) for r in results) if p]
    return " ".join(parts) or None
