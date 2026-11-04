"""Recipe schema.

The LLM is asked for a recipe with this schema as its structured-output
schema, and the reply is validated with pydantic, so a malformed reply fails
loudly instead of being string-sliced out of free text.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Ingredient(BaseModel):
    name: str = Field(min_length=1, description="Ingredient name, e.g. 'basmati rice'.")
    quantity: Optional[float] = Field(
        default=None, ge=0, description="Numeric amount, or null for 'to taste'."
    )
    unit: Optional[str] = Field(default=None, description="Unit such as 'g', 'cup', 'tsp'.")
    note: Optional[str] = Field(default=None, description="Prep note, e.g. 'finely chopped'.")
    original: Optional[str] = Field(
        default=None, description="Leave empty. The line exactly as published, for imported recipes.")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("ingredient name must not be blank")
        return v

    def describe(self, factor: float = 1.0) -> str:
        """Human/TTS-friendly line, e.g. '1 and 1/2 cups basmati rice, rinsed'.

        An imported ingredient at its original amount is given exactly as
        published; only a scaled amount is rewritten from the parsed parts.
        """
        if self.original and (factor == 1 or self.quantity is None):
            return self.original
        parts: list[str] = []
        scaled = None if self.quantity is None else self.quantity * factor
        if scaled is not None:
            parts.append(format_quantity(scaled))
        if self.unit:
            parts.append(_unit_for(self.unit, scaled))
        parts.append(self.name)
        text = " ".join(parts)
        if self.note:
            text += f", {self.note}"
        return text


_PLURAL_UNITS = {
    "cup": "cups", "tablespoon": "tablespoons", "teaspoon": "teaspoons", "clove": "cloves", "can": "cans",
    "tin": "tins", "packet": "packets", "stick": "sticks", "slice": "slices", "bunch": "bunches",
    "sprig": "sprigs", "handful": "handfuls", "piece": "pieces", "pinch": "pinches", "dash": "dashes",
    "pound": "pounds", "ounce": "ounces", "gram": "grams", "kilogram": "kilograms", "litre": "litres",
    "liter": "liters", "quart": "quarts", "pint": "pints",
}
_SINGULAR_UNITS = {v: k for k, v in _PLURAL_UNITS.items()}


def _unit_for(unit: str, quantity: Optional[float]) -> str:
    """'cup' for 1 or less, 'cups' above 1, so scaled lines read naturally."""
    if quantity is None:
        return unit
    key = unit.lower()
    if quantity > 1:
        return _PLURAL_UNITS.get(key, unit)
    return _SINGULAR_UNITS.get(key, unit)


class Step(BaseModel):
    instruction: str = Field(min_length=1, description="One action, spoken aloud to the cook.")
    timer_minutes: Optional[float] = Field(
        default=None, gt=0, description="If the step involves waiting, how many minutes."
    )

    @model_validator(mode="after")
    def _timer_from_text(self) -> "Step":
        # Recipes from the web or the cook's own files rarely mark timers, and
        # the LLM sometimes forgets: take the time the instruction itself states.
        if self.timer_minutes is None:
            self.timer_minutes = parse_duration_minutes(self.instruction)
        return self


class Recipe(BaseModel):
    title: str = Field(min_length=1)
    servings: int = Field(ge=1, le=50)
    ingredients: list[Ingredient] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)


def format_quantity(q: float) -> str:
    """Format a quantity for speech: 1.5 -> '1 and 1/2', 0.25 -> '1/4', 200.0 -> '200'."""
    if q <= 0:
        return "0"
    if q >= 20:
        return str(round(q))
    frac = Fraction(q).limit_denominator(8)
    whole, rem = divmod(frac.numerator, frac.denominator)
    if rem == 0:
        return str(whole)
    tail = f"{rem}/{frac.denominator}"
    return f"{whole} and {tail}" if whole else tail


# ---------------------------------------------------------------- numbers in text
_UNICODE_FRACTIONS = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8",
                      "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅕": "1/5", "⅙": "1/6"}
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
}
_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
_WORD = "|".join(sorted((re.escape(w) for w in _WORD_NUMBERS), key=len, reverse=True))
_AMOUNT = rf"(?:{_NUM}|\b(?:{_WORD})\b)"


def normalize_fractions(text: str) -> str:
    """'1½' -> '1 1/2', '⅓' -> '1/3', and the fraction slash to '/'."""
    for ch, frac in _UNICODE_FRACTIONS.items():
        text = re.sub(rf"(\d){ch}", rf"\1 {frac}", text).replace(ch, frac)
    return text.replace("⁄", "/")


def parse_number(token: str) -> Optional[float]:
    token = token.strip().lower()
    if token in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[token])
    try:
        return float(sum(Fraction(part) for part in token.split()))
    except (ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------- durations
_UNIT_MINUTES = {"second": 1 / 60, "sec": 1 / 60, "minute": 1, "min": 1, "hour": 60, "hr": 60}
_DURATION = re.compile(
    rf"(?P<a>{_AMOUNT})(?:\s*(?:-|–|—|to|or)\s*(?P<b>{_AMOUNT}))?\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_HALF_HOUR = re.compile(r"\bhalf an hour\b", re.IGNORECASE)
_AND_A_HALF = re.compile(r"^\s*and a half\b", re.IGNORECASE)
_THEN_MORE = re.compile(rf"^\s*(?:and\s+)?(?P<a>{_NUM})\s*(?P<unit>seconds?|secs?|minutes?|mins?)\b", re.IGNORECASE)


def _unit_minutes(unit: str) -> float:
    u = unit.lower().rstrip("s")
    return _UNIT_MINUTES.get(u, 1)


def parse_duration_minutes(text: str) -> Optional[float]:
    """First duration stated in a step, in minutes.

    'Simmer for 10-12 minutes' -> 10 (a range gives its lower bound, so the
    cook checks early), 'bake 1 hour 15 minutes' -> 75, 'an hour and a half'
    -> 90, 'half an hour' -> 30, 'Heat the oil' -> None.
    """
    text = normalize_fractions(text)
    half = _HALF_HOUR.search(text)
    m = _DURATION.search(text)
    if half and (m is None or half.start() < m.start()):
        return 30.0
    if m is None:
        return None
    amount = parse_number(m.group("a"))
    if amount is None or amount <= 0:
        return None
    unit = _unit_minutes(m.group("unit"))
    minutes = amount * unit
    rest = text[m.end():]
    if _AND_A_HALF.match(rest):
        minutes += 0.5 * unit
    elif unit == 60 and (more := _THEN_MORE.match(rest)):  # '1 hour 15 minutes'
        minutes += (parse_number(more.group("a")) or 0) * _unit_minutes(more.group("unit"))
    return round(minutes, 3)


def format_minutes(minutes: float) -> str:
    """Speakable duration: 0.5 -> '30 seconds', 12 -> '12 minutes', 90 -> '1 hour and 30 minutes'."""
    secs = round(minutes * 60)
    if secs < 60:
        return f"{secs} second{'s' if secs != 1 else ''}"
    hours, rem = divmod(secs, 3600)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if rem:
        q = format_quantity(rem / 60)
        parts.append(f"{q} minute{'' if q == '1' else 's'}")
    return " and ".join(parts)


# ---------------------------------------------------------------- ingredient lines
_UNITS = [
    "tablespoons", "tablespoon", "tbsp", "tbs", "tbl", "teaspoons", "teaspoon", "tsp", "cups", "cup",
    "grams", "gram", "g", "kilograms", "kilogram", "kg", "milligrams", "mg", "millilitres", "milliliters",
    "millilitre", "milliliter", "ml", "litres", "liters", "litre", "liter", "l", "ounces", "ounce", "oz",
    "pounds", "pound", "lbs", "lb", "pinches", "pinch", "dashes", "dash", "cloves", "clove", "cans", "can",
    "tins", "tin", "packets", "packet", "sticks", "stick", "slices", "slice", "bunches", "bunch",
    "sprigs", "sprig", "handfuls", "handful", "pieces", "piece", "quarts", "quart", "pints", "pint",
]
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_INGREDIENT = re.compile(
    rf"^\s*(?P<q>{_NUM}|\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\b)"
    rf"(?:\s*(?:-|–|to)\s*{_NUM})?"  # range: keep the lower bound
    rf"\s*(?P<paren>\([^)]*\))?"
    rf"\s*(?:(?P<unit>{_UNIT_RE})\.?(?![a-z]))?"
    rf"\s*(?:of\s+)?(?P<rest>.*)$",
    re.IGNORECASE,
)


def parse_ingredient_line(line: str) -> Ingredient:
    """'1½ cups basmati rice, rinsed' -> Ingredient(1.5, 'cups', 'basmati rice', 'rinsed').

    Lines without a leading amount ('Salt to taste') keep quantity None, so
    they are read out as-is and never scaled.
    """
    text = " ".join(normalize_fractions(line).replace(" ", " ").split()).strip(" -•*")
    quantity: Optional[float] = None
    unit: Optional[str] = None
    notes: list[str] = []
    m = _INGREDIENT.match(text)
    if m and (m.group("unit") or m.group("q")[0].isdigit()):  # 'a pinch' yes, 'a lemon' only via 'a'+unit
        quantity = parse_number(m.group("q"))
        unit = m.group("unit")
        if m.group("paren"):
            notes.append(m.group("paren").strip("() "))
        text = m.group("rest").strip()
    paren = re.search(r"\(([^)]*)\)\s*$", text)
    if paren:
        notes.append(paren.group(1).strip())
        text = text[: paren.start()].strip()
    if "," in text:
        text, note = text.split(",", 1)
        notes.append(note.strip())
    name = text.strip(" ,") or line.strip()
    return Ingredient(name=name, quantity=quantity, unit=unit.lower() if unit else None,
                      note=", ".join(n for n in notes if n) or None, original=" ".join(line.split()))
