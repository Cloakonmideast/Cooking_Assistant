"""Recipe schema.

The LLM is asked for a recipe with this schema as its structured-output
schema, and the reply is validated with pydantic, so a malformed reply fails
loudly instead of being string-sliced out of free text.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Ingredient(BaseModel):
    name: str = Field(min_length=1, description="Ingredient name, e.g. 'basmati rice'.")
    quantity: Optional[float] = Field(
        default=None, ge=0, description="Numeric amount, or null for 'to taste'."
    )
    unit: Optional[str] = Field(default=None, description="Unit such as 'g', 'cup', 'tsp'.")
    note: Optional[str] = Field(default=None, description="Prep note, e.g. 'finely chopped'.")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("ingredient name must not be blank")
        return v

    def describe(self, factor: float = 1.0) -> str:
        """Human/TTS-friendly line, e.g. '1 and 1/2 cups basmati rice, rinsed'."""
        parts: list[str] = []
        if self.quantity is not None:
            parts.append(format_quantity(self.quantity * factor))
        if self.unit:
            parts.append(self.unit)
        parts.append(self.name)
        text = " ".join(parts)
        if self.note:
            text += f", {self.note}"
        return text


class Step(BaseModel):
    instruction: str = Field(min_length=1, description="One action, spoken aloud to the cook.")
    timer_minutes: Optional[float] = Field(
        default=None, gt=0, description="If the step involves waiting, how many minutes."
    )


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
