"""Check ingredients against the cook's allergies and health conditions.

The LLM is told about allergies and conditions when it writes a recipe, but a
language model can still slip. This module is the deterministic safety net:
it looks for ingredients that conflict with what the cook said and returns
plain-English warnings. ``get_recipe`` uses it to send a generated recipe back
for a fix, and to warn about recipes from the recipe book (which are never
silently changed).

Matching is by whole words on ingredient names, with families for the common
allergens ("tree nuts" -> almond, cashew, ...) and exemptions for look-alikes
("coconut milk" is not dairy, "rice flour" is not wheat). It is a helper, not
a medical guarantee; the README says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

_DAIRY = ["milk", "cream", "butter", "buttermilk", "cheese", "paneer", "yogurt", "yoghurt", "curd", "dahi",
          "khoa", "khoya", "malai", "whey", "ice cream", "condensed milk", "evaporated milk", "milk powder",
          "custard", "ricotta", "mozzarella", "parmesan", "cheddar", "mascarpone", "kefir", "lassi", "raita"]
_PLANT_DAIRY = ["coconut", "almond", "oat", "soy", "soya", "rice", "cashew", "peanut", "vegan", "lactose-free",
                "lactose free", "dairy-free", "dairy free", "plant", "cocoa"]
_GLUTEN = ["wheat", "flour", "maida", "atta", "semolina", "sooji", "suji", "rava", "bread", "breadcrumbs",
           "pasta", "spaghetti", "noodles", "barley", "rye", "couscous", "bulgur", "seitan", "soy sauce",
           "beer", "malt", "naan", "roti", "chapati", "tortilla", "pita", "cracker", "biscuit", "cake", "farro",
           "spelt", "vermicelli", "sevai"]
_GLUTEN_FREE = ["rice", "gram", "chickpea", "corn", "maize", "almond", "coconut", "besan", "tapioca", "potato",
                "buckwheat", "millet", "jowar", "bajra", "ragi", "sorghum", "gluten-free", "gluten free",
                "tamari", "rice noodles", "glass noodles", "quinoa"]

# allergy word -> ingredient words it covers
ALLERGEN_FAMILIES: dict[str, list[str]] = {
    "nut": ["almond", "cashew", "walnut", "pistachio", "pecan", "hazelnut", "macadamia", "brazil nut",
            "pine nut", "chironji", "praline", "marzipan", "nut"],
    "tree nut": ["almond", "cashew", "walnut", "pistachio", "pecan", "hazelnut", "macadamia", "brazil nut",
                 "pine nut", "chironji", "praline", "marzipan"],
    "peanut": ["peanut", "groundnut", "monkey nut", "peanut butter"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish", "scallop", "mussel", "clam", "oyster", "squid"],
    "seafood": ["fish", "shrimp", "prawn", "crab", "lobster", "scallop", "mussel", "clam", "oyster", "squid",
                "anchovy", "salmon", "tuna", "cod", "sardine", "mackerel", "fish sauce"],
    "fish": ["fish", "anchovy", "salmon", "tuna", "cod", "sardine", "mackerel", "fish sauce", "pomfret", "hilsa"],
    "egg": ["egg", "mayonnaise", "mayo", "meringue", "albumen"],
    "dairy": _DAIRY + ["ghee"],
    "milk": _DAIRY + ["ghee"],
    "lactose": _DAIRY,
    "gluten": _GLUTEN,
    "wheat": _GLUTEN,
    "soy": ["soy", "soya", "tofu", "edamame", "tamari", "miso", "tempeh", "soy sauce"],
    "sesame": ["sesame", "tahini", "til", "gingelly"],
    "mustard": ["mustard"],
    "celery": ["celery", "celeriac"],
}

# look-alikes that don't count for an allergy family ('coconut milk' for a milk allergy)
ALLERGEN_EXEMPT: dict[str, list[str]] = {
    "dairy": _PLANT_DAIRY, "milk": _PLANT_DAIRY, "lactose": _PLANT_DAIRY,
    "gluten": _GLUTEN_FREE, "wheat": _GLUTEN_FREE,
    "egg": ["eggplant", "egg-free", "egg free", "eggless"],
}

# condition keyword -> (ingredient words to flag, look-alikes that are fine, guidance for the recipe writer)
CONDITIONS: dict[str, tuple[list[str], list[str], str]] = {
    "lactose": (_DAIRY, _PLANT_DAIRY,
                "no milk, cream, butter, paneer, yogurt, curd or soft cheese; lactose-free or plant-based "
                "versions are fine, ghee and aged hard cheese only in small amounts"),
    "celiac": (_GLUTEN, _GLUTEN_FREE,
               "strictly gluten-free: no wheat, barley, rye, maida, atta, semolina, regular soy sauce or "
               "anything made from them; rice, millet, besan, corn and certified gluten-free products are fine"),
    "coeliac": (_GLUTEN, _GLUTEN_FREE, "strictly gluten-free (see celiac)"),
    "gluten": (_GLUTEN, _GLUTEN_FREE, "gluten-free: no wheat, barley, rye or products made from them"),
    "diabetes": ([], [], "low in added sugar and refined carbohydrates; prefer whole grains, fibre and protein"),
    "diabetic": ([], [], "low in added sugar and refined carbohydrates; prefer whole grains, fibre and protein"),
    "blood pressure": ([], [], "low in salt; season with herbs, spices, citrus instead"),
    "hypertension": ([], [], "low in salt; season with herbs, spices, citrus instead"),
    "cholesterol": ([], [], "low in saturated fat: little butter, ghee, cream or fatty meat"),
    "ibs": ([], [], "gentle on the gut: avoid large amounts of onion, garlic, beans and very spicy food"),
}


def _singular(word: str) -> str:
    word = word.strip().lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "shes", "ches")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(word)}(?:e?s)?(?![a-z])", text) is not None


def _allergen_key(allergy: str) -> str:
    a = " ".join(_singular(w) for w in re.findall(r"[a-z]+", allergy.lower()))
    return a.removesuffix(" allergy").removesuffix(" product").strip()


def allergen_words(allergy: str) -> list[str]:
    """'tree nuts' -> almond, cashew, ...; 'peanuts' -> peanut, groundnut; 'kiwi' -> kiwi."""
    a = _allergen_key(allergy)
    for key, words in ALLERGEN_FAMILIES.items():
        if a == key or a.endswith(" " + key):
            return words
    return [a] if a else []


def allergen_exemptions(allergy: str) -> list[str]:
    a = _allergen_key(allergy)
    for key, words in ALLERGEN_EXEMPT.items():
        if a == key or a.endswith(" " + key):
            return words
    return []


def condition_rule(condition: str) -> Optional[tuple[list[str], list[str], str]]:
    c = condition.lower()
    for key, rule in CONDITIONS.items():
        if key in c:
            return rule
    return None


def condition_hint(condition: str) -> str:
    rule = condition_rule(condition)
    return f"{condition} ({rule[2]})" if rule else condition


@dataclass(frozen=True)
class Conflict:
    ingredient: str
    reason: str  # the allergy or condition it conflicts with

    def message(self) -> str:
        return f"{self.ingredient} (conflicts with {self.reason})"


def find_conflicts(ingredient_names: Iterable[str], allergies: Iterable[str] = (),
                   conditions: Iterable[str] = ()) -> list[Conflict]:
    rules: list[tuple[str, list[str], list[str]]] = []
    for a in allergies:
        rules.append((f"{a} allergy", allergen_words(a), allergen_exemptions(a)))
    for c in conditions:
        rule = condition_rule(c)
        if rule and rule[0]:
            rules.append((c, rule[0], rule[1]))
    out: list[Conflict] = []
    for name in ingredient_names:
        text = f" {name.lower()} "
        for reason, words, exempt in rules:
            hit = next((w for w in words if _has_word(text, w)), None)
            if hit is None:
                continue
            if any(_has_word(text, e) for e in exempt):
                continue  # 'coconut milk', 'rice flour', 'lactose-free yogurt'
            out.append(Conflict(name, reason))
            break
    return out
