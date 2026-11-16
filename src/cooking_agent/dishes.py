"""Whether a dish is built around an ingredient: "kaju curry" is a cashew dish,
so a habit of leaving cashews out shouldn't quietly remove them.

The word lists are never complete: they hold the names seen going wrong (in
tests or on the real model). A miss means the habit is applied as before."""

from __future__ import annotations

import re

from .preferences import habit_key
from .recipe import ingredient_words, stem_word

# Indian words dishes are named after, and what they mean (the first is what the
# recipe writer is told: "kaju means cashew").
_INDIAN_NAMES: dict[str, tuple[str, ...]] = {
    "kaju": ("cashew",), "kaaju": ("cashew",), "badam": ("almond",), "pista": ("pistachio",),
    "akhrot": ("walnut",), "moongphali": ("peanut",), "mungfali": ("peanut",), "nariyal": ("coconut",),
    "aloo": ("potato",), "alu": ("potato",), "shakarkandi": ("sweet potato",), "gobi": ("cauliflower",),
    "gobhi": ("cauliflower",), "palak": ("spinach",), "saag": ("spinach", "mustard greens"),
    "methi": ("fenugreek",), "matar": ("pea",), "mattar": ("pea",), "mutter": ("pea",),
    "chana": ("chickpea",), "channa": ("chickpea",), "chole": ("chickpea",), "chhole": ("chickpea",),
    "rajma": ("kidney bean",), "baingan": ("eggplant", "aubergine", "brinjal"), "bhindi": ("okra",),
    "lauki": ("bottle gourd",), "karela": ("bitter gourd",), "kaddu": ("pumpkin",), "gajar": ("carrot",),
    "tamatar": ("tomato",), "pyaz": ("onion",), "pyaaz": ("onion",), "kanda": ("onion",),
    "lahsun": ("garlic",), "lasooni": ("garlic",), "lehsuni": ("garlic",), "adrak": ("ginger",),
    "jeera": ("cumin",), "pudina": ("mint",), "dhaniya": ("coriander",), "dhania": ("coriander",),
    "nimbu": ("lemon",), "dahi": ("yogurt", "curd"), "malai": ("cream",), "makhani": ("butter",),
    "makhan": ("butter",), "khumb": ("mushroom",), "anda": ("egg",), "ande": ("egg",),
    "murgh": ("chicken",), "murg": ("chicken",), "gosht": ("mutton", "lamb"), "machli": ("fish",),
    "machhi": ("fish",), "macher": ("fish",), "jhinga": ("prawn", "shrimp"), "kela": ("banana",),
    "aam": ("mango",), "chawal": ("rice",), "dal": ("lentil",), "daal": ("lentil",), "dhal": ("lentil",),
    "mirchi": ("chilli",),
}
# English names for the same thing (not told to the writer).
_SAME_THING: dict[str, tuple[str, ...]] = {
    "curd": ("yogurt",), "yogurt": ("curd",), "prawn": ("shrimp",), "shrimp": ("prawn",),
    "eggplant": ("aubergine", "brinjal"), "aubergine": ("eggplant", "brinjal"), "brinjal": ("eggplant", "aubergine"),
    "omelette": ("egg",), "omelet": ("egg",),
}


def _words(names: tuple[str, ...]) -> set[str]:
    return {w for n in names for w in ingredient_words(n)}


_ALIASES = {stem_word(k): _words(v) for k, v in (_INDIAN_NAMES | _SAME_THING).items()}

# Ingredients whose name contains another one's: "peanut butter cookies" aren't a
# butter dish. Taken out of the dish name unless the habit is about them.
_COMPOUNDS = [tuple(ingredient_words(c)) for c in (
    "peanut butter", "sweet potato", "coconut milk", "coconut cream", "cream cheese", "sour cream", "ice cream",
    "butter bean", "milk chocolate", "rice noodle", "chilli oil", "cottage cheese")]

# The form an ingredient comes in: "cashew nuts" and "cumin seeds" are what cashew and
# cumin dishes are about. Not "leaf" ("curry leaves" aren't a curry), "powder" or
# "paste" ("curry powder" isn't a chicken curry), "oil"/"milk" ("coconut oil" isn't
# what coconut rice is about).
_FORM_WORDS = {"seed", "nut", "clove", "pod", "flake", "kernel", "piece", "floret"}

# What follows these isn't the dish: "pulao with cashews" (asked for this time),
# "dal without garlic", "for four".
_NOT_THE_DISH = re.compile(r"\s(?:with|without|for|but|minus|no)\s.*$|,.*$")
# "no-onion pav bhaji", "without onion dal", "no onion no garlic paneer", "sugar-free cookies".
_LEFT_OUT = re.compile(r"(?<![a-z])(?:no|without|minus)[\s-]+[a-z]+|[a-z]+[\s-]free(?![a-z])")


def _name(dish: str) -> str:
    """The dish name without what it's said to be without, or what comes after "with"."""
    return _LEFT_OUT.sub(" ", _NOT_THE_DISH.sub("", dish.lower()))


def dish_words(dish: str, about: str = "") -> set[str]:
    """The ingredient words a dish name stands for, including what Indian names
    mean. Compound ingredients ("peanut butter") are left out unless ``about``
    (the habit's ingredient) is that compound."""
    words = ingredient_words(_name(dish))
    keep = set(ingredient_words(about))
    for comp in _COMPOUNDS:
        if set(comp) <= keep:
            continue
        n = len(comp)
        i = 0
        while i + n <= len(words):
            if tuple(words[i:i + n]) == comp:
                words = words[:i] + words[i + n:]
            else:
                i += 1
    out = set(words)
    for w in words:
        out |= _ALIASES.get(w, set())
    return out


def name_glossary(dish: str) -> str:
    """'In the dish name, kaju means cashew.' for the recipe writer (qwen3 wrote
    "almonds (kaju)" for a kaju curry); "" when the name has no such word."""
    words = re.findall(r"[a-z]+", _name(dish))
    known = [(w, _INDIAN_NAMES[w][0]) for w in dict.fromkeys(words) if w in _INDIAN_NAMES]
    if not known:
        return ""
    return "In the dish name, " + ", ".join(f"{w} means {en}" for w, en in known) + ".\n"


def main_words(ingredient: str) -> set[str]:
    """'cashew nuts' -> {'cashew'}, 'unsalted butter' -> {'butter'}, 'coconut milk' -> both words."""
    key = set(habit_key(ingredient).split())
    return (key - _FORM_WORDS) or key


def same_ingredient(a: str, b: str) -> bool:
    """'cashews' and 'cashew nuts' are; 'butter' and 'peanut butter', or 'garlic
    powder' and 'garlic cloves', aren't."""
    return main_words(a) == main_words(b)


def built_around(dish: str, ingredient: str) -> bool:
    """True when the dish's name names the ingredient ("kaju curry", "cashews")."""
    main = main_words(ingredient)
    return bool(main) and main <= dish_words(dish, about=ingredient)
