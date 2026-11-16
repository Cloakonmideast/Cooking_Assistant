"""Whether a dish is built around an ingredient: "kaju curry" is a cashew dish,
so a habit of leaving cashews out shouldn't quietly remove them."""

from __future__ import annotations

import re

from .preferences import habit_key
from .recipe import ingredient_words, stem_word

# Names dishes commonly use for their main ingredient (Hindi and other Indian
# languages), mapped to the English name(s) a recipe lists it under.
_DISH_NAME_WORDS = {
    "kaju": "cashew", "badam": "almond", "pista": "pistachio", "akhrot": "walnut",
    "moongphali": "peanut", "mungfali": "peanut", "nariyal": "coconut", "til": "sesame",
    "aloo": "potato", "alu": "potato", "shakarkandi": "sweet potato", "gobi": "cauliflower",
    "gobhi": "cauliflower", "palak": "spinach", "methi": "fenugreek", "matar": "pea", "mutter": "pea",
    "chana": "chickpea", "chole": "chickpea", "chhole": "chickpea", "rajma": "kidney bean",
    "baingan": "eggplant aubergine brinjal", "bhindi": "okra", "lauki": "bottle gourd",
    "karela": "bitter gourd", "kaddu": "pumpkin", "gajar": "carrot", "tamatar": "tomato",
    "pyaz": "onion", "pyaaz": "onion", "kanda": "onion", "lahsun": "garlic", "lasooni": "garlic",
    "adrak": "ginger", "jeera": "cumin", "pudina": "mint", "dhaniya": "coriander", "nimbu": "lemon",
    "dahi": "yogurt curd", "malai": "cream", "makhani": "butter", "makhan": "butter",
    "khumb": "mushroom", "anda": "egg", "ande": "egg", "murgh": "chicken", "murg": "chicken",
    "gosht": "mutton lamb", "machli": "fish", "machhi": "fish", "macher": "fish",
    "jhinga": "prawn shrimp", "kela": "banana", "aam": "mango", "chawal": "rice",
    "dal": "lentil", "daal": "lentil",
}
_ALIASES = {stem_word(k): set(ingredient_words(v)) for k, v in _DISH_NAME_WORDS.items()}

# The form an ingredient comes in: "cashew nuts" and "cumin seeds" are cashew and
# cumin dishes' main ingredient. Not "leaf" ("curry leaves" aren't a curry) or
# "oil"/"milk" ("coconut oil" isn't what coconut rice is about).
_FORM_WORDS = {"seed", "nut", "powder", "paste", "clove", "pod", "flake", "kernel", "piece", "floret"}

# What follows these isn't the dish: "pulao with cashews" (asked for this time),
# "dal without garlic", "for four".
_NOT_THE_DISH = re.compile(r"\s(?:with|without|for|but|minus|no)\s.*$|,.*$")
# "sugar-free", "gluten free": the dish is without it.
_FREE_OF = re.compile(r"[a-z]+[\s-]free\b")


def dish_words(dish: str) -> set[str]:
    """The ingredient words a dish name stands for, including what Indian names mean."""
    name = _FREE_OF.sub(" ", _NOT_THE_DISH.sub("", dish.lower()))
    words = set(ingredient_words(name))
    for w in list(words):
        words |= _ALIASES.get(w, set())
    return words


def name_glossary(dish: str) -> str:
    """'In the dish name, kaju means cashew.' for the recipe writer (qwen3 wrote
    "almonds (kaju)" for a kaju curry); "" when the name has no such word."""
    words = re.findall(r"[a-z]+", dish.lower())
    known = [(w, _DISH_NAME_WORDS[w].split()[0]) for w in dict.fromkeys(words) if w in _DISH_NAME_WORDS]
    if not known:
        return ""
    return "In the dish name, " + ", ".join(f"{w} means {en}" for w, en in known) + ".\n"


def main_words(ingredient: str) -> set[str]:
    """'cashew nuts' -> {'cashew'}, 'unsalted butter' -> {'butter'}, 'coconut milk' -> both words."""
    key = set(habit_key(ingredient).split())
    return (key - _FORM_WORDS) or key


def built_around(dish: str, ingredient: str) -> bool:
    """True when the dish's name names the ingredient ("kaju curry", "cashews")."""
    main = main_words(ingredient)
    return bool(main) and main <= dish_words(dish)
