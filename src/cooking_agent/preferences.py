"""What the assistant knows about the cook, so new recipes fit them.

Two kinds of knowledge, kept apart because they deserve different weight:

* **Stated** preferences: things the cook said ("I'm vegetarian", "I'm
  allergic to peanuts", "I don't like mushrooms", "I always have paneer").
  Saved by the ``remember_preference`` tool or ``cooking-agent prefs add``.
* **Learned** habits, inferred from what the cook does: a diet they keep
  asking for, how many people they usually cook for, the ingredients
  that keep showing up in recipes they actually cook (reached step 1), and
  the changes they keep making (swaps, things left out or added, more or
  less of something). Every change is saved; it counts as a habit once it
  happened in ``ADJUST_MIN_TIMES`` different recipes.

``profile()`` merges both into the dict given to the LLM every turn and to
the recipe writer. Allergies are always hard rules; learned habits are
defaults the cook's request can override, and the assistant says when it
applied one.

Stored in the same local SQLite file as the recipe book; nothing leaves the
machine except as part of the prompt sent to the configured LLM.
"""

from __future__ import annotations

import collections
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .dietary import ALLERGEN_FAMILIES, DIETS, allergen_words, condition_hint, diet_hint, find_conflicts
from .recipe import ingredient_words

CATEGORIES = ("allergy", "condition", "diet", "dislike", "like", "pantry", "other")

# Constraint words that describe a diet, for spotting a habit in requests.
DIET_WORDS = {
    "eggetarian": "eggetarian", "eggitarian": "eggetarian", "eggetarians": "eggetarian",
    "eggitarians": "eggetarian", "ovo vegetarian": "eggetarian", "ovo-vegetarian": "eggetarian",
    "vegetarian": "vegetarian", "veg": "vegetarian", "veggie": "vegetarian", "vegan": "vegan",
    "eggless": "eggless", "jain": "jain", "gluten-free": "gluten-free", "gluten free": "gluten-free",
    "dairy-free": "dairy-free", "dairy free": "dairy-free", "low-carb": "low-carb", "low carb": "low-carb",
    "keto": "keto", "halal": "halal", "kosher": "kosher", "pescatarian": "pescatarian",
}
# Too common to say anything about the cook.
_BORING = {"salt", "water", "pepper", "black pepper", "sugar", "oil", "ice"}
_ADJECTIVES = {"fresh", "large", "small", "medium", "chopped", "finely", "sliced", "diced", "minced",
               "ground", "whole", "dried", "grated", "boneless", "skinless", "ripe", "cooked", "raw"}

# A learned habit needs this much evidence.
HISTORY_WINDOW = 12  # most recent recipe requests considered
MIN_TIMES = 3
DIET_SHARE = 0.5
INGREDIENT_MIN_RECIPES = 2
TOP_INGREDIENTS = 8
ADJUST_KINDS = ("swap", "remove", "add", "more", "less")
ADJUST_MIN_TIMES = 2  # the same change in this many different recipes is a habit
TOP_ADJUSTMENTS = 8
_HABIT_KEYS = {"swap": "usually_swaps", "remove": "usually_leaves_out", "add": "usually_adds",
               "more": "usually_more", "less": "usually_less"}


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


# Words that don't change what a habit is about ("red onion" is an onion habit).
_HABIT_IGNORE = {"unsalted", "salted", "red", "white", "yellow", "green", "organic", "plain", "extra", "virgin",
                 "chopped", "sliced", "diced", "minced", "grated", "fresh", "large", "small", "medium", "finely",
                 "ground", "whole", "dried", "ripe", "cooked", "raw", "boneless", "skinless"}


def habit_key(name: str) -> str:
    """'red onions' -> 'onion', 'unsalted butter' -> 'butter', 'green chillies' -> 'chilli'."""
    words = ingredient_words(name)
    return " ".join(w for w in words if w not in _HABIT_IGNORE) or " ".join(words)


def ingredient_key(name: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in _ADJECTIVES]
    return " ".join(words)


# "non-veg", "not vegetarian", "no veg" name the opposite of the diet.
_NEGATED = re.compile(r"(?<![a-z])(?:non|not|no)[\s-]+$")
# "with lots of veg", "less veg more meat": vegetables, not a diet.
_VEG_AS_VEGETABLES = re.compile(r"(?<![a-z])(?:of|more|less|some|any|the|with|extra|fresh|mixed|seasonal)\s+$")


# "vegetarian but I eat eggs", "eggs are fine", "I eat eggs but no meat": a vegetarian who eats eggs.
_EATS_EGGS = re.compile(r"(?<![a-z])(?:(?:eat|eats|have|has|take|takes|having|eating)\s+eggs?"
                        r"|eggs?\s+(?:are|is)\s+(?:fine|ok|okay|allowed|alright)"
                        r"|(?:vegetarian|veg|veggie)\s+(?:with|plus|and|\+|except(?:\s+for)?)\s+eggs?"
                        r"|eggs?\s+vegetarian)(?![a-z])")
# Anything in the same clause that says the cook doesn't eat them ("I don't like to eat eggs",
# "I've stopped eating eggs") or that it's someone else ("my husband eats eggs").
_EGGS_NEGATED = re.compile(r"(?:n't|(?<![a-z])(?:not|never|no|avoid|avoiding|stopped|stop|quit|gave up|give up|"
                           r"without|cannot|my|his|her|he|she|they|their|husband|wife|partner|kids|son|daughter|"
                           r"family|friend|guest)(?![a-z]))")
_EGGS_NEGATED_AFTER = re.compile(r"^[^.;!?]*?(?:but|though|although)\s+(?:i|we)\s+(?:don't|do not|never|won't|"
                                 r"can't|cannot|dont)(?![a-z])")
_CLAUSE_START = re.compile(r"[,.;:!?]|(?<![a-z])(?:but|though|although)(?![a-z])")
# "I eat eggs and fish": not an eggetarian (fish would be flagged)
_EATS_FISH = re.compile(r"(?<![a-z])(?:eat|eats|eating|have|has)\s+(?:\w+\s+){0,3}?(?:fish|seafood|prawns?|shrimps?)"
                        r"(?![a-z])")
_NO_MEAT = re.compile(r"(?<![a-z])(?:no|not|without|don't eat|do not eat|never eat)\s+(?:any\s+)?meat(?![a-z])")


def _eats_eggs(t: str) -> bool:
    for m in _EATS_EGGS.finditer(t):
        before = t[: m.start()]
        starts = [c.end() for c in _CLAUSE_START.finditer(before)]
        clause = before[starts[-1]:] if starts else before
        if _EGGS_NEGATED.search(clause) or _EGGS_NEGATED_AFTER.search(t[m.end():]):
            continue  # when in doubt, plain vegetarian (eggs get checked)
        return True
    return False


# The diets that replace one another when the cook states a new one.
DIET_FAMILY = {"vegetarian", "eggetarian", "vegan", "jain"}
_DIET_WORD = (r"(?:vegetarians?|veg|veggie|vegans?|jains?|eggetarians?|eggitarians?|ovo[- ]vegetarians?)"
              r"(?![a-z])")
# "I'm a vegetarian", "we're vegan", "I've gone vegan", "I'm strictly jain": the cook about themselves,
# a diet word right after (so "I'm making a vegetarian biryani" doesn't count).
_STATED_DIET = re.compile(
    r"(?:^|[,.;!?]\s*|(?<![a-z])(?:and|but|also|so)\s+)"
    r"(?:i'?m|i am|we'?re|we are|i'?ve (?:gone|become|turned)|we'?ve (?:gone|become|turned)|"
    r"i (?:have )?(?:gone|become|turned)|i follow|we follow|i'?m now)\s+"
    r"(?:(?:a|an|now|strictly|pure|fully|mostly|also|totally|completely|basically|still|actually|"
    r"a strict|a pure)\s+){0,3}" + _DIET_WORD)
_EATS_EGGS_NO_MEAT = re.compile(r"(?:^|[,.;!?]\s*|(?<![a-z])(?:and|but)\s+)(?:i|we) (?:only )?eat eggs?(?![a-z])")
_ONE_OFF = re.compile(r"(?<![a-z])(?:today|tonight|this time|for now|this week|this weekend|just for|on weekdays|"
                      r"on weekends|sometimes|for lent)(?![a-z])")


def stated_diets(text: Optional[str]) -> set[str]:
    """Diets the cook says they follow ("I'm vegetarian but I eat eggs" ->
    {"eggetarian"}); empty for requests, other people, negations and one-offs.
    Saved in code because the model often said "I'll remember" without
    saving (Phase 2 review)."""
    t = _norm(text or "")
    if not (_STATED_DIET.search(t) or _EATS_EGGS_NO_MEAT.search(t)) or _ONE_OFF.search(t):
        return set()
    sentence = re.split(r"[.!?]", t[(_STATED_DIET.search(t) or _EATS_EGGS_NO_MEAT.search(t)).start():] + ".")[0]
    return diets_in(sentence) & DIET_FAMILY


def diets_in(text: Optional[str]) -> set[str]:
    t = f" {_norm(text or '')} "
    found = _diet_words_in(t)
    eats_fish = "pescatarian" in found or _EATS_FISH.search(t)
    if _eats_eggs(t) and not eats_fish and (found & {"vegetarian", "eggetarian"} or _NO_MEAT.search(t)):
        found.add("eggetarian")
    return prefer_eggetarian(found)


def prefer_eggetarian(diets: set[str]) -> set[str]:
    """Eggetarian is vegetarian plus eggs: when both are named, the more
    specific one applies (an eggless request still checks eggs via 'eggless')."""
    return diets - {"vegetarian"} if "eggetarian" in diets else diets


def _diet_words_in(t: str) -> set[str]:
    found: set[str] = set()
    for word, diet in DIET_WORDS.items():
        for m in re.finditer(rf"(?<![a-z]){re.escape(word)}(?![a-z-])", t):
            before = t[: m.start()]
            if _NEGATED.search(before):
                continue
            if word in ("veg", "veggie") and _VEG_AS_VEGETABLES.search(before):
                continue
            found.add(diet)
            break
    return found


class PreferenceStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = Path(path).expanduser()
        self._db = sqlite3.connect(str(path))
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS preferences ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, value TEXT NOT NULL,"
            " added_at REAL NOT NULL, UNIQUE(category, value));"
            "CREATE TABLE IF NOT EXISTS requests ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, dish TEXT NOT NULL, constraints TEXT,"
            " servings INTEGER, title TEXT, cooked INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL);"
            "CREATE TABLE IF NOT EXISTS ingredient_use ("
            " name TEXT PRIMARY KEY, recipes INTEGER NOT NULL DEFAULT 0, last_ts REAL NOT NULL);"
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            # One row per change the cook made; load_id is the recipe it was made in, so
            # repeating a change within one recipe doesn't make it a habit.
            "CREATE TABLE IF NOT EXISTS adjustments ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, original TEXT NOT NULL,"
            " replacement TEXT NOT NULL DEFAULT '', load_id INTEGER, ts REAL NOT NULL);"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- stated preferences -------------------------------------------------
    def remember(self, category: str, value: str) -> bool:
        """Returns False if it was already known."""
        category, value = category.lower(), _norm(value)
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {', '.join(CATEGORIES)}")
        if not value:
            raise ValueError("empty preference")
        cur = self._db.execute(
            "INSERT OR IGNORE INTO preferences (category, value, added_at) VALUES (?, ?, ?)",
            (category, value, time.time()))
        self._db.commit()
        return cur.rowcount > 0

    def set_stated_diets(self, diets: set[str]) -> None:
        """Save diets the cook just stated; the newest replaces an older one
        from the same family ("I'm vegetarian now, no eggs" after eggetarian)."""
        if not diets:
            return
        for row in self.stated():
            if row["category"] == "diet" and diets_in(row["value"]) & DIET_FAMILY:
                self._db.execute("DELETE FROM preferences WHERE id = ?", (row["id"],))
        self._db.commit()
        for d in sorted(diets):
            self.remember("diet", d)

    def forget(self, value: str) -> list[dict]:
        """Remove stated preferences mentioning ``value`` (any category), and
        learned changes about it ("forget that I swap ghee for butter")."""
        v = _norm(value)
        rows = [r for r in self.stated() if v in r["value"] or r["value"] in v]
        for r in rows:
            self._db.execute("DELETE FROM preferences WHERE id = ?", (r["id"],))
        words = set(habit_key(v).split())
        habits: dict[str, None] = {}
        for i, kind, original, replacement in self._db.execute(
                "SELECT id, kind, original, replacement FROM adjustments").fetchall():
            if words and any(words <= set(habit_key(n).split()) for n in (original, replacement) if n):
                self._db.execute("DELETE FROM adjustments WHERE id = ?", (i,))
                habits[_habit_text(kind, original, replacement)] = None
        self._db.commit()
        return rows + [{"id": None, "category": "habit", "value": h} for h in habits]

    def stated(self) -> list[dict]:
        rows = self._db.execute("SELECT id, category, value FROM preferences ORDER BY category, added_at, id")
        return [{"id": i, "category": c, "value": v} for i, c, v in rows]

    # -- first-run setup -----------------------------------------------------
    def needs_setup(self) -> bool:
        """True until the cook has answered (or skipped) the first-run
        question about allergies, conditions and diet."""
        return self._db.execute("SELECT 1 FROM meta WHERE key = 'setup_done'").fetchone() is None

    def mark_setup_done(self, skipped: bool = False) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('setup_done', ?)",
                         ("skipped" if skipped else str(time.time()),))
        self._db.commit()

    def reset_setup(self) -> None:
        self._db.execute("DELETE FROM meta WHERE key = 'setup_done'")
        self._db.commit()

    # -- learning ------------------------------------------------------------
    def record_request(self, dish: str, constraints: Optional[str], servings: Optional[int]) -> int:
        cur = self._db.execute(
            "INSERT INTO requests (dish, constraints, servings, ts) VALUES (?, ?, ?, ?)",
            (_norm(dish), _norm(constraints) if constraints else None, servings, time.time()))
        self._db.commit()
        return int(cur.lastrowid)

    def record_cooked(self, request_id: Optional[int], title: str, ingredient_names: list[str]) -> None:
        """Called once when the cook actually starts a recipe (reaches step 1)."""
        now = time.time()
        if request_id is not None:
            self._db.execute("UPDATE requests SET cooked = 1, title = ? WHERE id = ?", (title, request_id))
        for key in {ingredient_key(n) for n in ingredient_names}:
            if key and key not in _BORING:
                self._db.execute(
                    "INSERT INTO ingredient_use (name, recipes, last_ts) VALUES (?, 1, ?) "
                    "ON CONFLICT(name) DO UPDATE SET recipes = recipes + 1, last_ts = excluded.last_ts",
                    (key, now))
        self._db.commit()

    def record_adjustment(self, kind: str, original: str, replacement: Optional[str] = None,
                          load_id: Optional[int] = None) -> None:
        """Save a change the cook made to a recipe (every time)."""
        if kind not in ADJUST_KINDS:
            raise ValueError(f"kind must be one of {', '.join(ADJUST_KINDS)}")
        original = ingredient_key(original)
        replacement = ingredient_key(replacement) if replacement else ""
        if not original or (kind == "swap" and not replacement):
            return
        self._db.execute(
            "INSERT INTO adjustments (kind, original, replacement, load_id, ts) VALUES (?, ?, ?, ?, ?)",
            (kind, original, replacement, load_id, time.time()))
        self._db.commit()

    def _grouped_adjustments(self) -> list[dict]:
        """Changes grouped by what they're about ("onions", "onion" and "red
        onion" are one), each with the number of different recipes it was made
        in, when it was last made, and the name it was most often called."""
        groups: dict[tuple, dict] = {}
        for i, kind, original, replacement, load_id, ts in self._db.execute(
                "SELECT id, kind, original, replacement, load_id, ts FROM adjustments ORDER BY id").fetchall():
            key = (kind, habit_key(original), habit_key(replacement) if replacement else "")
            g = groups.setdefault(key, {"loads": set(), "names": collections.Counter(),
                                        "repl": collections.Counter(), "last": 0.0})
            g["loads"].add(load_id if load_id is not None else -i)
            g["names"][original] += 1
            if replacement:
                g["repl"][replacement] += 1
            g["last"] = max(g["last"], ts)

        def name(c: collections.Counter) -> Optional[str]:
            return min(c, key=lambda n: (-c[n], len(n), n)) if c else None

        return [{"kind": k[0], "key": k[1:], "original": name(g["names"]), "replacement": name(g["repl"]),
                 "times": len(g["loads"]), "last": g["last"]} for k, g in groups.items()]

    def adjustments(self, min_times: int = 1) -> list[dict]:
        """Saved changes with the number of different recipes each was made in."""
        rows = [a for a in self._grouped_adjustments() if a["times"] >= min_times]
        rows.sort(key=lambda a: (-a["times"], -a["last"]))
        return [{"kind": a["kind"], "original": a["original"], "replacement": a["replacement"], "times": a["times"]}
                for a in rows]

    def forget_history(self) -> None:
        self._db.executescript("DELETE FROM requests; DELETE FROM ingredient_use; DELETE FROM adjustments;")
        self._db.commit()

    def learned(self) -> dict:
        rows = self._db.execute(
            "SELECT dish || ' ' || COALESCE(constraints, ''), servings FROM requests ORDER BY id DESC LIMIT ?",
            (HISTORY_WINDOW,)).fetchall()
        out: dict = {}
        if rows:
            # 'vegetarian lasagne' and dish='lasagne', constraints='vegetarian' both count
            diet_counts = collections.Counter(d for c, _ in rows for d in diets_in(c))
            diets = sorted(d for d, n in diet_counts.items() if n >= MIN_TIMES and n / len(rows) >= DIET_SHARE)
            if diets:
                out["usually_asks_for"] = diets
            sizes = collections.Counter(s for _, s in rows if s)
            if sizes:
                size, n = sizes.most_common(1)[0]
                if n >= MIN_TIMES:
                    out["usually_cooks_for"] = size
        used = self._db.execute(
            "SELECT name FROM ingredient_use WHERE recipes >= ? ORDER BY recipes DESC, last_ts DESC LIMIT ?",
            (INGREDIENT_MIN_RECIPES, TOP_INGREDIENTS)).fetchall()
        if used:
            out["often_cooks_with"] = [n for (n,) in used]
        # Opposite habits (more/less, leave out/add, ghee->butter/butter->ghee): the latest wins.
        decided: set = set()
        habits = sorted((a for a in self._grouped_adjustments() if a["times"] >= ADJUST_MIN_TIMES),
                        key=lambda a: -a["last"])
        for a in habits:
            kind, (orig, repl) = a["kind"], a["key"]
            topic = (("amount", orig) if kind in ("more", "less") else ("in-out", orig) if kind in ("remove", "add")
                     else ("swap", frozenset((orig, repl))))
            if topic in decided:
                continue
            decided.add(topic)
            items = out.setdefault(_HABIT_KEYS[kind], [])
            if len(items) < TOP_ADJUSTMENTS:
                items.append({"from": a["original"], "to": a["replacement"]} if kind == "swap" else a["original"])
        return out

    # -- combined ------------------------------------------------------------
    def profile(self) -> dict:
        """Compact view for the LLM. Empty dict when nothing is known."""
        out: dict = {}
        for row in self.stated():
            out.setdefault(row["category"], []).append(row["value"])
        learned = self.learned()
        said = out.get("diet", [])
        stated_diets = set(said) | diets_in(" ".join(said))
        if "usually_asks_for" in learned:
            # an eggetarian's "veg" requests aren't a stricter vegetarian habit
            learned["usually_asks_for"] = [
                d for d in learned["usually_asks_for"]
                if d not in stated_diets and not (d == "vegetarian" and "eggetarian" in stated_diets)]
            if not learned["usually_asks_for"]:
                del learned["usually_asks_for"]
        if learned:
            out["learned_from_history"] = learned
        return out

    def default_servings(self) -> Optional[int]:
        return self.learned().get("usually_cooks_for")


def _diet_rule(stated: str) -> str:
    """'vegetarian' -> 'vegetarian (no meat, poultry, fish, ...)'; unknown diets as said."""
    known = sorted(diets_in(stated) & DIETS.keys())
    return diet_hint(known[0]) if known else stated


def _habit_text(kind: str, original: str, replacement: Optional[str]) -> str:
    return {"swap": f"swaps {original} for {replacement}", "remove": f"leaves out {original}",
            "add": f"adds {original}", "more": f"more {original}", "less": f"less {original}"}[kind]


# A learned "more" or "add" of these isn't passed on with the condition (salt with high blood pressure).
_CONDITION_LIMITS = [("blood pressure", {"salt", "soy", "stock", "bouillon", "pickle", "papad", "msg"}),
                     ("hypertension", {"salt", "soy", "stock", "bouillon", "pickle", "papad", "msg"}),
                     ("diabet", {"sugar", "jaggery", "honey", "syrup", "gur", "condensed"}),
                     ("cholesterol", {"butter", "ghee", "cream", "lard"})]


def _safe(profile: dict, items: list[str], more: bool = False) -> list[str]:
    """Never tell the writer to prefer something the cook can't have ('paneer'
    + lactose intolerance, 'peanut oil' + peanut allergy, 'chicken' + vegetarian).
    With ``more`` (a learned "more" or "add"), also nothing a health condition
    limits, and a "nuts" allergy covers peanuts too."""
    diets = sorted(prefer_eggetarian(diets_in(" ".join(profile.get("diet", [])))) & DIETS.keys())
    allergies = list(profile.get("allergy", []))
    if any(allergen_words(a) == ALLERGEN_FAMILIES["nut"] for a in allergies):
        allergies.append("peanuts")  # learned habits only; the main check keeps its own word lists
    bad = {c.ingredient for c in find_conflicts(items, allergies, profile.get("condition", []), diets=diets)}
    if more:
        conditions = " ".join(profile.get("condition", [])).lower()
        limited = set().union(*(words for key, words in _CONDITION_LIMITS if key in conditions))
        bad |= {i for i in items if set(ingredient_words(i)) & limited}
    return [i for i in items if i not in bad]


def usual_changes(profile: dict) -> dict:
    """The learned changes that are safe to apply for this cook."""
    learned = profile.get("learned_from_history", {})
    swaps = learned.get("usually_swaps", [])
    safe_to = set(_safe(profile, [s["to"] for s in swaps], more=True))
    return {
        "swaps": [s for s in swaps if s["to"] in safe_to],
        "leaves_out": learned.get("usually_leaves_out", []),
        "adds": _safe(profile, learned.get("usually_adds", []), more=True),
        "more": _safe(profile, learned.get("usually_more", []), more=True),
        "less": learned.get("usually_less", []),
    }


def describe_for_recipe(profile: dict) -> str:
    """Instructions for the recipe writer, built from ``profile()``."""
    if not profile:
        return ""

    def safe(items: list[str]) -> list[str]:
        return _safe(profile, items)

    lines = []
    if profile.get("allergy"):
        lines.append(f"- ALLERGIES, never include, not even as garnish or optional: {', '.join(profile['allergy'])}.")
    if profile.get("condition"):
        lines.append("- HEALTH CONDITIONS, treat as strict rules for every ingredient: "
                     + "; ".join(condition_hint(c) for c in profile["condition"]) + ".")
    if profile.get("diet"):
        # one rule per diet in effect: "vegetarian" and "eggetarian" both stored means eggetarian
        known = sorted(prefer_eggetarian(diets_in(" ".join(profile["diet"]))) & DIETS.keys())
        other = [d for d in profile["diet"] if not diets_in(d) & DIETS.keys()]
        lines.append(f"- Diet: {'; '.join([diet_hint(d) for d in known] + other)}.")
    if profile.get("dislike"):
        lines.append(f"- Leave out (dislikes): {', '.join(profile['dislike'])}.")
    if profile.get("like"):
        lines.append(f"- Likes: {', '.join(profile['like'])}.")
    if pantry := safe(profile.get("pantry", [])):
        lines.append(f"- Usually has at home, prefer these where they fit: {', '.join(pantry)}.")
    if profile.get("other"):
        lines.append(f"- Also: {'; '.join(profile['other'])}.")
    learned = profile.get("learned_from_history", {})
    if learned.get("usually_asks_for"):
        lines.append(f"- Usually asks for {', '.join(learned['usually_asks_for'])} recipes; follow that "
                     "unless the request says otherwise.")
    if often := safe(learned.get("often_cooks_with", [])):
        lines.append(f"- Often cooks with: {', '.join(often)}; use them where natural.")
    usual = usual_changes(profile)
    # more/less are applied to the written recipe in code (tools._apply_usual_changes), not asked for
    # here, so a writer that follows the hint doesn't get them twice
    habits = ([f"usually swaps {s['from']} for {s['to']}" for s in usual["swaps"]]
              + [f"usually leaves out {x}" for x in usual["leaves_out"]]
              + [f"usually adds {x}" for x in usual["adds"]])
    if habits:
        lines.append(f"- Changes they usually make, apply them unless the request says otherwise: "
                     f"{'; '.join(habits)}.")
    if not lines:
        return ""
    return ("About this cook (the request itself wins where it conflicts, except allergies and health "
            "conditions):\n"
            + "\n".join(lines) + "\n")
