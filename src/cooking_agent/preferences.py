"""What the assistant knows about the cook, so new recipes fit them.

Two kinds of knowledge, kept apart because they deserve different weight:

* **Stated** preferences: things the cook said ("I'm vegetarian", "I'm
  allergic to peanuts", "I don't like mushrooms", "I always have paneer").
  Saved by the ``remember_preference`` tool or ``cooking-agent prefs add``.
* **Learned** habits, inferred from what the cook does: a diet they keep
  asking for, how many people they usually cook for, and the ingredients
  that keep showing up in recipes they actually cook (reached step 1).

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

CATEGORIES = ("diet", "allergy", "dislike", "like", "pantry", "other")

# Constraint words that describe a diet, for spotting a habit in requests.
DIET_WORDS = {
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


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def ingredient_key(name: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in _ADJECTIVES]
    return " ".join(words)


def diets_in(text: Optional[str]) -> set[str]:
    t = f" {_norm(text or '')} "
    return {d for word, d in DIET_WORDS.items() if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", t)}


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

    def forget(self, value: str) -> list[dict]:
        """Remove stated preferences mentioning ``value`` (any category)."""
        v = _norm(value)
        rows = [r for r in self.stated() if v in r["value"] or r["value"] in v]
        for r in rows:
            self._db.execute("DELETE FROM preferences WHERE id = ?", (r["id"],))
        self._db.commit()
        return rows

    def stated(self) -> list[dict]:
        rows = self._db.execute("SELECT id, category, value FROM preferences ORDER BY category, added_at")
        return [{"id": i, "category": c, "value": v} for i, c, v in rows]

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

    def forget_history(self) -> None:
        self._db.executescript("DELETE FROM requests; DELETE FROM ingredient_use;")
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
        return out

    # -- combined ------------------------------------------------------------
    def profile(self) -> dict:
        """Compact view for the LLM. Empty dict when nothing is known."""
        out: dict = {}
        for row in self.stated():
            out.setdefault(row["category"], []).append(row["value"])
        learned = self.learned()
        stated_diets = set(out.get("diet", []))
        if "usually_asks_for" in learned:
            learned["usually_asks_for"] = [d for d in learned["usually_asks_for"] if d not in stated_diets]
            if not learned["usually_asks_for"]:
                del learned["usually_asks_for"]
        if learned:
            out["learned_from_history"] = learned
        return out

    def default_servings(self) -> Optional[int]:
        return self.learned().get("usually_cooks_for")


def describe_for_recipe(profile: dict) -> str:
    """Instructions for the recipe writer, built from ``profile()``."""
    if not profile:
        return ""
    lines = []
    if profile.get("allergy"):
        lines.append(f"- ALLERGIES, never include, not even as garnish or optional: {', '.join(profile['allergy'])}.")
    if profile.get("diet"):
        lines.append(f"- Diet: {', '.join(profile['diet'])}.")
    if profile.get("dislike"):
        lines.append(f"- Leave out (dislikes): {', '.join(profile['dislike'])}.")
    if profile.get("like"):
        lines.append(f"- Likes: {', '.join(profile['like'])}.")
    if profile.get("pantry"):
        lines.append(f"- Usually has at home, prefer these where they fit: {', '.join(profile['pantry'])}.")
    if profile.get("other"):
        lines.append(f"- Also: {'; '.join(profile['other'])}.")
    learned = profile.get("learned_from_history", {})
    if learned.get("usually_asks_for"):
        lines.append(f"- Usually asks for {', '.join(learned['usually_asks_for'])} recipes; follow that "
                     "unless the request says otherwise.")
    if learned.get("often_cooks_with"):
        lines.append(f"- Often cooks with: {', '.join(learned['often_cooks_with'])}; use them where natural.")
    return ("About this cook (the request itself wins where it conflicts, except allergies):\n"
            + "\n".join(lines) + "\n")
