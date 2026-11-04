"""The cook's recipe book: a small SQLite file of validated recipes.

``get_recipe`` looks here before asking the LLM to write a recipe, so saved
recipes (imported from a web page, a file, dictated by voice, or a generated
one the cook liked) are read back exactly as stored. Quantities are stored at
the recipe's own serving count and scaled when loaded.

SQLite calls are synchronous; the file is tiny and every call is a single
indexed read or write, so they run on the event loop without blocking it in
any noticeable way.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .recipe import Recipe

_STOPWORDS = {"a", "an", "the", "recipe", "recipes", "for", "of", "to", "make", "how", "some", "with",
              "and", "my", "me", "i", "want", "please", "s"}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'s", ""))
    out = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude singular: 'pancakes' ~ 'pancake'
        out.append(w)
    return out


def _edits(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _token_match(q: str, title_tokens: list[str]) -> bool:
    """Exact, or a near-miss of the kind speech recognition makes ('rise' for
    'rice', 'biriyani' for 'biryani'): same first letter and one edit at most.
    Strict on purpose: 'chickpea' (two edits away) must not match 'chicken'."""
    for t in title_tokens:
        if q == t:
            return True
        if len(q) > 3 and q[0] == t[0] and _edits(q, t) <= 1:
            return True
    return False


@dataclass
class SavedRecipe:
    id: int
    recipe: Recipe
    source: str  # 'dictated', 'generated', a URL, or a file name
    added_at: float

    @property
    def title(self) -> str:
        return self.recipe.title


class RecipeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if str(path) == ":memory:" else self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS recipes ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " title TEXT NOT NULL,"
            " data TEXT NOT NULL,"
            " source TEXT NOT NULL DEFAULT '',"
            " added_at REAL NOT NULL)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- writes ------------------------------------------------------------
    def add(self, recipe: Recipe, source: str = "") -> int:
        cur = self._db.execute(
            "INSERT INTO recipes (title, data, source, added_at) VALUES (?, ?, ?, ?)",
            (recipe.title, recipe.model_dump_json(), source, time.time()),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def remove(self, recipe_id: int) -> bool:
        cur = self._db.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
        self._db.commit()
        return cur.rowcount > 0

    # -- reads -------------------------------------------------------------
    def _row(self, row) -> SavedRecipe:
        rid, data, source, added = row
        return SavedRecipe(rid, Recipe.model_validate(json.loads(data)), source, added)

    def get(self, recipe_id: int) -> Optional[SavedRecipe]:
        row = self._db.execute("SELECT id, data, source, added_at FROM recipes WHERE id = ?",
                               (recipe_id,)).fetchone()
        return self._row(row) if row else None

    def all(self) -> list[SavedRecipe]:
        rows = self._db.execute("SELECT id, data, source, added_at FROM recipes ORDER BY title COLLATE NOCASE")
        return [self._row(r) for r in rows]

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

    def search(self, query: str, limit: int = 5) -> list[SavedRecipe]:
        """Recipes whose title contains every meaningful word of ``query``
        (fuzzy per word, so 'jeera rice' finds 'Jeera Rice (Cumin Rice)' and
        a small STT misspelling still matches). Best match first: fewest extra
        title words, then most recently added."""
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for saved in self.all():
            title = _tokens(saved.title)
            if title and all(_token_match(w, title) for w in q):
                scored.append((len(title) - len(q), -saved.added_at, saved))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [s for *_, s in scored[:limit]]

    def find(self, query: str) -> Optional[SavedRecipe]:
        hits = self.search(query, limit=1)
        return hits[0] if hits else None
