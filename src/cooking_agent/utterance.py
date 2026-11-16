"""What kind of thing the cook said, judged in code where the model proved
unreliable (Phase 2 review).

* ``asks_permission``: "could I leave out the garlic?" asks whether, not to.
  On qwen3 it removed the garlic 3 times out of 3, and two such questions
  would have become a habit. "Can you leave out the garlic" is a request.
* ``is_question``: any question, e.g. "why do we soak it?" asked while the
  app's timer offer is still open, which must not count as declining it.
"""

from __future__ import annotations

import re

_LEAD = re.compile(r"^(?:(?:ok|okay|so|um|umm|uh|hmm|and|but|well|wait|hey|actually|right|alright|oh)[\s,.!]+)+")
_PERMISSION = re.compile(
    r"^(?:"
    r"(?:can|could|should|may|might|shall)\s+(?:i|we)\b"
    r"|(?:is|would|will)\s+(?:it|that|this)\s+(?:be\s+)?(?:ok|okay|alright|fine|work|matter|bad|safe)\b"
    r"|(?:would|will|does)\s+(?:it|this|that)\s+(?:still\s+)?work\b"
    r"|is\s+it\s+(?:ok|okay|alright|fine)\b"
    r"|(?:what|how)\s+about\s+(?:if\s+)?(?:i|we)\b"
    r"|what\s+if\s+(?:i|we)\b"
    r"|do\s+(?:i|we)\s+(?:have|need)\s+to\b"
    r")"
)
_QUESTION_START = re.compile(
    r"^(?:what|what's|whats|why|how|how's|when|where|which|who|is|are|was|do|does|did|can|could|should|"
    r"would|will|shall|may|might|have|has|isn't|aren't|don't|doesn't)\b")


def _clean(text: str) -> str:
    t = " ".join(text.lower().replace("’", "'").split())
    return _LEAD.sub("", t).strip()


def asks_permission(text: str) -> bool:
    """True for "could I / can we / is it okay to / would it work without / what if I ..."."""
    return bool(_PERMISSION.match(_clean(text)))


def is_question(text: str) -> bool:
    t = _clean(text)
    return t.endswith("?") or bool(_QUESTION_START.match(t))
