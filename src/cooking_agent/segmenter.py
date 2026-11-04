"""Split a streaming LLM reply into speakable segments.

TTS can start on the first sentence while the LLM is still writing the rest.
The first segment may be cut early at a clause boundary so speech starts
sooner; later segments are whole sentences. Markdown is stripped because it
would otherwise be read out loud.
"""

from __future__ import annotations

import re

_MARKDOWN = re.compile(r"[*_#`>]+")
# A period right after a digit ("Step 1. Heat...") is not treated as a sentence
# end: splitting there produces a tiny "Step 1." segment and an audible gap.
_SENTENCE_END = re.compile(r"((?:(?<!\d)\.+|[!?]+)[\"')\]]*)(\s+)")
_CLAUSE_END = re.compile(r"([,;:])(\s+)")


_FRACTIONS = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8"}


def clean_for_speech(text: str) -> str:
    # '1½ cups' -> '1 and 1/2 cups': same words, but a TTS voice can say it.
    for ch, frac in _FRACTIONS.items():
        text = re.sub(rf"(\d)\s*{ch}", rf"\1 and {frac}", text).replace(ch, frac)
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet markers
    return re.sub(r"\s+", " ", text).strip()


class SentenceSegmenter:
    def __init__(self, first_clause_min_chars: int = 30, max_chars: int = 220):
        self._buf = ""
        self._emitted_any = False
        self.first_clause_min_chars = first_clause_min_chars
        self.max_chars = max_chars

    def push(self, delta: str) -> list[str]:
        self._buf += delta.replace("\n", " \n ")
        out: list[str] = []
        while True:
            seg = self._take_one()
            if seg is None:
                break
            if seg:
                out.append(seg)
        return out

    def flush(self) -> list[str]:
        seg = clean_for_speech(self._buf)
        self._buf = ""
        if seg:
            self._emitted_any = True
            return [seg]
        return []

    def _take_one(self) -> str | None:
        buf = self._buf
        cut: int | None = None
        m = _SENTENCE_END.search(buf)
        nl = buf.find("\n")
        if m:
            cut = m.end(1)
        if nl != -1 and (cut is None or nl < cut):
            cut = nl
        if cut is None and not self._emitted_any:
            # Speak the first clause early if it is long enough to sound natural.
            for cm in _CLAUSE_END.finditer(buf):
                if cm.end(1) >= self.first_clause_min_chars:
                    cut = cm.end(1)
                    break
        if cut is None and len(buf) > self.max_chars:
            space = buf.rfind(" ", 0, self.max_chars)
            cut = space if space > 0 else self.max_chars
        if cut is None:
            return None
        seg, self._buf = buf[:cut], buf[cut:].lstrip(" \n")
        seg = clean_for_speech(seg)
        if seg:
            self._emitted_any = True
        return seg
