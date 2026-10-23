"""Script-aware word extraction and small text helpers.

Fertility is reported per *word*; for Indic scripts a word is a whitespace-delimited
run stripped of punctuation, and we additionally report a Unicode letter-class
definition so the audit does not depend on one debatable choice.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import regex as re

# letters + combining marks + digits (Indic vowel signs are marks: \p{M})
WORD_RE = re.compile(r"[\p{L}\p{M}\p{N}][\p{L}\p{M}\p{N}‌‍'’-]*")
PUNCT_STRIP_RE = re.compile(r"^[\p{P}\p{S}]+|[\p{P}\p{S}]+$")
DEVANAGARI_DANDA = "।॥"


def words_whitespace(text: str) -> list[str]:
    """Whitespace tokens with leading/trailing punctuation removed."""
    out = []
    for tok in text.split():
        tok = PUNCT_STRIP_RE.sub("", tok.strip(DEVANAGARI_DANDA))
        if tok:
            out.append(tok)
    return out


def words_unicode(text: str) -> list[str]:
    """Unicode letter-class words (script-aware, punctuation-free by construction)."""
    return WORD_RE.findall(text)


def n_chars(text: str) -> int:
    return len(text)


def n_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def word_type_counts(texts: Iterable[str], min_len: int = 2) -> Counter:
    """Frequency table of word *types* over a corpus (used to pick gold word lists)."""
    counts: Counter = Counter()
    for text in texts:
        for w in words_unicode(text):
            if len(w) >= min_len:
                counts[w] += 1
    return counts


def is_mostly_script(word: str, script_ranges: list[tuple[int, int]] | None) -> bool:
    if not script_ranges:
        return True
    hits = sum(any(lo <= ord(c) <= hi for lo, hi in script_ranges) for c in word)
    return hits >= max(1, int(0.8 * len(word)))


# Unicode blocks per script name used in languages.py
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Latin": [(0x0041, 0x024F)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
}
