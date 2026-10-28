"""Descriptive writing indicators, not measures of educational merit."""
from __future__ import annotations
import re
import numpy as np

WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def writing_indicators(text: str) -> dict[str, float | int]:
    text = text or ""
    words = WORD_RE.findall(text)
    sentences = [s for s in SENTENCE_RE.findall(text) if s.strip()]
    lengths = [len(WORD_RE.findall(s)) for s in sentences] or [0]
    unique = len({w.lower() for w in words})
    return {
        "word_count": len(words), "character_count": len(text), "sentence_count": len(sentences),
        "average_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
        "average_word_length": round(float(np.mean([len(w) for w in words])) if words else 0.0, 2),
        "paragraph_count": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        "vocabulary_diversity": round(unique / max(len(words), 1), 3),
        "punctuation_count": len(re.findall(r"[,:;.!?]", text)),
        "sentence_length_std": round(float(np.std(lengths)), 2),
    }
