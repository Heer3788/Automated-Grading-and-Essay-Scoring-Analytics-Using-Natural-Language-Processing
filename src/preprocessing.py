"""Cleaning decisions are recorded; raw text is retained in `essay_original`."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
import pandas as pd


@dataclass
class CleaningResult:
    data: pd.DataFrame
    exclusions: pd.DataFrame


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['-]\w+)*\b", text))


def prepare_training_data(frame: pd.DataFrame, minimum_words: int = 5, configured_ranges: dict | None = None) -> CleaningResult:
    data = frame.copy()
    data["essay_original"] = data["essay"]
    data["essay"] = data["essay"].map(normalize_text)
    data["essay_set"] = pd.to_numeric(data["essay_set"], errors="coerce")
    data["domain1_score"] = pd.to_numeric(data["domain1_score"], errors="coerce")
    data["word_count"] = data["essay"].map(word_count)

    reasons = pd.Series("", index=data.index, dtype="object")
    reasons = reasons.mask(data["essay"].eq(""), "missing_or_empty_essay")
    reasons = reasons.mask(data["word_count"].lt(minimum_words) & reasons.eq(""), "too_short")
    reasons = reasons.mask(data["essay_set"].isna() & reasons.eq(""), "invalid_prompt")
    reasons = reasons.mask(data["domain1_score"].isna() & reasons.eq(""), "invalid_or_missing_score")
    reasons = reasons.mask(data["domain1_score"].lt(0) & reasons.eq(""), "negative_score")
    # Optional verified rubric bounds are enforced before splitting or fitting.
    for prompt, bounds in (configured_ranges or {}).items():
        try:
            low, high = float(bounds[0]), float(bounds[1])
            prompt_value = int(prompt)
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"Invalid configured score range for prompt {prompt!r}; expected [minimum, maximum].")
        outside = data["essay_set"].eq(prompt_value) & ~data["domain1_score"].between(low, high)
        reasons = reasons.mask(outside & reasons.eq(""), "outside_configured_prompt_range")
    # Exact prompt/text duplicates would contaminate custom splits. Keep first deterministically.
    duplicate = data.duplicated(subset=["essay_set", "essay"], keep="first")
    reasons = reasons.mask(duplicate & reasons.eq(""), "duplicate_prompt_and_essay")
    excluded = data.loc[reasons.ne(""), [c for c in ["essay_id", "essay_set", "domain1_score", "word_count"] if c in data.columns]].copy()
    excluded["reason"] = reasons.loc[excluded.index]
    cleaned = data.loc[reasons.eq("")].copy().reset_index(drop=True)
    cleaned["essay_set"] = cleaned["essay_set"].astype(int)
    return CleaningResult(cleaned, excluded.reset_index(drop=True))


def score_ranges(frame: pd.DataFrame) -> dict[str, list[int]]:
    """Observed integer score bounds, to be verified against the source/rubric by users."""
    result: dict[str, list[int]] = {}
    for prompt, subset in frame.groupby("essay_set"):
        result[str(int(prompt))] = [int(subset.domain1_score.min()), int(subset.domain1_score.max())]
    return result
