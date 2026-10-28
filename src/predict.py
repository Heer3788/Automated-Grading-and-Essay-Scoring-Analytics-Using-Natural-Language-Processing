"""Artifact-compatible inference with validation and no invented fallback score."""
from __future__ import annotations

from pathlib import Path
import joblib

from src.evaluate import constrain_scores
from src.feature_engineering import writing_indicators
from src.preprocessing import normalize_text
from src.settings import ROOT


class PredictionError(RuntimeError):
    pass


def predict_essay(essay_text: str, prompt: int | str, model_dir: str | Path = ROOT / "models", minimum_words: int = 5) -> dict:
    try:
        prompt_int = int(prompt)
    except (TypeError, ValueError) as exc:
        raise PredictionError("Prompt must be a valid numeric prompt identifier.") from exc
    text = normalize_text(essay_text)
    indicators = writing_indicators(text)
    if not text or indicators["word_count"] < minimum_words:
        raise PredictionError(f"Essay must contain at least {minimum_words} words before an estimate is shown.")
    artifact_path = Path(model_dir) / f"prompt_{prompt_int}.joblib"
    if not artifact_path.exists():
        raise PredictionError(f"No trained model for prompt {prompt_int}. Run: python -m src.train")
    artifact = joblib.load(artifact_path)
    if artifact.get("prompt") != prompt_int or "model" not in artifact or "score_range" not in artifact:
        raise PredictionError("Model artifact is incompatible or corrupted; retrain with: python -m src.train")
    low, high = artifact["score_range"]
    raw = float(artifact["model"].predict([text])[0])
    constrained = int(constrain_scores([raw], low, high)[0])
    return {"prompt": prompt_int, "raw_prediction": raw, "predicted_score": constrained, "score_range": [low, high], "normalized_percent": round(100 * (constrained - low) / max(high - low, 1), 1), "indicators": indicators, "model_name": artifact.get("model_name"), "model_version": artifact.get("model_version")}
