"""Held-out essay scoring metrics, including guarded ordinal agreement."""
from __future__ import annotations

import math
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from src.settings import ROOT


def constrain_scores(values, low: int, high: int) -> np.ndarray:
    """Round to valid whole score categories then bound to the verified prompt range."""
    return np.clip(np.rint(np.asarray(values, dtype=float)), low, high).astype(int)


def scoring_metrics(y_true, raw_predictions, low: int, high: int) -> dict[str, float | int | None]:
    actual = np.asarray(y_true, dtype=float)
    raw = np.asarray(raw_predictions, dtype=float)
    predicted = constrain_scores(raw, low, high)
    metrics: dict[str, float | int | None] = {
        "n": int(len(actual)),
        "raw_mae": round(float(mean_absolute_error(actual, raw)), 4),
        "raw_rmse": round(float(math.sqrt(mean_squared_error(actual, raw))), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(actual, predicted))), 4),
        "exact_agreement": round(float(np.mean(actual == predicted)), 4),
        "within_one": round(float(np.mean(np.abs(actual - predicted) <= 1)), 4),
        "qwk": None,
    }
    # Kappa is undefined when either side lacks variation; disclose rather than manufacture a value.
    if len(actual) >= 2 and len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
        metrics["qwk"] = round(float(cohen_kappa_score(actual.astype(int), predicted, weights="quadratic")), 4)
    return metrics


def metrics_by_prompt(frame: pd.DataFrame) -> dict[str, dict]:
    results = {}
    for prompt, group in frame.groupby("essay_set"):
        low, high = int(group.score_min.iloc[0]), int(group.score_max.iloc[0])
        results[str(int(prompt))] = scoring_metrics(group.domain1_score, group.raw_prediction, low, high)
    return results


def evaluate_heldout(model_dir: str | Path = ROOT / "models", report_dir: str | Path = ROOT / "reports") -> dict:
    """Recompute metrics from genuine saved held-out rows; no training occurs here."""
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    path = model_dir / "heldout_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("No held-out predictions found. Run: python -m src.train")
    frame = pd.read_csv(path)
    by_prompt = metrics_by_prompt(frame)
    report = {"by_prompt": by_prompt, "evaluation_essays": int(len(frame))}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute metrics from saved held-out predictions.")
    parser.add_argument("--model-dir", default=str(ROOT / "models")); parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate_heldout(args.model_dir, args.report_dir), indent=2))
    except FileNotFoundError as exc:
        print(f"Evaluation not run: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
