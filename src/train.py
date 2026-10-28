"""Reproducible prompt-specific comparison of lightweight local estimators."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data_loader import load_dataset
from src.evaluate import scoring_metrics
from src.preprocessing import prepare_training_data, score_ranges
from src.settings import ROOT, load_config


def candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "word_tfidf_ridge": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("model", Ridge(alpha=10.0)),
        ]),
        "word_char_tfidf_ridge": Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
            ])),
            ("model", Ridge(alpha=15.0)),
        ]),
    }


def _split_prompt(data: pd.DataFrame, seed: int):
    # 60/20/20; stratification is used only where every score class permits it.
    strat = data.domain1_score if data.domain1_score.value_counts().min() >= 3 else None
    trainval, test = train_test_split(data, test_size=0.20, random_state=seed, stratify=strat)
    strat_tv = trainval.domain1_score if trainval.domain1_score.value_counts().min() >= 2 else None
    train, validation = train_test_split(trainval, test_size=0.25, random_state=seed, stratify=strat_tv)
    return train.copy(), validation.copy(), test.copy()


def train_all(config: dict, model_dir: str | Path = ROOT / "models") -> dict:
    raw = load_dataset(config["data_path"])
    cleaned_result = prepare_training_data(raw, int(config["minimum_words"]), config.get("score_ranges"))
    data = cleaned_result.data
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports" if model_dir.resolve() == (ROOT / "models").resolve() else model_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_ranges = score_ranges(data)
    report: dict = {"models": {}, "skipped_prompts": {}, "dataset_rows_clean": len(data), "exclusions": len(cleaned_result.exclusions)}
    prediction_rows = []
    for prompt, prompt_data in data.groupby("essay_set"):
        prompt_key = str(int(prompt))
        if len(prompt_data) < int(config["minimum_rows_per_prompt"]):
            report["skipped_prompts"][prompt_key] = f"Only {len(prompt_data)} valid rows; needs {config['minimum_rows_per_prompt']}."
            continue
        train, validation, test = _split_prompt(prompt_data, int(config["random_seed"]))
        low, high = all_ranges[prompt_key]
        trials = {}
        for name, model in candidates(int(config["random_seed"])).items():
            model.fit(train.essay, train.domain1_score)
            trials[name] = scoring_metrics(validation.domain1_score, model.predict(validation.essay), low, high)
        chosen_name = min(trials, key=lambda n: trials[n]["mae"])
        chosen = candidates(int(config["random_seed"]))[chosen_name]
        # Refit only after selection; the untouched test set has never influenced either choice or fit.
        chosen.fit(pd.concat([train.essay, validation.essay]), pd.concat([train.domain1_score, validation.domain1_score]))
        raw_predictions = chosen.predict(test.essay)
        test_metrics = scoring_metrics(test.domain1_score, raw_predictions, low, high)
        artifact = {
            "model": chosen, "prompt": int(prompt), "score_range": [low, high], "model_name": chosen_name,
            "model_version": config["model_version"], "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_source": str(config["data_path"]), "validation_metrics": trials, "test_metrics": test_metrics,
        }
        joblib.dump(artifact, model_dir / f"prompt_{prompt_key}.joblib")
        report["models"][prompt_key] = {"selected": chosen_name, "validation": trials, "test": test_metrics, "n_train": len(train), "n_validation": len(validation), "n_test": len(test)}
        rows = test[[c for c in ["essay_id", "essay_set", "domain1_score"] if c in test.columns]].copy()
        rows["raw_prediction"] = raw_predictions
        rows["score_min"], rows["score_max"] = low, high
        prediction_rows.append(rows)
    if not report["models"]:
        raise ValueError("No prompt had enough valid rows to train. See config.yaml minimum_rows_per_prompt.")
    report["score_ranges"] = all_ranges
    (model_dir / "metadata.json").write_text(json.dumps(report, indent=2))
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(model_dir / "heldout_predictions.csv", index=False)
    # A compact comparison plot is deliberately derived only from held-out predictions.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for prompt, group in predictions.groupby("essay_set"):
        ax.scatter(group.domain1_score, group.raw_prediction, alpha=.55, s=18, label=f"Prompt {prompt}")
    lower = min(predictions.domain1_score.min(), predictions.raw_prediction.min())
    upper = max(predictions.domain1_score.max(), predictions.raw_prediction.max())
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="perfect agreement")
    ax.set(xlabel="Human-assigned score", ylabel="Raw model prediction", title="Held-out prediction comparison")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(report_dir / "heldout_prediction_comparison.png", dpi=150); plt.close(fig)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train prompt-specific ASAP essay estimators.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    args = parser.parse_args()
    try:
        report = train_all(load_config(args.config), args.model_dir)
        print(json.dumps(report, indent=2))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training not run: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
