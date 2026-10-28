"""Reproducible data quality report; charts are descriptive, never causal claims."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import prepare_training_data, score_ranges
from src.settings import ROOT, load_config


def quality_summary(raw: pd.DataFrame, cleaned: pd.DataFrame, exclusions: pd.DataFrame) -> dict:
    summary = {
        "raw_rows": int(len(raw)), "clean_rows": int(len(cleaned)), "columns": list(raw.columns),
        "missing_values": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "duplicate_full_records": int(raw.duplicated().sum()), "excluded_records": int(len(exclusions)),
        "counts_by_prompt": {str(int(k)): int(v) for k, v in cleaned.essay_set.value_counts().sort_index().items()},
        "score_ranges_observed": score_ranges(cleaned),
        "word_count_summary": {k: round(float(v), 2) for k, v in cleaned.word_count.describe().items()},
    }
    if {"rater1_domain1", "rater2_domain1"}.issubset(raw.columns):
        rated = raw[["rater1_domain1", "rater2_domain1"]].apply(pd.to_numeric, errors="coerce").dropna()
        summary["rater_difference_domain1"] = {"n": int(len(rated)), "mean_absolute_difference": round(float((rated.iloc[:, 0] - rated.iloc[:, 1]).abs().mean()), 3) if len(rated) else None}
    return summary


def create_quality_report(config: dict, output_dir: str | Path = ROOT / "reports") -> dict:
    raw = load_dataset(config["data_path"])
    result = prepare_training_data(raw, int(config["minimum_words"]), config.get("score_ranges"))
    summary = quality_summary(raw, result.data, result.exclusions)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(ROOT / "data" / "processed" / "essays_cleaned.csv", index=False)
    result.exclusions.to_csv(ROOT / "data" / "processed" / "exclusions.csv", index=False)
    lines = ["# ASAP Data Quality Report", "", "Generated locally from the configured licensed dataset.", "", f"- Raw rows: {summary['raw_rows']}", f"- Clean rows: {summary['clean_rows']}", f"- Excluded rows: {summary['excluded_records']}", f"- Full-record duplicates: {summary['duplicate_full_records']}", "", "## Observed score ranges by prompt", "", "| Prompt | Minimum | Maximum | Essays |", "|---|---:|---:|---:|"]
    for p, bounds in summary["score_ranges_observed"].items():
        lines.append(f"| {p} | {bounds[0]} | {bounds[1]} | {summary['counts_by_prompt'].get(p, 0)} |")
    lines += ["", "## Missing values", "", "```json", str(summary["missing_values"]), "```", "", "Exclusions are saved in `data/processed/exclusions.csv`; records are excluded only for missing/invalid essentials, short text, or exact prompt/text duplicates.", "", "Word-count correlations are descriptive and must not be interpreted as evidence that longer essays are better."]
    (output_dir / "data_quality_report.md").write_text("\n".join(lines))
    # Descriptive visualizations are saved as files to keep the report reproducible.
    counts = result.data.essay_set.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4)); counts.plot.bar(ax=ax, color="#3b82b6")
    ax.set(xlabel="Essay prompt", ylabel="Essay count", title="Essays by ASAP prompt")
    fig.tight_layout(); fig.savefig(output_dir / "essays_by_prompt.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); result.data.word_count.plot.hist(ax=ax, bins=30, color="#5aa469")
    ax.set(xlabel="Words per essay", title="Essay word-count distribution")
    fig.tight_layout(); fig.savefig(output_dir / "word_count_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for prompt, group in result.data.groupby("essay_set"):
        ax.scatter(group.word_count, group.domain1_score, alpha=.35, s=12, label=f"Prompt {prompt}")
    ax.set(xlabel="Words per essay", ylabel="Human domain-1 score", title="Word count and score (descriptive only)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output_dir / "word_count_vs_score.png", dpi=150); plt.close(fig)
    return summary


def main() -> int:
    args = argparse.ArgumentParser(); args.add_argument("--config", default=None); ns = args.parse_args()
    print(create_quality_report(load_config(ns.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
