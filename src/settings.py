"""Configuration and paths. Dataset location is intentionally explicit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    values: dict[str, Any] = {}
    if config_path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to read config.yaml. Install requirements.txt.")
        values = yaml.safe_load(config_path.read_text()) or {}
    values.setdefault("data_path", "data/raw/training_set_rel3.tsv")
    values.setdefault("random_seed", 42)
    values.setdefault("minimum_words", 5)
    values.setdefault("minimum_rows_per_prompt", 20)
    values.setdefault("model_version", "1.0.0")
    override = os.getenv("ASAP_DATA_PATH")
    raw_path = Path(override or values["data_path"])
    values["data_path"] = raw_path if raw_path.is_absolute() else ROOT / raw_path
    return values
