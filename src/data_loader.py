"""Safe loading and schema checks for the licensed, locally supplied ASAP file."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"essay_set", "essay", "domain1_score"}
OPTIONAL_RATER_COLUMNS = {"essay_id", "rater1_domain1", "rater2_domain1", "domain1_score"}


class DatasetSchemaError(ValueError):
    """The supplied file cannot support this project."""


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(c).strip().lower() for c in cleaned.columns]
    return cleaned


def validate_schema(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetSchemaError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing)) +
            ". Expected ASAP-compatible essay_set, essay, and domain1_score."
        )


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download training_set_rel3.tsv from the official ASAP "
            "Kaggle competition page after accepting its terms, then set ASAP_DATA_PATH or config.yaml."
        )
    # pandas' sniffer supports manually downloaded TSV and compatible CSV files.
    frame = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    frame = normalize_columns(frame)
    validate_schema(frame)
    return frame
