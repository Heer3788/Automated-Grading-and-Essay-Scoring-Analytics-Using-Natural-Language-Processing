import pandas as pd
import pytest

from src.data_loader import DatasetSchemaError, validate_schema
from src.evaluate import constrain_scores, evaluate_heldout, scoring_metrics
from src.predict import PredictionError, predict_essay
from src.preprocessing import prepare_training_data, score_ranges
from src.train import train_all


def fixture_frame():
    return pd.DataFrame({"essay_id": [1, 2, 3, 4, 5], "essay_set": [1, 1, 2, 2, 2], "essay": ["A valid academic essay contains enough meaningful words here.", None, "short", "Another sufficiently detailed response is included here today.", "Another sufficiently detailed response is included here today."], "domain1_score": [2, 3, -1, 1, 1]})


def test_required_schema_validation():
    with pytest.raises(DatasetSchemaError): validate_schema(pd.DataFrame({"essay": ["x"]}))


def test_cleaning_handles_missing_invalid_and_duplicates():
    result = prepare_training_data(fixture_frame(), minimum_words=5)
    assert len(result.data) == 2
    assert set(result.exclusions.reason) == {"missing_or_empty_essay", "too_short", "duplicate_prompt_and_essay"}


def test_negative_score_is_rejected_when_text_is_valid():
    frame = pd.DataFrame({"essay_set": [1], "essay": ["This is a sufficiently long essay with valid looking text."], "domain1_score": [-2]})
    assert prepare_training_data(frame, minimum_words=5).exclusions.reason.iloc[0] == "negative_score"


def test_configured_prompt_range_rejects_outside_score():
    frame = pd.DataFrame({"essay_set": [1], "essay": ["This is a sufficiently long essay with valid looking text."], "domain1_score": [8]})
    result = prepare_training_data(frame, minimum_words=5, configured_ranges={"1": [0, 4]})
    assert result.exclusions.reason.iloc[0] == "outside_configured_prompt_range"


def test_prompt_ranges_are_separate():
    data = pd.DataFrame({"essay_set": [1, 1, 2, 2], "domain1_score": [2, 12, 0, 3]})
    assert score_ranges(data) == {"1": [2, 12], "2": [0, 3]}


def test_score_constraints_and_kappa_edge_case():
    assert constrain_scores([-9, 2.51, 20], 0, 4).tolist() == [0, 3, 4]
    metrics = scoring_metrics([2, 2], [2.1, 2.4], 0, 4)
    assert metrics["qwk"] is None and metrics["mae"] == 0.0


def test_prediction_rejects_empty_and_missing_artifact(tmp_path):
    with pytest.raises(PredictionError, match="at least"):
        predict_essay("", 1, tmp_path)
    with pytest.raises(PredictionError, match="No trained model"):
        predict_essay("One two three four five.", 1, tmp_path)


def test_small_prompt_training_keeps_target_out_of_features(tmp_path):
    rows = []
    for idx in range(30):
        score = idx % 3
        rows.append({"essay_id": idx, "essay_set": 1, "essay": f"This is a sufficiently detailed academic response number {idx} with argument {score}.", "domain1_score": score})
    path = tmp_path / "fixture.csv"; pd.DataFrame(rows).to_csv(path, index=False)
    report = train_all({"data_path": path, "minimum_words": 5, "minimum_rows_per_prompt": 20, "random_seed": 7, "model_version": "test"}, tmp_path / "models")
    assert report["models"]["1"]["n_test"] == 6
    result = predict_essay("This is a sufficiently detailed academic response with an argument today.", 1, tmp_path / "models")
    assert 0 <= result["predicted_score"] <= 2
    assert evaluate_heldout(tmp_path / "models", tmp_path / "reports")["by_prompt"]["1"]["n"] == 6
