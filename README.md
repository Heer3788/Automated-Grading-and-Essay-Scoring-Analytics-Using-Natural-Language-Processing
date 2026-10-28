# Automated Grading and Essay Scoring Analytics Using NLP

## Abstract and scope

This local academic prototype learns patterns from the human-scored **ASAP Automated Essay Scoring** training data and produces a prompt-specific preliminary score estimate. It is for written academic essays only. It is not an examination system and never makes a final grade decision.

> Scores are preliminary model estimates trained on historical human-graded essays. They may be inaccurate or biased, particularly for prompts or writing styles underrepresented in the training data. A qualified educator must review the essay and determine the final grade.

## Objectives

The project validates a licensed local dataset, documents data quality, trains separate estimators per prompt, evaluates them only on held-out essays, and provides a Streamlit interface with a score range and descriptive writing indicators. It deliberately excludes identity, `essay_id`, rater columns, and all human scores from the model inputs.

## Dataset, attribution, and configuration

The intended source is the [ASAP Automated Essay Scoring Kaggle competition](https://www.kaggle.com/competitions/asap-aes/data). Obtain the data through that official page, accept its terms, and follow its attribution/redistribution conditions. This repository neither includes nor downloads student essays, and does not assume Kaggle credentials.

Place the manually downloaded `training_set_rel3.tsv` in `data/raw/`, or configure an absolute path:

```bash
export ASAP_DATA_PATH=/absolute/path/to/training_set_rel3.tsv
```

The loader accepts a compatible TSV or CSV, normalizes headers, and requires `essay_set`, `essay`, and `domain1_score`. The actual row count, columns, missing values, score ranges, rater differences, and exclusions are reported only after the local dataset is inspected. **The data is currently unavailable in this repository; no results or model accuracy have been fabricated.**

The original file may include `essay_id`, `essay_set`, `essay`, `rater1_domain1`, `rater2_domain1`, `domain1_score`, and prompt-dependent domain-2 fields. Exact fields are validated rather than assumed. `domain1_score` is the training target; neither it nor rater scores, ids, or post-assessment fields become features.

## Methods

### Cleaning and exploration

`src.analytics` writes a reproducible quality report and cleaned/exclusion files. Empty text, text below the configured minimum word count, invalid prompts/scores, negative scores, scores outside optional verified `config.yaml` bounds, and exact prompt/text duplicates are excluded with a stated reason. Original text is preserved as `essay_original`; the raw data is never overwritten. Observed score ranges are calculated per prompt and stored with artifacts. Outliers are reported, not automatically removed.

Exploration records prompt counts, missing values, exact duplicates, word-count summary, observed score ranges, and rater differences where both rater fields exist. Word count is descriptive only: it is not evidence that longer essays are better.

### NLP and model selection

The baseline is word TF-IDF (1–2 grams) plus Ridge regression. The alternative combines word TF-IDF with character TF-IDF (3–5 grams) plus Ridge. A separate model is selected for every adequately represented essay prompt using validation MAE. This is preferable to treating prompts with incompatible score scales as one homogeneous task. The model pipeline performs text transformations during both training and prediction.

For each prompt, data is de-duplicated before a reproducible 60/20/20 train/validation/test split. Stratification is used only when score-class counts support it. Selection happens on validation data; the selected estimator is then refit on train+validation and evaluated once on the untouched test split. Small prompts are explicitly skipped.

Predictions are reported as raw model output and as a displayed score. The displayed score is rounded to the nearest whole score category and bounded to that prompt’s verified observed range. Both raw and constrained MAE/RMSE are retained so bounds do not disguise model behavior.

### Evaluation

Reported test metrics are MAE, RMSE, quadratic weighted kappa (QWK), exact agreement, and agreement within one point, overall per prompt rather than as a misleading sole aggregate. QWK is shown as unavailable when either side has insufficient variation. `models/heldout_predictions.csv` contains legitimate held-out identifiers, scores, predictions, and errors after training. No full essays are exposed in aggregate views.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# after manually downloading ASAP and accepting its terms
export ASAP_DATA_PATH=/absolute/path/to/training_set_rel3.tsv
python -m src.analytics        # quality report + processed files
python -m src.train            # compare, select, train, evaluate, save artifacts
python -m src.evaluate         # recompute saved held-out metrics only
streamlit run app/main.py      # teacher-facing interface
pytest -q                      # fixtures only; no dataset download needed
```

The processed-data command is `python -m src.analytics`; training also runs the same cleaning logic. It creates prompt-count, word-count, and descriptive word-count/score plots. Training creates a held-out prediction comparison chart. Models are saved as `models/prompt_<id>.joblib`, with model version, source path, score range, train date, supported prompt, validation comparison, and test metrics.

## Interface

**Essay Assessment** accepts a selected supported prompt and essay, then shows a preliminary score, valid range, normalized percentage, and writing indicators (words, sentences, vocabulary diversity, punctuation, paragraphs, sentence-length variation). Empty or extremely short essays and unavailable models return an actionable message, never a random score. **Score Explanation** explicitly limits interpretation. **Dataset Analytics** shows held-out metrics separate from source-dataset information. **Evaluation Examples** shows only held-out identifiers and score/error fields.

## Limitations and ethics

Historical human scores can contain rater disagreement and bias. ASAP prompts, rubric scales, and writing styles may not generalize to another institution, age group, language variety, or assignment. A test set does not prove fairness. TF-IDF models do not fully assess coherence, argument quality, factual accuracy, grammar, originality, or learning. This project does not infer intelligence or personal traits; detect plagiarism or AI text; grade math, code, multiple-choice work, attendance, or behavior; or generate teacher comments.

Potential improvements include verified prompt rubrics, carefully governed educator feedback loops, additional robustness analysis, and subgroup/fairness studies only where ethically appropriate data and approval exist.

## References

- Kaggle, *The Hewlett Foundation: Automated Essay Scoring* competition dataset and terms: https://www.kaggle.com/competitions/asap-aes/data
- Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision for scaled disagreement.
- scikit-learn documentation: TF-IDF vectorization, Ridge regression, and `cohen_kappa_score`.
