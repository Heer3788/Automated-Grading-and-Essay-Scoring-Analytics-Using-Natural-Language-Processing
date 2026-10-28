# Methodology: Automated Grading and Essay Scoring Analytics

## Problem statement

Estimate the historical human-assigned score of an academic essay for a selected ASAP prompt, while keeping the educator responsible for final assessment.

## Data and cleaning

The project accepts only an official manually obtained ASAP TSV/CSV. It checks the essential schema before processing. Cleaning is deterministic and auditable: retain original text, normalize whitespace and Unicode for input text, safely parse numeric fields, exclude missing/empty or very short essays, invalid/negative labels, and exact prompt-text duplicates. Exclusion reasons are saved. Score ranges are observed and reported per prompt; no range is silently shared across prompts.

## Features and estimators

The baseline uses word unigram/bigram TF-IDF and Ridge. The alternative adds character word-boundary 3–5 grams. Descriptive indicators include word/character/sentence/paragraph counts, vocabulary diversity, punctuation, and sentence-length variation. These indicators are for review, not target features or claims about merit. Labels, raters, identifiers, and other post-assessment information are absent from model features.

## Experimental procedure

Each viable prompt gets a reproducible train/validation/test split. Validation selects the candidate; test data remains untouched. Final output includes raw prediction and a rounded, range-bounded score; both raw and constrained errors are reported. Metrics are MAE, RMSE, QWK, exact agreement, and within-one agreement, separately by prompt.

## Results status

Pending local acquisition of the official dataset. No dataset statistics, trained models, or experimental results are claimed in advance.

## Ethical position

The interface labels every output as preliminary. It neither finalizes grades nor asserts objectivity, fairness, comprehensive writing understanding, or causal explanations. Educators must review individual essays and use applicable rubrics.
