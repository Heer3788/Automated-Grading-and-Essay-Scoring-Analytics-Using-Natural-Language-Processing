from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.grading import LIMITATIONS, NOTICE
from src.predict import PredictionError, predict_essay
from src.settings import ROOT

st.set_page_config(page_title="Academic Essay Assessment", layout="wide")
st.title("Academic Essay Assessment")
st.caption("Decision-support prototype for written academic essays — never a final grading system.")
st.warning(NOTICE)
metadata_path = ROOT / "models" / "metadata.json"

if not metadata_path.exists():
    st.error("No model artifacts found. Add the official ASAP file, then run `python -m src.train`.")
    st.code("ASAP_DATA_PATH=/path/to/training_set_rel3.tsv python -m src.train")
    st.stop()

metadata = json.loads(metadata_path.read_text())
prompts = sorted(metadata["models"], key=int)
tab_assess, tab_explain, tab_analytics, tab_examples = st.tabs(["Essay Assessment", "Score Explanation", "Dataset Analytics", "Evaluation Examples"])
with tab_assess:
    prompt = st.selectbox("Essay prompt / assessment type", prompts, format_func=lambda p: f"ASAP prompt {p}")
    essay = st.text_area("Student essay", height=260, placeholder="Paste an academic essay here.")
    if st.button("Generate preliminary estimate", type="primary"):
        try:
            result = predict_essay(essay, prompt)
            a, b, c = st.columns(3)
            a.metric("Preliminary score", result["predicted_score"])
            b.metric("Valid prompt range", f"{result['score_range'][0]}–{result['score_range'][1]}")
            c.metric("Normalized score", f"{result['normalized_percent']}%")
            st.caption(f"Raw model output: {result['raw_prediction']:.2f}; displayed score is rounded to the nearest valid score category.")
            st.json(result["indicators"])
            st.warning("Teacher review is required before assigning any final grade.")
        except PredictionError as exc:
            st.info(str(exc))
with tab_explain:
    st.write("The application reports a model estimate plus descriptive writing statistics. These signals are not causal explanations of educational quality.")
    st.info(LIMITATIONS)
with tab_analytics:
    st.subheader("Held-out evaluation metrics")
    rows = []
    for prompt, info in metadata["models"].items():
        rows.append({"prompt": prompt, "evaluation essays": info["n_test"], **info["test"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Dataset statistics and evaluation metrics describe the external ASAP dataset, not a school or class.")
with tab_examples:
    path = ROOT / "models" / "heldout_predictions.csv"
    if path.exists():
        examples = pd.read_csv(path)
        examples["prediction_error"] = examples["raw_prediction"] - examples["domain1_score"]
        st.dataframe(examples.drop(columns=[c for c in ["essay"] if c in examples]), use_container_width=True)
    else:
        st.info("Held-out examples are created after training.")
