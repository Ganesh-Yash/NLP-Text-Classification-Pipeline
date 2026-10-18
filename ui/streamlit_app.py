"""Streamlit demo: classify a support message and review model evaluation.

Run from the project root:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ticket_classifier import LABEL_DESCRIPTIONS  # noqa: E402
from ticket_classifier.config import (  # noqa: E402
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    REVIEW_CONFIDENCE_THRESHOLD,
    MAX_TEXT_LENGTH,
)
from ticket_classifier.history import recent_predictions, save_prediction  # noqa: E402
from ticket_classifier.predict import ModelNotFoundError, predict_text  # noqa: E402

EXAMPLES = {
    "Billing": "I was charged twice for my subscription this month and need a refund.",
    "Technical issue": "The app crashes every time I open the reports page.",
    "Account access": "I forgot my password and the reset email never arrived.",
    "General enquiry": "Do you offer a discount for non-profit organisations?",
}

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🗂️", layout="wide")
st.title("Support Ticket Classifier")
st.caption("Paste a customer message to get its category, confidence, and review flag.")


def _set_example(text: str) -> None:
    st.session_state["message"] = text


classify_tab, evaluation_tab, history_tab = st.tabs(
    ["Classify", "Model evaluation", "History"]
)

with classify_tab:
    st.write("Try an example:")
    example_columns = st.columns(len(EXAMPLES))
    for column, (name, example) in zip(example_columns, EXAMPLES.items()):
        column.button(name, on_click=_set_example, args=(example,), width="stretch")

    message = st.text_area(
        "Support message",
        key="message",
        height=140,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="e.g. My invoice shows a charge I don't recognise.",
    )
    save = st.checkbox("Save this prediction to history", value=True)

    if st.button("Classify", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                result = predict_text(message.strip())
            except ModelNotFoundError as exc:
                st.error(str(exc))
            else:
                if save:
                    save_prediction(message.strip(), result)
                st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Predicted category")
            st.markdown(f"### `{result['label']}`")
            st.write(result["description"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            if result["needs_review"]:
                st.warning(
                    f"Confidence is below {REVIEW_CONFIDENCE_THRESHOLD:.0%}. "
                    "A person should review this ticket."
                )
            else:
                st.success("Confident prediction.")
        with right:
            st.subheader("Probability by category")
            chart = pd.DataFrame(result["probabilities"]).set_index("label")
            st.bar_chart(chart)

with evaluation_tab:
    if not METRICS_PATH.exists():
        st.info("No evaluation results yet. Run `python -m ticket_classifier.train` first.")
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(f"Model: {metrics['model']} · trained {metrics['trained_at']}")

        st.subheader("Hand-written examples the model never saw in training")
        unseen = metrics["unseen_examples"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{unseen['accuracy']:.1%}")
        cols[1].metric("Macro precision", f"{unseen['macro_precision']:.1%}")
        cols[2].metric("Macro recall", f"{unseen['macro_recall']:.1%}")
        cols[3].metric("Macro F1", f"{unseen['macro_f1']:.1%}")

        per_class = pd.DataFrame(unseen["per_class"]).T.round(3)
        st.dataframe(per_class, width="stretch")

        st.subheader("Held-out split from the generated dataset")
        test = metrics["test_split"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{test['accuracy']:.1%}")
        cols[1].metric("Macro precision", f"{test['macro_precision']:.1%}")
        cols[2].metric("Macro recall", f"{test['macro_recall']:.1%}")
        cols[3].metric("Macro F1", f"{test['macro_f1']:.1%}")
        st.caption(
            "The generated data is template-based, so this split is easier than the "
            "hand-written set above. Treat the hand-written numbers as the more realistic estimate."
        )

        left, right = st.columns([1, 1])
        with left:
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix (unseen examples)")
        with right:
            st.subheader("Classification report")
            if CLASSIFICATION_REPORT_PATH.exists():
                st.code(CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8"), language="text")

        st.subheader("Categories")
        for label, description in LABEL_DESCRIPTIONS.items():
            st.markdown(f"- **`{label}`**: {description}")

with history_tab:
    rows = recent_predictions(limit=50)
    if not rows:
        st.info("No predictions saved yet.")
    else:
        history_frame = pd.DataFrame(rows)
        history_frame["confidence"] = history_frame["confidence"].map(lambda v: f"{v:.1%}")
        st.dataframe(history_frame, width="stretch", hide_index=True)
