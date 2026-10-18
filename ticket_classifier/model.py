"""Model definition: TF-IDF features feeding a Logistic Regression classifier."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticket_classifier.preprocessing import clean_text
from ticket_classifier.config import RANDOM_STATE


def build_pipeline() -> Pipeline:
    """Return an untrained text-classification pipeline.

    The pipeline contains the text cleaning step, so the saved model accepts
    raw text and can be used directly by the API and the UI.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
