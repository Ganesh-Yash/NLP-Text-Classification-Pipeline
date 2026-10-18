"""Load the trained model and make predictions with confidence scores."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from ticket_classifier.config import (
    LABEL_DESCRIPTIONS,
    MODEL_PATH,
    REVIEW_CONFIDENCE_THRESHOLD,
)


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model file has not been created yet."""


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    model_path = path or str(MODEL_PATH)
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise ModelNotFoundError(
            f"No trained model at {model_path}. Run `python -m ticket_classifier.train` first."
        ) from exc


def predict_text(text: str, model=None) -> dict[str, Any]:
    """Classify one piece of text.

    Returns the predicted label, its confidence, the full probability
    distribution, and whether the prediction should be reviewed by a person.
    """
    model = model or load_model()
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best_index = int(probabilities.argmax())
    label = str(classes[best_index])
    confidence = float(probabilities[best_index])

    ranked = sorted(
        (
            {"label": str(cls), "probability": float(prob)}
            for cls, prob in zip(classes, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "label": label,
        "description": LABEL_DESCRIPTIONS.get(label, ""),
        "confidence": confidence,
        "needs_review": confidence < REVIEW_CONFIDENCE_THRESHOLD,
        "probabilities": ranked,
    }
