"""FastAPI service for the ticket classifier.

Run from the project root:

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs are served at http://localhost:8000/docs.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ticket_classifier import LABEL_DESCRIPTIONS, LABELS
from ticket_classifier.config import (
    MAX_TEXT_LENGTH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    REVIEW_CONFIDENCE_THRESHOLD,
)
from ticket_classifier.history import recent_predictions, save_prediction
from ticket_classifier.predict import ModelNotFoundError, load_model, predict_text

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="Assigns a support message to a category with a confidence score.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The support message to classify.",
        examples=["I was charged twice for my subscription this month."],
    )
    save_history: bool = Field(
        default=True, description="Store this prediction in the local history database."
    )


class Probability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    label: str
    description: str
    confidence: float
    needs_review: bool
    review_threshold: float
    probabilities: list[Probability]


def _ensure_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    return stripped


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Report whether the service can load the trained model."""
    try:
        load_model()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/labels", tags=["meta"])
def labels() -> list[dict]:
    """List the categories the model can assign."""
    return [{"label": label, "description": LABEL_DESCRIPTIONS[label]} for label in LABELS]


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one support message."""
    text = _ensure_text(request.text)
    try:
        result = predict_text(text)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.save_history:
        save_prediction(text, result)

    return PredictResponse(
        **result,
        review_threshold=REVIEW_CONFIDENCE_THRESHOLD,
    )


@app.get("/history", tags=["prediction"])
def history(limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    """Return the most recent stored predictions, newest first."""
    return recent_predictions(limit=limit)


@app.get("/metrics", tags=["evaluation"])
def metrics() -> dict:
    """Return the evaluation results saved by the last training run."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Run training first.")
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if MODEL_METADATA_PATH.exists():
        data["model_metadata"] = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return data
