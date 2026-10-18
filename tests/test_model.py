import joblib
import pytest

from ticket_classifier.config import DATASET_PATH, LABELS, MODEL_PATH, UNSEEN_DATASET_PATH
from ticket_classifier.data import load_dataset, load_unseen_dataset, train_test_frames
from ticket_classifier.evaluate import compute_metrics
from ticket_classifier.model import build_pipeline
from ticket_classifier.predict import predict_text


def test_dataset_is_balanced_and_valid():
    frame = load_dataset(DATASET_PATH)
    assert set(frame["label"]) == set(LABELS)
    counts = frame["label"].value_counts()
    assert counts.min() == counts.max()
    assert frame["text"].str.len().min() > 0


def test_unseen_dataset_covers_all_labels():
    frame = load_unseen_dataset(UNSEEN_DATASET_PATH)
    assert set(frame["label"]) == set(LABELS)


def test_split_is_stratified_and_disjoint():
    frame = load_dataset(DATASET_PATH)
    train, test = train_test_frames(frame)
    assert len(train) + len(test) == len(frame)
    assert set(test["label"]) == set(LABELS)
    assert not set(train["text"]) & set(test["text"])


def test_pipeline_fits_and_predicts_on_toy_data():
    texts = [
        "I was charged twice on my invoice",
        "refund my payment please",
        "the app crashes on startup",
        "error 500 when I save",
        "I cannot log in to my account",
        "reset my password please",
        "do you have a free trial",
        "what are your opening hours",
    ]
    labels = [
        "billing", "billing",
        "technical_issue", "technical_issue",
        "account_access", "account_access",
        "general_enquiry", "general_enquiry",
    ]
    model = build_pipeline().fit(texts, labels)
    result = predict_text("my invoice is wrong and I was charged", model=model)
    assert result["label"] == "billing"
    assert 0 < result["confidence"] <= 1
    assert abs(sum(p["probability"] for p in result["probabilities"]) - 1) < 1e-6


def test_compute_metrics_matches_perfect_predictions():
    truth = ["billing", "technical_issue", "account_access", "general_enquiry"]
    metrics = compute_metrics(truth, truth)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"]["matrix"][0][0] == 1


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not built yet")
def test_saved_model_meets_minimum_unseen_accuracy():
    frame = load_unseen_dataset(UNSEEN_DATASET_PATH)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(frame["text"])
    accuracy = (predictions == frame["label"]).mean()
    assert accuracy >= 0.75
