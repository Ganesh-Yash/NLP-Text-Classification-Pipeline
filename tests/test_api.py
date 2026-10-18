import pytest
from fastapi.testclient import TestClient

from ticket_classifier.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not built yet")


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep API tests from writing into the real history database.
    import ticket_classifier.config as config

    monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "history.db")
    import api.main as main

    monkeypatch.setattr(main, "save_prediction", _make_saver(tmp_path / "history.db"))
    monkeypatch.setattr(main, "recent_predictions", _make_reader(tmp_path / "history.db"))
    return TestClient(main.app)


def _make_saver(path):
    from ticket_classifier.history import save_prediction

    return lambda text, result: save_prediction(text, result, db_path=path)


def _make_reader(path):
    from ticket_classifier.history import recent_predictions

    return lambda limit=20: recent_predictions(limit=limit, db_path=path)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_labels_endpoint(client):
    labels = client.get("/labels").json()
    assert {item["label"] for item in labels} == {
        "billing",
        "technical_issue",
        "account_access",
        "general_enquiry",
    }


def test_predict_returns_label_confidence_and_probabilities(client):
    response = client.post("/predict", json={"text": "I was charged twice this month"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "billing"
    assert 0 <= body["confidence"] <= 1
    assert body["needs_review"] == (body["confidence"] < body["review_threshold"])
    assert len(body["probabilities"]) == 4
    probs = [item["probability"] for item in body["probabilities"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_rejects_blank_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_overlong_text(client):
    from ticket_classifier.config import MAX_TEXT_LENGTH

    response = client.post("/predict", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 422


def test_predict_saves_history_and_history_endpoint_returns_it(client):
    client.post("/predict", json={"text": "I forgot my password"})
    rows = client.get("/history?limit=5").json()
    assert rows and rows[0]["text"] == "I forgot my password"


def test_predict_can_skip_history(client):
    client.post("/predict", json={"text": "unique message for skip test", "save_history": False})
    texts = [row["text"] for row in client.get("/history?limit=200").json()]
    assert "unique message for skip test" not in texts


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "unseen_examples" in response.json()
