from ticket_classifier.history import recent_predictions, save_prediction


def _result(label="billing", confidence=0.9, needs_review=False):
    return {"label": label, "confidence": confidence, "needs_review": needs_review}


def test_save_and_read_back(tmp_path):
    db = tmp_path / "h.db"
    new_id = save_prediction("hello", _result(), db_path=db)
    rows = recent_predictions(limit=5, db_path=db)
    assert rows[0]["id"] == new_id
    assert rows[0]["predicted_label"] == "billing"
    assert rows[0]["needs_review"] is False


def test_limit_and_newest_first(tmp_path):
    db = tmp_path / "h.db"
    for i in range(5):
        save_prediction(f"msg {i}", _result(), db_path=db)
    rows = recent_predictions(limit=3, db_path=db)
    assert [row["text"] for row in rows] == ["msg 4", "msg 3", "msg 2"]
