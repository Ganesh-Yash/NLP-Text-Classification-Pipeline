"""Train, evaluate and persist the ticket classifier.

Run from the project root:

    python -m ticket_classifier.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import sklearn

from ticket_classifier.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    LABELS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    UNSEEN_DATASET_PATH,
)
from ticket_classifier.data import load_dataset, load_unseen_dataset, train_test_frames
from ticket_classifier.evaluate import compute_metrics, plot_confusion_matrix, text_report
from ticket_classifier.model import build_pipeline


def train_and_evaluate() -> dict:
    dataset = load_dataset(DATASET_PATH)
    train_df, test_df = train_test_frames(dataset)
    unseen_df = load_unseen_dataset(UNSEEN_DATASET_PATH)

    model = build_pipeline()
    model.fit(train_df["text"], train_df["label"])

    # Held-out split from the generated dataset (in-distribution).
    test_pred = model.predict(test_df["text"])
    test_metrics = compute_metrics(test_df["label"], test_pred)

    # Hand-written examples never used for training or tuning (out-of-sample).
    unseen_pred = model.predict(unseen_df["text"])
    unseen_metrics = compute_metrics(unseen_df["label"], unseen_pred)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "TF-IDF (1-2 grams) + Logistic Regression (class_weight=balanced)",
        "labels": LABELS,
        "dataset": {
            "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "total_rows": int(len(dataset)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "unseen_rows": int(len(unseen_df)),
        },
        "test_split": test_metrics,
        "unseen_examples": unseen_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_sections = [
        "Test split (generated dataset, stratified 20% hold-out)",
        "=" * 60,
        text_report(test_df["label"], test_pred),
        "",
        "Unseen hand-written examples",
        "=" * 60,
        text_report(unseen_df["label"], unseen_pred),
    ]
    CLASSIFICATION_REPORT_PATH.write_text("\n".join(report_sections), encoding="utf-8")

    plot_confusion_matrix(
        unseen_df["label"],
        unseen_pred,
        CONFUSION_MATRIX_PATH,
        title="Confusion matrix: unseen hand-written examples",
    )

    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "trained_at": metrics["trained_at"],
                "scikit_learn_version": sklearn.__version__,
                "training_rows": int(len(train_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    test = metrics["test_split"]
    unseen = metrics["unseen_examples"]
    print("Saved model to", MODEL_PATH)
    print(
        f"Test split  : accuracy={test['accuracy']:.3f} "
        f"macro_f1={test['macro_f1']:.3f} weighted_f1={test['weighted_f1']:.3f}"
    )
    print(
        f"Unseen set  : accuracy={unseen['accuracy']:.3f} "
        f"macro_f1={unseen['macro_f1']:.3f} weighted_f1={unseen['weighted_f1']:.3f}"
    )
    print("Reports written to", REPORTS_DIR)


if __name__ == "__main__":
    main()
