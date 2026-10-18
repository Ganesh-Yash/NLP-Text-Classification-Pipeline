"""Project-wide constants: labels, file locations, and model settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "support_tickets.csv"
UNSEEN_DATASET_PATH = DATA_DIR / "unseen_examples.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# Override with the TICKET_HISTORY_DB environment variable to store history elsewhere.
HISTORY_DB_PATH = Path(os.environ.get("TICKET_HISTORY_DB", str(DATA_DIR / "prediction_history.db")))

LABELS: list[str] = ["billing", "technical_issue", "account_access", "general_enquiry"]

LABEL_DESCRIPTIONS: dict[str, str] = {
    "billing": "Invoices, charges, refunds, payments and pricing questions.",
    "technical_issue": "Bugs, errors, crashes, outages and performance problems.",
    "account_access": "Logins, passwords, two-factor authentication and permissions.",
    "general_enquiry": "Product information, pre-sales questions and other general requests.",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Predictions below this top-class probability are flagged for human review.
REVIEW_CONFIDENCE_THRESHOLD = 0.5

MAX_TEXT_LENGTH = 5000
