"""Loading and splitting the labelled datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_classifier.config import (
    DATASET_PATH,
    LABELS,
    RANDOM_STATE,
    TEST_SIZE,
    UNSEEN_DATASET_PATH,
)


def load_labelled_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with ``text`` and ``label`` columns and validate its labels."""
    frame = pd.read_csv(path).dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"{path} contains unknown labels: {unknown}")
    return frame.reset_index(drop=True)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return load_labelled_csv(path)


def load_unseen_dataset(path: Path = UNSEEN_DATASET_PATH) -> pd.DataFrame:
    return load_labelled_csv(path)


def train_test_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a stratified train/test split with a fixed random seed."""
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
