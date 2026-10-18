# NLP-Text-Classification-Pipeline

Classifies customer-support messages into categories such as **billing**, **technical issue**, **account access**, and **general enquiry**. Each prediction comes with a confidence score, and low-confidence tickets are flagged for human review.

The project includes:

- a reproducible sample dataset and a separate hand-written evaluation set
- a TF-IDF + Logistic Regression baseline, saved as a single scikit-learn pipeline
- evaluation with accuracy, precision, recall, F1, and a confusion matrix
- a **FastAPI** prediction service
- a **Streamlit** demo UI with a classify page, an evaluation page, and prediction history
- optional prediction history in **SQLite**
- tests for preprocessing, the model, the API, and history

## Quick start

Tested with Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. (Optional) regenerate the sample dataset. It is deterministic.
python data/generate_dataset.py

# 2. Train, evaluate, and save the model. Writes models/ and reports/.
python -m ticket_classifier.train

# 3. Run the API (interactive docs at http://localhost:8000/docs)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Run the demo UI (http://localhost:8501)
streamlit run ui/streamlit_app.py

# 5. Run the tests
pytest
```

The trained model and reports are already committed, so you can skip steps 1 and 2 to run the API or UI straight away.

## API

| Method | Path        | Description                                                      |
|--------|-------------|------------------------------------------------------------------|
| GET    | `/health`   | Checks that the trained model loads.                             |
| GET    | `/labels`   | Lists the categories and what each one covers.                   |
| POST   | `/predict`  | Classifies one message. Returns label, confidence, probabilities, and `needs_review`. |
| GET    | `/history`  | Returns recent saved predictions, newest first (`?limit=20`).    |
| GET    | `/metrics`  | Returns the evaluation results from the last training run.       |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my subscription this month"}'
```

Response (probabilities abbreviated):

```json
{
  "label": "billing",
  "description": "Invoices, charges, refunds, payments and pricing questions.",
  "confidence": 0.87,
  "needs_review": false,
  "review_threshold": 0.5,
  "probabilities": [
    {"label": "billing", "probability": 0.87},
    {"label": "account_access", "probability": 0.05}
  ]
}
```

Pass `"save_history": false` to skip storing a prediction. History is kept in `data/prediction_history.db` by default. Set `TICKET_HISTORY_DB` to use a different file.

## Project layout

```
data/
  generate_dataset.py      Builds the sample dataset (deterministic, seed 42)
  support_tickets.csv      Training and test data (600 rows, 150 per class)
  unseen_examples.csv      24 hand-written messages, never used for training
ticket_classifier/
  config.py                Labels, paths, and thresholds
  preprocessing.py         Text cleaning (lowercase, URL/email/number tokens)
  model.py                 TF-IDF + Logistic Regression pipeline
  data.py                  Loading and stratified splitting
  train.py                 Training, evaluation, and artefact saving
  evaluate.py              Metrics, text report, and confusion-matrix plot
  predict.py               Model loading and single-message prediction
  history.py               SQLite prediction history
api/main.py                FastAPI application
ui/streamlit_app.py        Streamlit demo
models/                    Saved model and metadata
reports/                   Metrics JSON, text report, and confusion-matrix PNG
tests/                     pytest suite
```

## Method

1. **Categories.** Four support categories, each described in `ticket_classifier/config.py`.
2. **Preprocessing.** Lowercases text, replaces URLs and emails with `urltoken` / `emailtoken`, replaces numbers of three or more digits (order and ticket IDs) with `numbertoken`, and removes punctuation and extra whitespace.
3. **Features.** TF-IDF on word unigrams and bigrams, with sublinear term frequency.
4. **Classifier.** Logistic Regression with `class_weight="balanced"`, so smaller classes are not drowned out.
5. **Split.** 80/20 stratified train/test split of the generated dataset, with `random_state=42`.
6. **Out-of-sample check.** The hand-written `unseen_examples.csv` is scored after training. Its wording is different from the generated data, so it is the more realistic estimate of performance.
7. **Review flag.** A prediction whose top probability is below 0.5 is marked `needs_review`.

## Results

Numbers come from the last run of `python -m ticket_classifier.train` and are saved in `reports/metrics.json`.

### Hand-written unseen examples (24 messages, 6 per class)

This is the more realistic estimate.

| Metric            | Score  |
|-------------------|--------|
| Accuracy          | 83.3%  |
| Macro precision   | 88.1%  |
| Macro recall      | 83.3%  |
| Macro F1          | 80.6%  |

| Class            | Precision | Recall | F1    | Support |
|------------------|-----------|--------|-------|---------|
| billing          | 0.857     | 1.000  | 0.923 | 6       |
| technical_issue  | 1.000     | 0.333  | 0.500 | 6       |
| account_access   | 0.667     | 1.000  | 0.800 | 6       |
| general_enquiry  | 1.000     | 1.000  | 1.000 | 6       |

Confusion matrix (rows are true labels, columns are predictions):

| True \ Predicted | billing | technical_issue | account_access | general_enquiry |
|------------------|---------|-----------------|----------------|-----------------|
| billing          | 6       | 0               | 0              | 0               |
| technical_issue  | 1       | 2               | 3              | 0               |
| account_access   | 0       | 0               | 6              | 0               |
| general_enquiry  | 0       | 0               | 0              | 6               |

A PNG version is saved at `reports/confusion_matrix.png`.

### Held-out split of the generated dataset (120 messages)

| Metric          | Score  |
|-----------------|--------|
| Accuracy        | 99.2%  |
| Macro F1        | 99.2%  |

These scores are high because the generated messages reuse a fixed vocabulary. Don't read them as real-world accuracy.

### Review flag on the unseen set

All 4 wrong predictions fall below the 0.5 confidence threshold, so all 4 are flagged for review. The threshold also flags 11 correct predictions, so reviewers would handle 15 of the 24 messages. In this setup, the flag catches every error at the cost of extra review work.

## Limitations

- **Synthetic training data.** The dataset is generated from hand-written issue sentences combined with greetings, sign-offs, and random details such as order numbers and plan names. It is not real customer data. The 99% held-out score overstates real performance, and the 83% hand-written score is the better guide.
- **Weak on technical issues.** Technical-issue messages that avoid common words such as "error", "crash", or "bug" are often mislabelled. In the unseen set, "freezes", "stuck", "times out", and "grey boxes" were read as account access or billing. Adding real support tickets with varied wording is the most effective fix.
- **Small evaluation set.** The unseen set has only 24 messages, so each error moves accuracy by about 4 points. Treat the per-class numbers as rough.
- **Single-label only.** A message that covers two issues, such as a billing error that also blocks login, gets one label.
- **English only.** The cleaning and vocabulary assume English text.
- **No retraining from user feedback.** Corrections are not stored or used to retrain the model.

## Next steps

- Replace or supplement the synthetic data with real, anonymised support tickets (for example, a public customer-support dataset) and keep the hand-written set as a blind test.
- Try a transformer model (for example, a fine-tuned DistilBERT) as an extension, and compare it with the baseline on the same unseen set.
- Log the review flag and reviewer corrections to build a feedback loop.
