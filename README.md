# COMP262 Group Project

This repository contains the Phase 2 sentiment-analysis pipeline, generated artifacts, LLM task helpers, and a lightweight backend for frontend integration.

## Project layout

- [ml/sentiment_models.py](/GroupProject262/ml/sentiment_models.py): main Phase 2 training and evaluation pipeline
- [ml/run_phase2_llm_tasks.py](/GroupProject262/ml/run_phase2_llm_tasks.py): requirement 16 and 17 helper for local Hugging Face summarization and service-response generation
- [ml/PHASE2README.md](GroupProject262/ml/PHASE2README.md): submission-focused Phase 2 run guide
- [backend/README.md](/GroupProject262/backend/README.md): backend API documentation
- [OldPDF/build_report.py](/GroupProject262/OldPDF/build_report.py): PDF report builder for the saved artifacts

## Current ML pipeline

The current saved artifact run uses:

- a balanced real subset with `2996` reviews per sentiment class
- TF-IDF with unigrams and bigrams
- extra numeric features: `review_length`, `exclamation_count`, `uppercase_count`
- a `70/30` split stratified by the `overall` rating field
- two machine-learning models: `Naive Bayes` and `MLP`
- lexicon baselines: `VADER` and `TextBlob`
- a confidence-weighted rating-enhancement step that blends the original star rating with review-derived opinion from `VADER` and `TextBlob`

## Current saved results

From [ml/artifacts/reports/run_metadata.json](/GroupProject262/ml/artifacts/reports/run_metadata.json):

- `Naive Bayes` weighted F1: `0.7026`
- `MLP` weighted F1: `0.7084`
- current best model: `MLP`
- mean original rating: `3.0401`
- mean enhanced rating: `3.1325`

The generated report artifacts are in:

- [ml/artifacts/reports](/GroupProject262/ml/artifacts/reports)
- [ml/artifacts/figures](/GroupProject262/ml/artifacts/figures)
- [ml/artifacts/phase2_report.pdf](/GroupProject262/ml/artifacts/phase2_report.pdf)

## Run the ML pipeline

From the project root:

```powershell
.\.venv\Scripts\python.exe ml\sentiment_models.py
```

This will:

- train and evaluate `VADER`, `TextBlob`, `Naive Bayes`, and `MLP`
- save figures to `ml/artifacts/figures`
- save reports and CSV outputs to `ml/artifacts/reports`
- update the saved joblib model artifacts in `ml/artifacts`

## Run the LLM tasks

For Phase 2 requirements `16` and `17`:

```powershell
.\.venv\Scripts\python.exe ml\run_phase2_llm_tasks.py --device -1
```

This saves:

- [ml/artifacts/reports/llm_outputs.json](/GroupProject262/ml/artifacts/reports/llm_outputs.json)

The default local Hugging Face model is `Qwen/Qwen2.5-0.5B-Instruct`. It was chosen because the assignment requires a locally hosted Hugging Face LLM for summarization and customer-service response generation, and this model is instruction-tuned while still being small enough to run locally more easily than much larger models.

Use `--device 0` if you have a local GPU and the model is already available.

## Rebuild the PDF report

The current report builder lives here:

- [OldPDF/build_report.py](/GroupProject262/OldPDF/build_report.py)

Run it with:

```powershell
.\.venv\Scripts\python.exe OldPDF\build_report.py
```

## Backend

The backend serves the trained artifacts for:

- single-message sentiment classification
- batch CSV/JSON upload classification

Run it with:

```powershell
.\.venv\Scripts\python.exe -m backend.server
```

See [backend/README.md]/GroupProject262/backend/README.md) for API details.
