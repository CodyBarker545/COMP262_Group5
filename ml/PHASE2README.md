# Phase 2 Guide

This file maps the assignment requirements to the current code, saved artifacts, and the commands you should run before submission.

## Main Phase 2 files

- [sentiment_models.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/sentiment_models.py): training, evaluation, figures, enhanced ratings, and saved model artifacts
- [run_phase2_llm_tasks.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/run_phase2_llm_tasks.py): local Hugging Face tasks for requirements `16` and `17`
- [OldPDF/build_report.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/OldPDF/build_report.py): report builder that assembles the PDF from saved artifacts

## What the current pipeline does

The current implementation covers:

- `11(a)`: selects a balanced real subset from the original dataset
- `11(b)`: preprocesses text with contraction expansion, negation handling, stopword filtering, and lemmatization
- `11(c)`: uses TF-IDF text representation with extra numeric features
- `11(d)`: uses a `70/30` train-test split stratified by `overall` rating
- `11(e)`: trains `Naive Bayes` and `MLP`
- `13`: saves accuracy, precision, recall, F1, and confusion matrices
- `14`: compares lexicon models and ML models on the same held-out test set
- `15`: enhances ratings using review-derived opinion
- `16`: generates 10 summaries for long reviews
- `17`: generates one customer-service response for a question-style review

## Current saved artifact snapshot

From [artifacts/reports/run_metadata.json](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/run_metadata.json):

- subset rows: `8988`
- train rows: `6291`
- test rows: `2697`
- `Naive Bayes` weighted F1: `0.7026`
- `MLP` weighted F1: `0.7084`
- best model: `MLP`

The saved LLM outputs already exist in:

- [artifacts/reports/llm_outputs.json](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/llm_outputs.json)

For requirements `16` and `17`, the project currently uses `Qwen/Qwen2.5-0.5B-Instruct`. The assignment does not require Qwen specifically; it requires a local Hugging Face LLM. This model was chosen because it is instruction-tuned and lightweight enough to be practical for local summarization and response generation.

## Recommended run order

From the project root:

```powershell
.\.venv\Scripts\python.exe ml\sentiment_models.py
.\.venv\Scripts\python.exe ml\run_phase2_llm_tasks.py --device -1
.\.venv\Scripts\python.exe OldPDF\build_report.py
```

If you want to run from inside `ml`:

```powershell
..\.venv\Scripts\python.exe sentiment_models.py
..\.venv\Scripts\python.exe run_phase2_llm_tasks.py --device -1
..\.venv\Scripts\python.exe ..\OldPDF\build_report.py
```

## Outputs to include in the report

The pipeline saves these important outputs:

- [artifacts/reports/model_results.csv](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/model_results.csv)
- [artifacts/reports/classification_reports.txt](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/classification_reports.txt)
- [artifacts/reports/enhanced_ratings.csv](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/enhanced_ratings.csv)
- [artifacts/reports/llm_outputs.json](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/reports/llm_outputs.json)
- [artifacts/figures](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/figures)
- [artifacts/phase2_report.pdf](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/artifacts/phase2_report.pdf)

## Notes for requirement 15

The current code uses a confidence-weighted blended review-opinion method:

- original `overall` rating remains the anchor
- review text is scored with both `VADER` and `TextBlob`
- the combined text score is converted into a `1-5` text-based rating
- text influence is scaled by review confidence
- the final result is saved as `enhanced_rating`

This is implemented in [sentiment_models.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/ml/sentiment_models.py).
