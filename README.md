# Sentiment Backend

This repository includes a lightweight backend that serves the trained sentiment models saved in `ml/artifacts`.

## Current project status

The machine learning pipeline was updated and retrained before connecting the backend. The main improvements were:

- built a **balanced real subset** of the Amazon Gift Cards dataset using:
  - 2000 Positive reviews
  - 2000 Neutral reviews
  - 2000 Negative reviews
- used a **70/30 train-test split** on that balanced subset
- improved preprocessing by:
  - keeping negation words such as `not`, `no`, `never`, `nor`, `none`, `nothing`, and `nowhere`
  - expanding contractions such as `didn't` → `did not`
  - applying **negation handling** before cleaning so phrases like `not good` become `NOT_good`
  - keeping exclamation marks in the cleaned text to preserve strong sentiment cues
- used TF-IDF with:
  - `max_features=8000`
  - `ngram_range=(1, 2)`
  - `min_df=2`
  - `max_df=0.85`
  - `sublinear_tf=True`
- added extra numeric features:
  - `review_length`
  - `exclamation_count`
  - `uppercase_count`
- trained and compared:
  - VADER
  - TextBlob
  - Naive Bayes
  - MLP

## Latest model scores

All models were evaluated on the same 30% test split.

## What caused the score updates

The scores changed over time because the preprocessing and training pipeline were corrected and improved.

### Earlier issues
Earlier experimental runs produced inflated scores because of data handling problems such as:
- balancing before splitting
- oversampling and augmentation affecting evaluation
- unrealistic test distributions

These issues made results look stronger than they really were.

### What improved the final results
The current scores are more realistic because of the following changes:

1. **Balanced real sampling**
   - Instead of duplicating minority classes, the pipeline now uses real reviews from each class.
   - This created a fairer train/test evaluation setup.

2. **Negation handling**
   - Phrases like `not good`, `not working`, and `not worth` are now represented more clearly for the model.
   - This especially helped Naive Bayes.

3. **Contraction expansion**
   - Examples like `didn't` are converted to `did not` before negation handling.
   - This made negation detection more reliable.

4. **Improved feature engineering**
   - TF-IDF settings were expanded to capture more useful text patterns.
   - Additional numeric features captured emotional intensity and writing style.

5. **Cleaner evaluation**
   - All models now run on the same held-out test split.
   - This gives a valid apples-to-apples comparison.

## Best current model

The best current model by weighted F1 is:

- **MLP**
  - F1: **0.707907**

Although NB remains very competitive, MLP slightly outperformed it in the latest run after the negation and contraction updates.

## What the backend does

- Accepts a single text message and returns sentiment predictions from both models.
- Accepts an uploaded `.csv` or `.json` file, classifies each text row, and returns a new downloadable file.
- Adds these columns to batch outputs:
  - `nb_sentiment`
  - `mlp_sentiment`
  - `final_sentiment`
  - `models_agree`

`final_sentiment` uses the MLP result when the two models disagree.

## API

### `GET /health`

Returns a simple status payload.

### `POST /api/classify-text`

Request body:

```json
{
  "text": "This card arrived quickly and worked great!"
}