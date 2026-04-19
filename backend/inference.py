from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - fallback if nltk is unavailable
    stopwords = None
    WordNetLemmatizer = None


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
DEFAULT_TEXT_COLUMNS = ("text", "reviewText", "message", "content", "review")


class TextProcessor:
    """Match the training-time preprocessing as closely as possible."""

    def __init__(self) -> None:
        stop_words: set[str] = set()

        if stopwords is not None:
            try:
                stop_words = set(stopwords.words("english"))
            except LookupError:
                stop_words = set()

        self.stop_words = stop_words - {"not", "no"}

        if WordNetLemmatizer is not None:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.lemmatizer.lemmatize("tests")
            except LookupError:
                self.lemmatizer = None
        else:
            self.lemmatizer = None

    def handle_negation(self, text: object) -> str:
        words = str(text or "").split()
        negation_words = {"not", "no", "never", "n't"}

        result: list[str] = []
        negate_window = 0

        for word in words:
            if word.lower() in negation_words:
                negate_window = 2
                result.append(word)
                continue

            if negate_window > 0:
                result.append(f"NOT_{word}")
                negate_window -= 1
            else:
                result.append(word)

        return " ".join(result)

    def clean(self, text: object) -> str:
        lowered = str(text or "").lower()
        lowered = re.sub(r"[^a-zA-Z\s!]", "", lowered)

        words = [word for word in lowered.split() if word not in self.stop_words]

        if self.lemmatizer is not None:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)

    def preprocess_texts(self, texts: Iterable[object]) -> pd.DataFrame:
        df = pd.DataFrame({"reviewText": ["" if text is None else str(text) for text in texts]})
        df["reviewText"] = df["reviewText"].apply(self.handle_negation)
        df["cleaned_text"] = df["reviewText"].apply(self.clean)
        df["review_length"] = df["cleaned_text"].apply(len)
        df["exclamation_count"] = df["reviewText"].str.count("!").fillna(0)
        df["uppercase_count"] = df["reviewText"].apply(
            lambda value: sum(1 for word in str(value).split() if word.isupper())
        )
        return df


@dataclass
class PredictionResult:
    text: str
    nb_sentiment: str
    mlp_sentiment: str
    final_sentiment: str
    models_agree: bool


class SentimentClassifier:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.processor = TextProcessor()
        self.vectorizer = joblib.load(self.artifacts_dir / "tfidf.joblib")
        self.scaler = joblib.load(self.artifacts_dir / "scaler.joblib")
        self.nb_model = joblib.load(self.artifacts_dir / "nb_model.joblib")
        self.mlp_model = joblib.load(self.artifacts_dir / "mlp_model.joblib")
        self.label_encoder = joblib.load(self.artifacts_dir / "label_encoder.joblib")
        self.numeric_cols = ["review_length", "exclamation_count", "uppercase_count"]

    def _vectorize(self, texts: Iterable[object]):
        prepared = self.processor.preprocess_texts(texts)
        x_text = self.vectorizer.transform(prepared["cleaned_text"])
        x_extra = self.scaler.transform(prepared[self.numeric_cols])

        try:
            from scipy.sparse import hstack
        except Exception as exc:  # pragma: no cover - scipy exists in runtime
            raise RuntimeError("scipy is required for feature stacking") from exc

        return hstack([x_text, x_extra]), prepared

    def predict_texts(self, texts: Iterable[object]) -> list[PredictionResult]:
        features, prepared = self._vectorize(texts)
        nb_predictions = self.nb_model.predict(features)
        mlp_predictions = self.label_encoder.inverse_transform(self.mlp_model.predict(features))

        results: list[PredictionResult] = []
        for raw_text, nb_pred, mlp_pred in zip(prepared["reviewText"], nb_predictions, mlp_predictions):
            final_prediction = mlp_pred if nb_pred != mlp_pred else nb_pred
            results.append(
                PredictionResult(
                    text=raw_text,
                    nb_sentiment=str(nb_pred),
                    mlp_sentiment=str(mlp_pred),
                    final_sentiment=str(final_prediction),
                    models_agree=bool(nb_pred == mlp_pred),
                )
            )
        return results

    def classify_frame(self, frame: pd.DataFrame, text_column: str | None = None) -> tuple[pd.DataFrame, str]:
        resolved_column = text_column or resolve_text_column(frame)
        if resolved_column not in frame.columns:
            raise ValueError(f"Text column '{resolved_column}' was not found in the uploaded file.")

        predictions = self.predict_texts(frame[resolved_column].fillna("").astype(str).tolist())
        classified = frame.copy()
        classified["nb_sentiment"] = [item.nb_sentiment for item in predictions]
        classified["mlp_sentiment"] = [item.mlp_sentiment for item in predictions]
        classified["final_sentiment"] = [item.final_sentiment for item in predictions]
        classified["models_agree"] = [item.models_agree for item in predictions]
        return classified, resolved_column


def resolve_text_column(frame: pd.DataFrame) -> str:
    for candidate in DEFAULT_TEXT_COLUMNS:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not infer the text column. Provide one of: "
        + ", ".join(DEFAULT_TEXT_COLUMNS)
        + "."
    )
