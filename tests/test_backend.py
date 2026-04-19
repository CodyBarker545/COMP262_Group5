from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import pandas as pd

from backend.inference import SentimentClassifier, resolve_text_column
from backend.server import dump_frame, load_uploaded_frame, parse_json_or_lines


class BackendInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = SentimentClassifier()

    def test_single_text_prediction_returns_expected_keys(self) -> None:
        result = self.classifier.predict_texts(["This gift card worked perfectly and arrived fast!"])[0]

        self.assertIn(result.nb_sentiment, {"Positive", "Neutral", "Negative"})
        self.assertIn(result.mlp_sentiment, {"Positive", "Neutral", "Negative"})
        self.assertIn(result.final_sentiment, {"Positive", "Neutral", "Negative"})
        self.assertIsInstance(result.models_agree, bool)

    def test_classify_frame_appends_prediction_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "text": [
                    "I absolutely loved it.",
                    "It was okay, nothing special.",
                    "Terrible experience and waste of money.",
                ]
            }
        )

        classified, resolved_column = self.classifier.classify_frame(frame)

        self.assertEqual(resolved_column, "text")
        self.assertIn("nb_sentiment", classified.columns)
        self.assertIn("mlp_sentiment", classified.columns)
        self.assertIn("final_sentiment", classified.columns)
        self.assertIn("models_agree", classified.columns)
        self.assertEqual(len(classified), 3)


class BackendFileParsingTests(unittest.TestCase):
    def test_resolve_text_column_uses_known_names(self) -> None:
        frame = pd.DataFrame({"message": ["hello"]})
        self.assertEqual(resolve_text_column(frame), "message")

    def test_parse_json_lines_payload(self) -> None:
        payload = b'{"text":"good"}\n{"text":"bad"}\n'
        frame = parse_json_or_lines(payload)
        self.assertEqual(frame["text"].tolist(), ["good", "bad"])

    def test_csv_round_trip_dump(self) -> None:
        original = pd.DataFrame({"text": ["hello"], "final_sentiment": ["Positive"]})
        dumped, content_type = dump_frame(original, "csv")
        round_trip = pd.read_csv(io.BytesIO(dumped))

        self.assertEqual(content_type, "text/csv; charset=utf-8")
        self.assertEqual(round_trip.to_dict(orient="records"), original.to_dict(orient="records"))

    def test_json_upload_loader(self) -> None:
        payload = json.dumps([{"text": "nice"}, {"text": "awful"}]).encode("utf-8")
        frame, file_format = load_uploaded_frame("reviews.json", payload)

        self.assertEqual(file_format, "json")
        self.assertEqual(frame["text"].tolist(), ["nice", "awful"])


if __name__ == "__main__":
    unittest.main()
