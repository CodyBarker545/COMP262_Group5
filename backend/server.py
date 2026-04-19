from __future__ import annotations

import io
import json
import os
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from backend.inference import SentimentClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
HOST = os.getenv("SENTIMENT_API_HOST", "127.0.0.1")
PORT = int(os.getenv("SENTIMENT_API_PORT", "8000"))
CLASSIFIER = SentimentClassifier()


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def parse_json_or_lines(data: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    try:
        return pd.read_json(buffer)
    except ValueError:
        buffer.seek(0)
        return pd.read_json(buffer, lines=True)


def load_uploaded_frame(filename: str, payload: bytes) -> tuple[pd.DataFrame, str]:
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(payload)), "csv"
    if lower_name.endswith(".json"):
        return parse_json_or_lines(payload), "json"
    raise ValueError("Only .csv and .json uploads are supported.")


def dump_frame(frame: pd.DataFrame, file_format: str) -> tuple[bytes, str]:
    if file_format == "csv":
        output = io.StringIO()
        frame.to_csv(output, index=False)
        return output.getvalue().encode("utf-8"), "text/csv; charset=utf-8"
    if file_format == "json":
        output = frame.to_json(orient="records", indent=2)
        return output.encode("utf-8"), "application/json; charset=utf-8"
    raise ValueError(f"Unsupported output format '{file_format}'.")


def parse_multipart(content_type: str, body: bytes) -> dict[str, dict[str, str | bytes | None]]:
    parser_input = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    message = BytesParser(policy=default).parsebytes(parser_input)

    if not message.is_multipart():
        raise ValueError("Expected a multipart/form-data request.")

    fields: dict[str, dict[str, str | bytes | None]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fields[name] = {
            "filename": part.get_filename(),
            "value": part.get_payload(decode=True),
            "content_type": part.get_content_type(),
        }
    return fields


class SentimentRequestHandler(BaseHTTPRequestHandler):
    server_version = "SentimentAPI/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "models_loaded": True,
                    "available_routes": [
                        "POST /api/classify-text",
                        "POST /api/classify-file",
                    ],
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/classify-text":
            self._handle_classify_text()
            return
        if parsed.path == "/api/classify-file":
            self._handle_classify_file(parsed.query)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def _handle_classify_text(self) -> None:
        try:
            body = self._read_body()
            payload = json.loads(body.decode("utf-8"))
            text = str(payload.get("text", "")).strip()

            if not text:
                raise ValueError("Request body must include a non-empty 'text' value.")

            prediction = CLASSIFIER.predict_texts([text])[0]
            self._send_json(
                HTTPStatus.OK,
                {
                    "text": text,
                    "nb_sentiment": prediction.nb_sentiment,
                    "mlp_sentiment": prediction.mlp_sentiment,
                    "final_sentiment": prediction.final_sentiment,
                    "models_agree": prediction.models_agree,
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Body must be valid JSON."})
        except Exception as exc:  # pragma: no cover - defensive path
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_classify_file(self, query: str) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                raise ValueError("Use multipart/form-data with a file field named 'file'.")

            body = self._read_body()
            form = parse_multipart(content_type, body)
            file_field = form.get("file")
            if not file_field or not file_field.get("filename") or file_field.get("value") is None:
                raise ValueError("Upload a file in the 'file' form field.")

            filename = str(file_field["filename"])
            payload = bytes(file_field["value"])
            frame, source_format = load_uploaded_frame(filename, payload)

            query_values = parse_qs(query)
            requested_column = _decode_field(form.get("text_column")) or query_values.get("text_column", [None])[0]
            requested_format = (
                _decode_field(form.get("output_format"))
                or query_values.get("output_format", [source_format])[0]
                or source_format
            ).lower()

            classified, resolved_column = CLASSIFIER.classify_frame(frame, requested_column)
            response_body, response_type = dump_frame(classified, requested_format)
            output_filename = f"classified_{Path(filename).stem}.{requested_format}"

            self.send_response(HTTPStatus.OK)
            self._send_cors_headers()
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Disposition", f'attachment; filename="{output_filename}"')
            self.send_header("X-Text-Column", resolved_column)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive path
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _decode_field(field: dict[str, str | bytes | None] | None) -> str | None:
    if not field:
        return None
    value = field.get("value")
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8").strip() or None
    return str(value).strip() or None


def run_server(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), SentimentRequestHandler)
    print(f"Sentiment backend listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
