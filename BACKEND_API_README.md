# Backend API README

This project includes a Python backend that loads the trained sentiment models from `ml/artifacts` and exposes them as HTTP APIs for your frontend.

The backend supports two main user flows:

1. A user types a single message and gets its sentiment.
2. A user uploads a `.csv` or `.json` file, the backend classifies each row, and returns a new file with sentiment columns added.

## Backend files

- [backend/server.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/backend/server.py): HTTP server and API routes
- [backend/inference.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/backend/inference.py): model loading, preprocessing, and prediction logic
- [tests/test_backend.py](D:/SoftwareEng/SEMESTER6/COMP262-Natural%20Language%20%26%20Recom%20Sys/GroupProject262/tests/test_backend.py): backend tests

## How the backend works

When the server starts, it:

- loads `tfidf.joblib`
- loads `scaler.joblib`
- loads `nb_model.joblib`
- loads `mlp_model.joblib`
- loads `label_encoder.joblib`

For every input text, it:

- applies preprocessing similar to training
- creates TF-IDF and numeric features
- runs both the Naive Bayes model and the MLP model
- returns both predictions
- sets `final_sentiment` to the MLP result if the two models disagree

## API base URL

If you run the server locally with the default settings, the base URL is:

```text
http://127.0.0.1:8000
```

## How to run the backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m backend.server
```

Optional environment variables:

- `SENTIMENT_API_HOST`
- `SENTIMENT_API_PORT`

## API endpoints

### `GET /health`

Purpose:
Checks whether the backend is running and whether the models were loaded.

What it returns:

```json
{
  "status": "ok",
  "models_loaded": true,
  "available_routes": [
    "POST /api/classify-text",
    "POST /api/classify-file"
  ]
}
```

Frontend use:

- call this when your app loads
- if it fails, show "backend unavailable"

Example:

```js
const response = await fetch("http://127.0.0.1:8000/health");
const data = await response.json();
```

### `POST /api/classify-text`

Purpose:
Classifies one text message entered by the user.

Request body:

```json
{
  "text": "This gift card worked perfectly and arrived quickly."
}
```

Response body:

```json
{
  "text": "This gift card worked perfectly and arrived quickly.",
  "nb_sentiment": "Positive",
  "mlp_sentiment": "Positive",
  "final_sentiment": "Positive",
  "models_agree": true
}
```

Fields returned:

- `text`: original user text
- `nb_sentiment`: Naive Bayes prediction
- `mlp_sentiment`: MLP prediction
- `final_sentiment`: label the frontend should show
- `models_agree`: whether both models predicted the same label

Possible labels:

- `Positive`
- `Neutral`
- `Negative`

Frontend example:

```js
async function classifyText(text) {
  const response = await fetch("http://127.0.0.1:8000/api/classify-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to classify text");
  }

  return response.json();
}
```

Example UI usage:

```js
const result = await classifyText(userMessage);
resultLabel.textContent = result.final_sentiment;
details.textContent = `NB: ${result.nb_sentiment}, MLP: ${result.mlp_sentiment}`;
```

### `POST /api/classify-file`

Purpose:
Accepts an uploaded `.csv` or `.json` file, classifies each row, and returns a processed file for download.

Accepted upload types:

- `.csv`
- `.json`

Supported text column names if `text_column` is not provided:

- `text`
- `reviewText`
- `message`
- `content`
- `review`

Form fields:

- `file`: required
- `text_column`: optional
- `output_format`: optional, `csv` or `json`

Columns added to the returned file:

- `nb_sentiment`
- `mlp_sentiment`
- `final_sentiment`
- `models_agree`

Frontend example:

```js
async function classifyFile(file, textColumn = "", outputFormat = "csv") {
  const formData = new FormData();
  formData.append("file", file);

  if (textColumn) {
    formData.append("text_column", textColumn);
  }

  formData.append("output_format", outputFormat);

  const response = await fetch("http://127.0.0.1:8000/api/classify-file", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to classify file");
  }

  return response;
}
```

Download example:

```js
async function handleFileUpload(file) {
  const response = await classifyFile(file, "text", "csv");
  const blob = await response.blob();

  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename=\"(.+)\"/);
  const filename = match ? match[1] : "classified_output.csv";

  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
}
```

## Example input file formats

### CSV

```csv
text
This was amazing
It was okay
I hated it
```

### JSON

```json
[
  { "text": "This was amazing" },
  { "text": "It was okay" },
  { "text": "I hated it" }
]
```

The backend also accepts line-delimited JSON:

```json
{"text":"This was amazing"}
{"text":"It was okay"}
{"text":"I hated it"}
```

## Main backend functions

### In `backend/inference.py`

#### `TextProcessor`

Purpose:
Preprocesses raw text before prediction.

It:

- handles negation
- cleans text
- builds numeric features

#### `SentimentClassifier`

Purpose:
Main inference service used by the API routes.

Important methods:

- `predict_texts(texts)`: classify a list of strings
- `classify_frame(frame, text_column=None)`: classify every row in a dataframe

#### `resolve_text_column(frame)`

Purpose:
Finds the text column in an uploaded file.

### In `backend/server.py`

#### `run_server(host, port)`

Starts the HTTP server.

#### `parse_json_or_lines(data)`

Reads normal JSON or line-delimited JSON.

#### `load_uploaded_frame(filename, payload)`

Reads uploaded CSV or JSON into a pandas dataframe.

#### `dump_frame(frame, file_format)`

Converts the processed dataframe back into CSV or JSON for download.

#### `parse_multipart(content_type, body)`

Reads `multipart/form-data` requests from the frontend.

## Frontend integration flow

Use these backend calls:

1. `POST /api/classify-text` for a text form
2. `POST /api/classify-file` for a file upload form

Typical flow:

1. User types text or selects a file.
2. Frontend sends `fetch` request to backend.
3. Backend returns JSON or a downloadable file.
4. Frontend shows the label or starts the download.

## CORS

The backend currently allows:

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type`

That means you can connect a separate frontend during development without extra backend changes.

## Error responses

The backend returns errors like:

```json
{
  "error": "Request body must include a non-empty 'text' value."
}
```

Frontend recommendation:

- check `response.ok`
- if false, read `await response.json()`
- show `error.error` in the UI

## Testing

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
