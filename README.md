# Customer Churn Prediction

This repository has one responsibility: train and serve a customer churn model. It uses FastAPI for local ASGI execution and Modal production deployment.

## Features

- model training and artifact generation
- single-customer churn prediction through JSON or the web form
- batch churn prediction through JSON or CSV
- per-record batch validation with `fail_fast` and `partial` modes
- model health and metadata reporting
- typed OpenAPI documentation with Swagger UI and ReDoc
- Modal deployment with build-time model training

Prediction responses contain model outputs only. The application does not make retention decisions, calculate ROI, generate outreach, or send email.

## Setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Train and run locally

`application.py` exports the FastAPI application as `app`. Run it with Uvicorn after training:

```bash
python -m src.train
uvicorn application:app --host 0.0.0.0 --port 5001
```

The application listens on `http://localhost:5001` by default. `python application.py` is also supported and uses `PORT` (default `5001`) and `UVICORN_RELOAD=1` for auto-reload. `make run PORT=8000` changes the Make target's port. Explicit Uvicorn CLI flags take precedence when using the command above.

Equivalent Make targets are available:

```bash
make train
make run
make test
```

Training reads `dataset/Churn_Modelling.csv` and writes the model, preprocessing objects, schema, metrics, and metadata under `artifacts/`.

## Interactive API documentation

With the application running, FastAPI serves:

- Swagger UI at `http://localhost:5001/docs`
- ReDoc at `http://localhost:5001/redoc`
- the generated OpenAPI document at `http://localhost:5001/openapi.json`

In Swagger UI, expand `POST /api/predict`, select **Try it out**, keep or edit the supplied example, and execute the request. For a JSON batch, use either `POST /api/predict/batch` or its compatibility alias and select `fail_fast` or `partial` in `options.mode`. For `POST /api/batch_predict_csv`, choose a CSV with the file picker and optionally enter `{"mode":"partial"}` in the `options` field.

## Web UI

- `/` — navigation
- `/predictdata` — single-customer prediction form
- `/predictbatch` — CSV batch upload and prediction results

The UI displays only the predicted churn/stay label, churn probability, and validation or model errors.

## HTTP API

### Health

```bash
curl http://localhost:5001/health
```

The response reports service health, artifact readiness in `model_loaded`, and model metadata.

### Single prediction

```bash
curl -X POST http://localhost:5001/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

Successful response shape:

```json
{
  "status": "success",
  "predicted_label": 1,
  "p_churn": 0.82,
  "model_name": "churn_predictor",
  "model_version": "1.0.0",
  "timestamp": "2026-08-17T12:00:00+00:00"
}
```

`p_churn` is `null` when the trained estimator does not provide probabilities.

### JSON batch prediction

POST to `/api/predict/batch` (or `/api/batch_predict`) with a `records` list and an optional mode:

```json
{
  "records": [
    {
      "customer_id": "CUST_001",
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88
    }
  ],
  "options": {"mode": "partial"}
}
```

Each successful result contains exactly `index`, `id`, `predicted_label`, and `p_churn`. The envelope also contains validation errors, a summary, model metadata, and a timestamp. `fail_fast` stops validation at the first invalid record; `partial` scores every valid record and returns row-level errors for invalid records.

### CSV batch prediction

```bash
curl -X POST http://localhost:5001/api/batch_predict_csv \
  -F 'file=@customers.csv' \
  -F 'options={"mode":"partial"}'
```

CSV files require the ten model fields shown above. `customer_id`, `row_id`, or `id` may be included for response passthrough. The maximum batch size is 100.

## Smoke tests

The cross-platform smoke scripts validate `/health`, `/api/predict`, `/docs`, and `/openapi.json` against a local or Modal URL.

```bash
BASE_URL=http://localhost:5001 ./scripts/smoke.sh
```

```powershell
./scripts/smoke.ps1 -BaseUrl "https://your-modal-url"
```

## Modal deployment

Modal is the only production deployment target. Authenticate the Modal CLI, then serve a development deployment or deploy to production:

```bash
modal serve modal_app.py
modal deploy modal_app.py
```

The Modal image installs `requirements.txt`, copies the application, trains the model during image construction, and serves the FastAPI application through Modal's ASGI adapter. Runtime scaling and timeout settings are defined in `modal_app.py`. The deployed URL exposes the same `/docs`, `/redoc`, and `/openapi.json` routes as local development.

GitHub deployment runs on pushes to `main` and manual dispatch. Configure these repository secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

## Dependencies

`pyproject.toml` is the dependency source of truth. `requirements.txt` and `uv.lock` are generated from it. Visualization libraries are isolated in the optional `notebook` extra and are not installed in the production Modal image.

```bash
uv lock
uv export --no-dev --no-emit-project --no-annotate --no-header -o requirements.txt
```

For notebook exploration:

```bash
python -m pip install -e '.[notebook]'
```

## Project layout

```text
application.py             FastAPI app, Pydantic contracts, and HTTP/UI routes
modal_app.py               Modal image build and ASGI deployment
src/train.py               Training command and metadata generation
src/components/            Data ingestion, transformation, and model training
src/pipeline/               Training and prediction pipelines
src/services/              Batch validation and prediction service
templates/                 Prediction-only web UI
scripts/smoke.sh           POSIX smoke test
scripts/smoke.ps1          PowerShell smoke test
tests/                      API, OpenAPI, batch, metrics, and training tests
notebooks/                  Optional exploratory analysis
```

## Model limitations

The model is trained on the included bank churn dataset. Its quality depends on the representativeness and freshness of that data. Predictions should be monitored for drift and evaluated for fairness before use in consequential workflows.

## License

See [LICENSE](LICENSE).
