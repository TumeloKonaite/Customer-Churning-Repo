# Customer Churn Prediction Platform

An end-to-end machine learning project that predicts whether a bank customer is likely to churn. It demonstrates the complete path from model training and versioning to a deployed API, React frontend, and production monitoring.

## Architecture

![Customer churn platform high-level architecture](customer-churn-hld.svg)

The system has two main paths:

1. The offline pipeline trains and evaluates models, tracks experiments in DagsHub MLflow, and packages an exact model version for deployment.
2. The online platform serves predictions through FastAPI on Modal, stores events and a transactional export outbox in Neon PostgreSQL, and uses Arize AX for model monitoring and observability.

## What this project demonstrates

- End-to-end ML training with reusable preprocessing and model pipelines
- Experiment tracking, model lineage, and registry workflows with MLflow
- Reproducible deployment using exact model versions and checksum validation
- Single-customer and JSON batch predictions through a typed FastAPI API
- A responsive React interface for interacting with the model
- Persistent prediction, outcome, label, and export-delivery records in PostgreSQL
- Scheduled Arize exports for drift, data quality, and delayed model performance
- Privacy-aware telemetry that excludes direct customer identifiers

## Tech stack

| Area | Technology |
| --- | --- |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| API | FastAPI, Pydantic |
| Machine learning | scikit-learn, pandas, NumPy |
| Experiment tracking | DagsHub MLflow |
| Database | Neon PostgreSQL, Alembic |
| Monitoring | Arize AX Cloud |
| Deployment | Modal, Vercel, GitHub Actions |
| Tooling | uv, pytest, Vitest |

## How predictions work

1. A user submits customer data from the React app or directly to the API.
2. FastAPI validates the request against the model's input contract.
3. The packaged preprocessing pipeline transforms the data and predicts churn probability.
4. The API returns the prediction and records a privacy-safe monitoring event.
5. An hourly worker exports approved production telemetry to Arize.
6. Arize compares production traffic with the approved reference baseline and
   incorporates delayed actual labels when they mature.

Arize export is asynchronous and disabled until a matching, unexpired privacy
approval exists. Operational setup and rollback are in
[the Arize runbook](docs/monitoring/arize-operations.md).

### Main API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service and model readiness |
| `POST` | `/api/predict` | Predict churn for one customer |
| `POST` | `/api/predict/batch` | Predict churn for a JSON batch |

Outcome-ingestion endpoints are protected and are used by trusted monitoring workflows rather than the public frontend.

## Run locally

### Prerequisites

- Python 3.12
- Node.js 20+
- [uv](https://docs.astral.sh/uv/)

### Backend

```bash
uv sync --locked
cp .env.example .env
uv run uvicorn application:app --reload --port 5001
```

The API documentation is available at `http://localhost:5001/docs`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

The frontend is available at `http://localhost:5173`.

### Train a model

```bash
uv run python -m src.train train --config configs/training.yaml
```

Local training works without a tracking server. Add the DagsHub MLflow variables from `.env.example` to enable remote experiment tracking and model registration.

## Test the project

```bash
uv run pytest -q
cd frontend
npm test
npm run build
```

## Repository map

```text
frontend/            React prediction workspace
deployment/          Modal-specific image and deployment resources
src/api/             FastAPI routes and request contracts
src/components/      Data ingestion and preprocessing construction
src/training/        Typed model fitting, evaluation, and selection
src/pipeline/        Training and inference orchestration
src/mlops/           Artifact publication, tracking, registry, and packaging
src/database/        Persistence layer and repositories
src/monitoring/      Arize export and delayed-label workflows
src/workers/         Runtime composition for API and scheduled jobs
configs/             Versioned training and monitoring configuration
migrations/          PostgreSQL schema migrations
tests/               Backend test suite
```

## Further documentation

- [Frontend setup and deployment](frontend/README.md)
- [Monitoring overview](docs/monitoring/README.md)
- [Arize operations](docs/monitoring/arize-operations.md)
- [Outcome labels](docs/monitoring/outcomes-labels-v1.md)
- [Production monitoring contract](docs/monitoring/production-monitoring-contract-v1.md)

## License

This project is available under the [MIT License](LICENSE).
