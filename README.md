# Customer Churn Prediction

This service trains one fitted scikit-learn pipeline locally, records eligible runs and models in DagsHub MLflow, packages one exact registered version for Modal inference, and uses Neon PostgreSQL as the operational database foundation. Evidently monitoring and prediction persistence are intentionally deferred.

## Responsibility boundaries

| Component | Responsibility |
| --- | --- |
| Local environment | Training, evaluation, privacy-safe references, publication, validation, packaging |
| DagsHub MLflow | Runs, metrics, lineage, contracts, references, model artifacts, numeric registry versions |
| Neon PostgreSQL | Future deployment, prediction, outcome, label, and monitoring-run state |
| Modal | Single and batch inference from a prepackaged model only |
| Evidently | Future data quality, drift, and delayed-performance calculations |

Modal never trains, refits, registers, selects `latest`, or contacts DagsHub during a prediction. There is no self-hosted MLflow server, separate MLflow database/bucket, or custom content-addressed registry.

## Install

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`pyproject.toml` is the dependency source of truth. Regenerate pinned files with:

```bash
uv lock
uv export --no-dev --no-emit-project --no-annotate --no-header -o requirements.txt
```

## Local training and optional tracking

Training always runs locally through the existing `DataIngestion`, `ModelTrainer`, and `TrainingPipeline` components:

```bash
python -m src.train train --config configs/training.yaml
```

`ModelTrainer` fits every classifier configured in `configs/training.yaml`, selects the best validation ROC AUC, and evaluates only that winner on the untouched test cohort. It then writes the selected pipeline, model comparison, schema, metrics, configuration, contracts, and privacy-safe references to `artifacts/training`. This local result does not depend on MLflow or DagsHub.

Remote tracking is disabled by default:

```bash
ENABLE_DAGSHUB_TRACKING=false python -m src.train train --config configs/training.yaml
```

Enable DagsHub tracking without hard-coding its tracking URI:

```bash
ENABLE_DAGSHUB_TRACKING=true \
DAGSHUB_REPO_OWNER=your-owner \
DAGSHUB_REPO_NAME=your-repository \
DAGSHUB_TOKEN=your-token \
python -m src.train train --config configs/training.yaml
```

`dagshub.init(..., mlflow=True)` selects DagsHub as the only remote backend; all experiment logging still goes through MLflow. A tracking outage produces a warning and leaves the completed local artifacts intact. There is no separate local or generic MLflow tracking mode.

Production registration is deliberately stricter. Add `ENABLE_MODEL_REGISTRATION=true` to the DagsHub command. Registration or verification failure then exits non-zero, and a successful run prints the exact numeric `churn_predictor` version, source run ID, application identity, and checksum. Open the repository’s MLflow tab in DagsHub to inspect experiments, runs, metrics, artifacts, and registered versions.

`RowNumber`, `CustomerId`, `Surname`, email/phone/address aliases, and case variants are rejected before references are written. Mutable local files are useful development outputs, but they are not production identities and Modal never deploys them directly.

## Validate and package an exact model

These commands also use the DagsHub settings shown above, including `ENABLE_DAGSHUB_TRACKING=true`.

Only a positive numeric registry version is accepted:

```bash
python -m src.mlops validate-model \
  --model-uri models:/churn_predictor/7 \
  --output json
```

The validator resolves the source run, checks parameters, metrics, tags, signature, input example, contracts, reference purposes/privacy, checksum, model loading, and a smoke prediction. It rejects `latest`, aliases such as `@champion`, and stages such as `/Production`.

The immutable application identity is:

```text
dagshub:<owner>/<repository>:churn_predictor:<numeric-version>
```

Prepare the inference-only Modal directory before deployment:

```bash
python -m src.mlops prepare-deployment \
  --model-uri models:/churn_predictor/7 \
  --output-dir build/model \
  --expected-run-id "$EXPECTED_MLFLOW_RUN_ID" \
  --expected-model-version-id "$EXPECTED_MODEL_VERSION_ID" \
  --expected-pipeline-sha256 "$EXPECTED_PIPELINE_SHA256" \
  --output json
```

`build/model` contains the MLflow model, feature contract, and `deployment_metadata.json`. It excludes training/evaluation/reference rows and all DagsHub/Neon credentials. Serialized sklearn files are executable trusted artifacts: package only a model from the configured repository after validation.

## Neon database foundation

In the Neon console:

1. Create a project and the `churn_monitoring` database.
2. Create a dedicated least-privilege runtime role; do not use the project owner from Modal.
3. Grant it connect privileges now and only the table privileges introduced by later migrations.
4. Copy the pooled connection string and use the psycopg SQLAlchemy scheme with mandatory SSL:

```text
DATABASE_URL=postgresql+psycopg://<role>:<password>@<pooled-host>/churn_monitoring?sslmode=require
DATABASE_CONNECT_TIMEOUT_SECONDS=10
DATABASE_POOL_SIZE=2
DATABASE_MAX_OVERFLOW=1
```

The small per-container pool is intentional because Modal may scale to many containers. Production configuration rejects missing URLs, SQLite, local hosts, and PostgreSQL URLs without `sslmode=require`; it never silently falls back.

Run the secret-safe check and the baseline migration:

```bash
python -m src.database check --output json
python -m alembic upgrade head
```

Alembic is the only supported production schema-management mechanism. The baseline establishes migration ownership but creates none of the future operational tables. Under `APP_ENV=test`, Alembic rejects remote targets to prevent tests from reaching production.

Rotate the runtime credential by creating/rotating the Neon role password, updating local/GitHub/Modal secret stores, redeploying, running the connectivity check, and then revoking the old credential. Never print the full URL.

## Modal deployment

Create a Modal secret named `customer-churn-production` containing only inference/runtime settings:

```text
APP_ENV=production
DATABASE_URL
DATABASE_CONNECT_TIMEOUT_SECONDS
DATABASE_POOL_SIZE
DATABASE_MAX_OVERFLOW
MLFLOW_REGISTERED_MODEL_NAME=churn_predictor
MLFLOW_MODEL_VERSION=7
EXPECTED_MLFLOW_RUN_ID=<run-id>
EXPECTED_MODEL_VERSION_ID=dagshub:<owner>/<repository>:churn_predictor:7
EXPECTED_PIPELINE_SHA256=<digest>
```

Do not add DagsHub credentials to this Modal secret. They are used locally or in the protected GitHub deployment environment only to prepare `build/model` before `modal deploy`.

```bash
modal deploy modal_app.py
```

At container startup, Modal verifies the package, expected model/run/application identity, checksum, feature-schema version, deserialization, smoke prediction, typed production database configuration, and database connectivity before returning the FastAPI application. Individual requests load the local package and make no DagsHub call.

The GitHub deployment workflow requires protected production secrets for Modal authentication, DagsHub download, and every expected model identity value. Configure the same runtime identity and Neon values in the Modal secret.

## API and local serving

Production inference must use a verified package prepared from an exact registered version.

```bash
uvicorn application:app --host 0.0.0.0 --port 5001
```

Endpoints:

- `GET /health`
- `POST /api/predict`
- `POST /api/predict/batch` (and compatibility alias `/api/batch_predict`)
- `POST /api/batch_predict_csv`
- `/docs`, `/redoc`, and `/openapi.json`

Verified health metadata exposes `deployment_id`, model name, exact numeric version, `model_version_id`, source MLflow run, and feature-schema version. Single responses and batch metadata expose `deployment_id` and `model_version_id`. No fallback version is fabricated when verified metadata is unavailable.

## Rollback

Rollback is a new deployment, not an alias change or runtime hot swap:

1. Choose a historical numeric `churn_predictor` version in DagsHub.
2. Run `validate-model` for that exact URI.
3. Update all expected identity/checksum values.
4. Prepare a fresh `build/model`; this generates a new `deployment_id`.
5. Update the Modal secret identity and redeploy.
6. Confirm `/health` reports the intended numeric model version and new deployment ID.

## Security and data publication

Allowed DagsHub artifacts are approved raw model features, labelled evaluation reference rows, deterministic dataset identities, aggregate evaluation results, versioned contracts, and the fitted pipeline. Never upload `.env` files, credentials, connection strings, raw identifiers, unrestricted source datasets, or production prediction payloads. Logs must contain identity and durations—not raw features, reference rows, authentication headers, or URLs with credentials.

## Tests

```bash
pytest -q
```

Unit tests do not access DagsHub or Neon. External integration tests are opt-in and skip unless their dedicated credentials are supplied. The production monitoring rules remain documented in [docs/monitoring/production-monitoring-contract-v1.md](docs/monitoring/production-monitoring-contract-v1.md); Evidently execution and scheduling are outside this foundation.
