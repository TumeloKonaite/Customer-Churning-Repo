# Customer Churn Prediction

This service trains one fitted scikit-learn pipeline locally, records eligible runs and models in DagsHub MLflow, packages one exact registered version for Modal inference, and uses Neon PostgreSQL for operational state. Scheduled Modal workers run reproducible Evidently data-quality and drift reports outside prediction requests.

## Responsibility boundaries

| Component | Responsibility |
| --- | --- |
| Local environment | Training, evaluation, privacy-safe references, publication, validation, packaging |
| DagsHub MLflow | Runs, metrics, lineage, contracts, references, model artifacts, numeric registry versions |
| Neon PostgreSQL | Prediction-event cursors plus immutable monitoring policy, baseline, and run associations |
| Modal | Inference plus separate scheduled/manual monitoring workers |
| Evidently | Offline data quality and drift reports only; never imported by the prediction path |

Modal never trains, refits, registers, selects `latest`, or contacts DagsHub during a prediction. There is no self-hosted MLflow server, separate MLflow database/bucket, or custom content-addressed registry.

## Install

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Churn Insight React frontend

The browser application lives in [`frontend/`](frontend/). It provides the live service overview, single-customer assessment, JSON batch assessment (up to 100 records), partial and fail-fast processing, and row-level errors. The FastAPI service remains the only backend.

Prerequisites are Node.js 20+ and npm 10+. Start it locally with:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

`VITE_API_BASE_URL` selects the API origin and defaults to `http://localhost:5001` when absent. The checked-in example points to the deployed Modal API. Every `VITE_*` value is public at build time: never place credentials, `OUTCOME_INGESTION_API_KEY`, or other secrets there. Customer form values are not persisted in browser storage or sent to analytics.

Run the frontend checks and create the production assets with:

```bash
cd frontend
npm test
npm run build
npm run preview
```

Deploy `frontend/dist` on a static Vite-compatible host. Set the host’s base directory to `frontend`, build command to `npm run build`, output directory to `dist`, and configure `VITE_API_BASE_URL` before building. Vercel, Netlify, and Cloudflare Pages SPA fallbacks are included in `frontend/vercel.json`, `frontend/netlify.toml`, and `frontend/public/_redirects`; unknown routes must serve `index.html` so `/predict` and `/batch` work after a direct refresh. See [`frontend/README.md`](frontend/README.md) for the deployment checklist.

### Frontend CORS configuration

The API reads a comma-separated exact-origin allowlist from `FRONTEND_ALLOWED_ORIGINS`. Development and test add `http://localhost:5173` and `http://127.0.0.1:5173`; production does not add local origins and rejects `*`.

```text
APP_ENV=production
FRONTEND_ALLOWED_ORIGINS=https://churn.example.com,https://operations.example.com
```

Only `GET` and `POST` methods and the `Accept` and `Content-Type` request headers are permitted cross-origin. Add the final frontend origin to the Modal environment and redeploy the backend before production smoke testing. Do not bypass a missing origin with browser `no-cors` mode.

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

Production registration is deliberately stricter. Add `ENABLE_MODEL_REGISTRATION=true` to the DagsHub command. Registration or verification failure then exits non-zero, and a successful run prints the exact numeric `churn_predictor` version, source run ID, application identity, pipeline checksum, and artifact-manifest checksum. Open the repository’s MLflow tab in DagsHub to inspect experiments, runs, metrics, artifacts, and registered versions.

`RowNumber`, `CustomerId`, `Surname`, email/phone/address aliases, and case variants are rejected before references are written. Mutable local files are useful development outputs, but they are not production identities and Modal never deploys them directly.

## Validate and package an exact model

These commands also use the DagsHub settings shown above, including `ENABLE_DAGSHUB_TRACKING=true`.

Only a positive numeric registry version is accepted:

```bash
python -m src.mlops validate-model \
  --model-uri models:/churn_predictor/7 \
  --output json
```

The validator resolves the source run, checks parameters, metrics, tags, signature, input example, contracts, reference purposes/privacy, checksums, model loading, and a smoke prediction. It rejects `latest`, aliases such as `@champion`, and stages such as `/Production`. Integrity-enabled versions additionally require `integrity_status=complete`, a valid manifest and completion marker, exact run/version/application linkage, and a byte-for-byte match for every protected artifact. A missing, extra, resized, replaced, or corrupted protected artifact returns a non-zero exit status before the serialized pipeline is loaded.

The immutable application identity is:

```text
dagshub:<owner>/<repository>:churn_predictor:<numeric-version>
```

Four related values have different purposes:

| Value | Purpose |
| --- | --- |
| Numeric registered-model version | Authoritative DagsHub model-registry version used in `models:/churn_predictor/<version>` |
| `model_version_id` | Repository-qualified application identity for that numeric version |
| `pipeline_sha256` | Digest of the serialized fitted inference pipeline |
| `artifact_manifest_sha256` | Digest of the canonical inventory covering the complete protected publication |

The last two checksums supplement the numeric version; neither creates a second registry or a content-addressed model identity.

### Integrity manifest and publication ordering

New production publications add these run artifacts:

```text
integrity/artifact_manifest.json
integrity/publication_complete.json
```

The manifest schema is versioned at `configs/contracts/artifact_manifest.schema.json`. Protected files include the complete logged MLflow model directory (MLmodel, fitted pipeline, environment/dependency files, and input examples), contracts, training configuration, aggregate evaluation/model-comparison results, dataset identities, and the approved drift/evaluation reference metadata and datasets. Transient files, caches, logs, `.env` files, credentials, connection strings, raw identifiers, unrestricted source datasets, and absolute local paths are rejected.

Canonicalization version `sorted-compact-json-v1` uses UTF-8, sorted object keys, no insignificant whitespace, stable path ordering, repository-independent POSIX paths, explicit JSON nulls only where the schema permits them, and rejects NaN or infinity. The checksum is `sha256(canonical_json(artifact_manifest))`; the manifest never contains its own checksum.

Publication first marks the registered version incomplete, logs all approved artifacts and the fitted MLflow model, resolves the exact numeric version, inventories and verifies the protected files, logs and tags the manifest checksum, and runs a verified smoke prediction. `publication_complete.json` is the final artifact write. Only then are the run and model-version integrity tags changed to `complete`. A failure leaves the version incomplete, publishes no completion marker when the failure precedes completion, exits non-zero when registration is required, and never falls back to another version.

Versions created before this format are returned by `validate-model` with `"integrity_status": "legacy"`; they are not presented as integrity-verified and cannot be packaged by the current production deployment command. Historical metadata is not retrofitted or mutated.

Prepare the inference-only Modal directory before deployment:

```bash
python -m src.mlops prepare-deployment \
  --model-uri models:/churn_predictor/7 \
  --output-dir build/model \
  --expected-run-id "$EXPECTED_MLFLOW_RUN_ID" \
  --expected-model-version-id "$EXPECTED_MODEL_VERSION_ID" \
  --expected-pipeline-sha256 "$EXPECTED_PIPELINE_SHA256" \
  --expected-artifact-manifest-sha256 "$EXPECTED_ARTIFACT_MANIFEST_SHA256" \
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

Alembic is the only supported production schema-management mechanism. Revision `20260822_0002` adds prediction events and monitoring policy, baseline, and run tables. Revision `20260824_0003` adds idempotent outcomes, append-only source watermarks and label revisions, and performance-run metadata. Under `APP_ENV=test`, Alembic rejects remote targets to prevent tests from reaching production.

## Scheduled data-quality and drift monitoring

Monitoring uses an enabled immutable policy, an immutable baseline for one exact model version, and a cadence-aligned extraction cutoff. The database watermark combines `persisted_at` with the maximum persisted `event_id`, so late arrivals are excluded and a retry selects the same cohort. A rolling window doubles up to the policy maximum/fixed boundary; excess rows are limited deterministically by `prediction_timestamp DESC, event_id DESC`. Model versions are never combined.

Apply migrations and register policy v1 idempotently:

```bash
python -m alembic upgrade head
python -m src.monitoring register-policy --file configs/monitoring/policy-v1.0.0.json
```

Create a baseline JSON record containing the exact model version, immutable `s3://` reference URI and SHA-256, schema version, creation/activation times, purpose, and approval metadata, then register it:

```bash
python -m src.monitoring register-baseline --file baseline-v1.json
```

The reference must be Parquet or CSV, contain at least the configured minimum rows, match the policy feature schema, and include `prediction_probability` and `predicted_class`. Its bytes are checksum-verified before Evidently runs. The prediction API appends one `prediction_events` row for every successfully returned single prediction and every valid row in a successful or partial batch. Batch caller identifiers are used only in the response and are never persisted. The insert contains only approved canonical features plus prediction output and verified deployment identity metadata; a failed insert fails the API request, and a batch is committed atomically. Public prediction requests remain `monitoring_eligible=false` because they do not carry a trusted tokenized customer identity for outcome attribution.

The prediction runtime role needs `INSERT` on `prediction_events` and access to its generated identity sequence. Grant those privileges to the dedicated prediction role through the platform migration/admin identity; do not broaden the role to schema ownership or customer-level monitoring reads.

Manual local/operations execution is:

```bash
python -m src.monitoring run \
  --environment production \
  --model-version-id dagshub:owner/repository:churn_predictor:7 \
  --as-of 2026-08-22T12:15:00Z
```

Modal deploys `scheduled_monitoring` at `15 */6 * * *` UTC and exposes `run_monitoring` for manual debugging. Both have three bounded retries; neither is called by FastAPI. Successful runs publish `report.html`, strict `report.json`, normalized `summary.json`, and `checksums.json` beneath the immutable model/baseline/run prefix. Insufficient-volume runs publish only a non-green summary and checksums and persist `insufficient_data`, not `completed`.

Policy v1 values—including row counts, ranges, methods, and thresholds—are explicitly initial hypotheses. Calibrate them from observed production volume and distributions. Any result-affecting change requires a new policy version; a changed reference requires a new baseline version. Start with the concise [monitoring operations guide](docs/monitoring/README.md); the detailed reproducibility contract remains in [data-quality-drift-jobs-v1.md](docs/monitoring/data-quality-drift-jobs-v1.md).

## Outcome labels and matured-cohort performance

The protected `/api/monitoring/outcomes` endpoint ingests real or simulated events with per-record batch results, HMAC tokenization, source-identity idempotency, and append-only corrections. Source owners declare completeness through `/api/monitoring/outcomes/watermarks`. Raw customer identifiers and rejected payloads are never persisted or logged.

Modal runs daily label materialization at `02:45 UTC` and weekly performance reporting at `03:30 UTC` on Monday. Negatives require horizon maturity, grace, and all required watermarks. Performance artifacts use the immutable `monitoring/{model_version_id}/performance/{monitoring_run_id}/` layout. See [outcomes-labels-performance-v1.md](docs/monitoring/outcomes-labels-performance-v1.md) for ingestion, correction, reproducibility, metric-availability, simulation, and privacy-suppression behavior.

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
EXPECTED_ARTIFACT_MANIFEST_SHA256=<digest>
MONITORING_ARTIFACT_BUCKET=<bucket>
MONITORING_ARTIFACT_REGION=<region>
CUSTOMER_TOKEN_HMAC_SECRET=<at-least-32-random-bytes>
CUSTOMER_TOKEN_KEY_ID=<key-version>
OUTCOME_INGESTION_API_KEY=<at-least-24-random-characters>
OUTCOME_ALLOWED_REAL_SOURCES=customer-master
REQUIRED_OUTCOME_SOURCES=customer-master
MONITORING_DEPLOYMENT_IDS=<deployment-id>
LABEL_CONTRACT_VERSION=1.0.0
LABEL_CONTRACT_APPROVED=<true-only-after-recorded-approval>
MONITORING_POLICY_VERSION=1.0.0
PREDICTION_HORIZON_DAYS=90
LABEL_GRACE_PERIOD_DAYS=7
PERFORMANCE_COHORT_DAYS=30
DEPLOYED_CLASSIFICATION_THRESHOLD=0.5
MONITORING_MINIMUM_PRIVACY_SIZE=20
# MONITORING_ARTIFACT_ENDPOINT_URL=<S3-compatible-endpoint-if-needed>
# Provider workload credentials, scoped to the monitoring prefix
```

Do not add DagsHub credentials to this Modal secret. They are used locally or in the protected GitHub deployment environment only to prepare `build/model` before `modal deploy`.

```bash
modal deploy modal_app.py
```

At container startup, Modal verifies the package, expected model/run/application identity, pipeline checksum, artifact-manifest checksum, complete integrity status, feature-schema version, deserialization, smoke prediction, typed production database configuration, and database connectivity before returning the FastAPI application. Individual requests load the local package and make no DagsHub call.

The GitHub deployment workflow requires protected production secrets for Modal authentication, DagsHub download, and every expected model identity value. Configure the same runtime identity and Neon values in the Modal secret.

## API and local serving

Production inference must use a verified package prepared from an exact registered version.

```bash
uvicorn application:app --host 0.0.0.0 --port 5001
```

Endpoints:

- `GET /health`
- `POST /api/predict`
- `POST /api/predict/batch`
- `POST /api/monitoring/outcomes` (protected; single or partial-success batch)
- `POST /api/monitoring/outcomes/watermarks` (protected)
- `/docs`, `/redoc`, and `/openapi.json`

Verified health metadata exposes `deployment_id`, model name, exact numeric version, `model_version_id`, source MLflow run, feature-schema version, `pipeline_sha256`, `artifact_manifest_sha256`, and `integrity_status`. Safe startup and prediction logs contain these deployment/integrity identifiers but never feature values or customer identifiers. Single responses and batch metadata retain their existing identity fields. No fallback version is fabricated when verified metadata is unavailable.

## Rollback

Rollback is a new deployment, not an alias change or runtime hot swap:

1. Choose a historical numeric `churn_predictor` version in DagsHub.
2. Run `validate-model` for that exact URI and confirm it is integrity-complete (legacy versions are not deployment-eligible).
3. Update all expected identity/checksum values, including `EXPECTED_ARTIFACT_MANIFEST_SHA256`.
4. Prepare a fresh `build/model`; this generates a new `deployment_id`.
5. Update the Modal secret identity and redeploy.
6. Confirm `/health` reports the intended numeric model version and new deployment ID.

## Security and data publication

Allowed DagsHub artifacts are approved raw model features, labelled evaluation reference rows, deterministic dataset identities, aggregate evaluation results, versioned contracts, and the fitted pipeline. DagsHub MLflow remains the only tracking backend, artifact source, and model registry; no local historical model store, hash hierarchy, latest pointer, or alternate publication CLI exists. Never upload `.env` files, credentials, connection strings, raw identifiers, unrestricted source datasets, or production prediction payloads. Logs must contain identity, checksums, integrity status, and durations—not raw features, reference rows, authentication headers, database URLs, DagsHub tokens, or URLs with credentials.

The fitted sklearn pipeline is a trusted executable artifact. Validation accepts it only from the configured DagsHub repository and verifies the complete inventory and pipeline checksum before deserialization. DagsHub publishing credentials stay in the local/protected deployment environment, while the Neon URL and monitoring bucket configuration exist only in the Modal runtime secret. The inference package copies only the MLflow model, feature contract, and safe deployment metadata; training metrics and reference rows remain outside the Modal image.

## Tests

```bash
pytest -q
```

Unit tests do not access DagsHub, Neon, or the artifact bucket. External integration tests are opt-in and skip unless their dedicated credentials are supplied. Outcome/label rules remain documented in [production-monitoring-contract-v1.md](docs/monitoring/production-monitoring-contract-v1.md); offline data-quality/drift execution is documented separately.
