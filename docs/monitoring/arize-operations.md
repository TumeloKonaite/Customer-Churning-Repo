# Arize production operations

## Safety boundaries

Arize export is disabled unless `ARIZE_EXPORT_ENABLED=true` and the referenced
`arize_privacy_approvals` row is active, unexpired, unrevoked, and covers the
exact feature/tag allowlists. Neon remains authoritative. Credentials belong in
the Modal secret `customer-churn-arize`, never in Git or logs.

## Database and baseline

Apply migrations before deploying inference code:

```bash
uv run --env-file .env alembic upgrade head
uv run --env-file .env alembic current
```

Upload only the registered approved reference artifact:

```dotenv
MONITORING_ARTIFACT_BACKEND=mlflow
ENABLE_DAGSHUB_TRACKING=true
DAGSHUB_REPO_OWNER=<owner>
DAGSHUB_REPO_NAME=<repository>
DAGSHUB_TOKEN=<token>
```

Store the exact run URI in `monitoring_baselines.reference_dataset_uri`:

```text
runs:/<exact-run-id>/references/evaluation_reference.parquet
```

Arize validation baselines require the labeled evaluation reference so the schema
contains both prediction and actual labels. Do not register the unlabeled drift
reference for this upload.

Then run the upload from the repository checkout (the scheduled export image does
not need DagsHub dependencies):

```bash
uv run --extra arize-export --env-file .env python -m src.monitoring.arize upload-baseline \
  --model-version-id 'dagshub:<owner>/<repo>:churn_predictor:<version>' \
  --baseline-version-id '<approved-baseline-id>' \
  --package-dir build/model
```

The command verifies checksum, schema, approval metadata, and exact packaged
model identity. It rejects empty, incompatible, or non-allowlisted datasets.

## Backfill and delivery

```bash
uv run --extra arize-export --env-file .env python -m src.monitoring.arize backfill \
  --from 2026-01-01T00:00:00Z --to 2026-02-01T00:00:00Z \
  --model-version '<version>' --batch-size 500 --dry-run
```

Remove `--dry-run` after reviewing counts. Repeat bounded invocations until the
selected count is zero. Prediction pages are enqueued before actual pages;
existing export identities are skipped. Preserve emitted resume cursors.

The Modal functions `scheduled_arize_export` and `run_arize_export` use the same
worker path. Inspect only aggregate delivery state:

```sql
SELECT status, event_type, count(*)
FROM arize_export_events
GROUP BY status, event_type
ORDER BY status, event_type;
```

Temporary failures use bounded exponential backoff. Exhausted and permanent
failures become recoverable dead letters. Stale claims are automatically returned
to pending. Retries retain the original prediction ID.

## Arize monitors

Configure prediction volume/no-data, positive-score and predicted-label drift,
approved feature drift, missing/range/cardinality checks, unexpected categorical
values, and model-version/deployment separation. After delayed labels arrive,
configure accuracy, precision, recall, F1, FPR/FNR, AUC, PR-AUC, and log loss.

Validate Neon-to-Arize counts, model versions, distributions, delayed actuals,
label revisions, retries, and dead letters for a complete monitoring window.

## Rollback

Revoke the approval or set `ARIZE_EXPORT_ENABLED=false`, redeploy Modal, and keep
pending outbox rows for later replay. Do not delete prediction, outcome, label,
baseline, or delivery history.
