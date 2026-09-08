# Arize monitoring operations

Arize AX is the sole model-monitoring and observability platform. Neon remains
the durable source of prediction events, export delivery state, outcomes, and
append-only label revisions.

The FastAPI request path never imports or contacts Arize. Successful predictions
atomically create `prediction_events` and pending `arize_export_events`. The
hourly Modal worker claims bounded batches and sends the canonical ten features,
binary label, positive-class score, exact model version, deployment ID, source,
and optional batch ID through the Arize Python v8 client.

Delayed real labels come only from the protected outcome-ingestion and daily
label-materialization workflow. Actuals are deferred until the observation
window closes and always join with the original opaque `prediction_id`.

See [Arize operations](arize-operations.md) for approval, baseline, backfill,
deployment, retry/dead-letter handling, validation, and rollback.
