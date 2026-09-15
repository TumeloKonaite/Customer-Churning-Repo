# Outcome and delayed-label workflow

`POST /api/monitoring/outcomes` is the protected source for churn outcomes.
Customer identifiers are HMAC-tokenized before persistence; raw identifiers and
payloads are neither stored nor logged. Source event identity is idempotent and
corrections/retractions remain append-only.

The daily `scheduled_label_materialization` worker snapshots outcome arrival and
source completeness. Attribution uses:

```text
same customer token
AND prediction_timestamp < event_timestamp <= horizon_end
```

Negatives require the complete horizon, grace period, and all required source
watermarks. Accepted real revisions create Arize actual outbox records in the
same transaction. Pending or simulated labels are never exported to the
production Arize environment. Every actual uses the prediction's original opaque
`prediction_id` and model version.
