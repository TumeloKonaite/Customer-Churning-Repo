# Outcome, label, and performance monitoring v1

This implementation observes churn only after a prediction's future horizon. It
never treats the absence of an outcome as a negative until the complete horizon,
the grace period, and every required source-completeness watermark have elapsed.

## Outcome boundary

`POST /api/monitoring/outcomes` accepts one event or `{"records": [...]}`. The
endpoint requires `X-Outcome-API-Key`. A record includes source namespace and
event ID, environment and tenant-scoped stable customer identity, event type and
UTC event/received timestamps, real/simulated metadata, and an operation plus
reference for a correction, retraction, or supersession.

The service HMAC-tokenizes the customer ID before persistence. Neither the raw ID,
tenant ID, request payload, nor Pydantic validation input is stored, logged, or
returned. `(source_namespace, source_event_id)` is the concurrency-safe unique
boundary: identical content returns `replayed`; different canonical content is a
409 conflict. Batch results are per-record and safe to retry. Quarantine stores
only source identity and a reason code, never the rejected payload.

`POST /api/monitoring/outcomes/watermarks` records append-only, monotonically
advancing source-completeness declarations. Reproductions select the last
declaration observed at or before their extraction time.

## Attribution and revisions

The daily Modal `scheduled_label_materialization` job snapshots outcome arrival
and source completeness, then evaluates every eligible prediction. The exact rule
is:

```text
same customer token
AND prediction_timestamp < event_timestamp <= horizon_end
```

An eligible event fans out to every overlapping prediction. Positives reference
the selected outcome. Negatives additionally require maturity and source
completeness through the horizon end. Corrections create append-only label
revisions linked by `supersedes_label_revision_id`; no outcome or label fact is
updated or deleted. Retry identity includes the complete outcome snapshot and
contract configuration.

Pre-migration predictions remain `monitoring_eligible=false`. They must not be
backfilled by guessing identity or horizon values.

Production label and performance workers additionally require
`LABEL_CONTRACT_APPROVED=true`. Set it only after the named approval record in the
production monitoring contract is complete; the repository's current proposed
contract remains fail-closed until then.

## Performance reports

The weekly Modal `scheduled_performance_monitoring` job first materializes labels,
then selects a `[cohort_start, cohort_end)` prediction-time cohort. It fixes the
outcome watermark and maximum label revision, requires the recorded horizon and
grace period to have elapsed, and uses exactly one model and outcome mode.

Reports include confusion counts, classification metrics, ROC/PR AUC, log loss,
Brier score, reliability bins, optional ECE, calibration intercept/slope when
appropriate, deployed-threshold results, support/rates, exclusions, and explicit
unavailable reasons. Analysis thresholds never modify production policy.

Approved segment definitions are versioned. Primary suppression below `k` is
paired with complementary suppression so totals cannot reveal a small group;
hidden names, counts, support, and metrics are omitted. Simulated reports are
separate and visibly labeled `SIMULATED — NOT PRODUCTION PERFORMANCE`.

Artifacts are immutable and checksum protected at:

```text
monitoring/{model_version_id}/performance/{monitoring_run_id}/
  report.html
  report.json
  summary.json
  checksums.json
```

Neon stores the run configuration, model/deployment/policy and contract versions,
cohort bounds/rule, both watermarks, artifact URIs and checksums, summary metrics,
suppression metadata, status, and safe error details. Late outcomes therefore
produce a new snapshot/run and never mutate an earlier report.
