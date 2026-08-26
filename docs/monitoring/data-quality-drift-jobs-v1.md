# Data-quality and drift jobs v1

This is the operations and reproducibility contract for `policy_version=1.0.0`.
It is separate from the delayed-outcome contract. Evidently 0.7.21 is deliberately
pinned and executed only by
the scheduled/manual Modal functions in `modal_app.py`; FastAPI imports neither
the runner nor Evidently and performs no report work.

## Versioned inputs

`monitoring_policies` stores the full configuration and its canonical SHA-256.
Result-affecting content is database-protected against updates. Enable/disable is
a lifecycle operation, but changing thresholds, methods, row limits, windows,
latency, environments/models, feature rules, exclusions, or suppressions requires
a new version.

`monitoring_baselines` maps one baseline version to exactly one model version and
records the reference URI/checksum, schema version, creation/activation/retirement
times, purpose, and approvals. Dataset identity cannot be updated. A baseline may
be retired once; replacement data requires a new version. Activation cannot
predate creation, preventing retroactive retry changes.

The values in `configs/monitoring/policy-v1.0.0.json` are hypotheses, not proven
production limits. In particular, 200/10,000 current rows, 500 reference rows,
24-hour initial/30-day maximum lookback, 30-minute latency, feature ranges,
5% p-value thresholds, 30% drifted-column share, and 50% row-count warning need
calibration from production volume, seasonality, data delays, and false-alert
rates. Calibration produces policy v1.1.0 or later; v1.0.0 is never edited.

## Deterministic extraction

For a cadence slot, the worker subtracts the policy latency allowance and records
that fixed cutoff. It queries the maximum `event_id` already persisted by the
cutoff and the maximum eligible prediction timestamp for the exact environment
and model. Every later count/extract uses all of:

```text
environment = requested environment
model_version_id = requested exact version
window_start <= prediction_timestamp <= window_end
persisted_at <= extraction_cutoff
event_id <= maximum_persisted_event_id
```

Thus a late-arriving row is excluded even if its business prediction timestamp is
inside the interval. The window starts at the initial lookback and doubles until
the minimum is met, the maximum lookback is reached, or the fixed historical
boundary applies. If eligible volume exceeds the maximum, the worker keeps the
newest rows ordered by `(prediction_timestamp DESC, event_id DESC)` and emits the
final report in ascending order. It never broadens across models.

The SHA-256 run identity includes job type, environment, exact model, baseline,
policy, final window, cutoff, cursor, and maximum eligible prediction timestamp.
A PostgreSQL advisory lock plus primary/unique constraints reserves it before
artifact publication. A completed/insufficient retry returns the record; a failed
retry can resume the same identity. Immutable object writes accept identical
bytes but reject conflicting bytes.

## Validation and statuses

Before Evidently, the worker verifies reference bytes, row count, exact columns,
data types, and schema-version compatibility. Decode, checksum, extraction-count,
and schema failures are persisted as `failed` with
`schema_or_extraction_failure`; they are not statistical warnings.

Deterministic checks cover missing/unexpected columns, types, missing values,
feature ranges/categories, duplicate prediction IDs, probability `[0,1]`, row
count change, and feature integrity. Evidently runs numeric, categorical,
probability, and predicted-class drift using policy methods/thresholds. Operational
status (`completed`, `insufficient_data`, `failed`) remains separate from quality
and drift status (`pass`, `warning`, `fail`, `not_evaluated`). Insufficient data
never produces a green result.

## Artifacts and reproduction

Completed paths are:

```text
monitoring/{url-encoded-model-version-id}/{baseline-version-id}/drift/{run-id}/
  report.html
  report.json
  summary.json
  checksums.json
```

`summary.json` records report identity, exact counts/window/watermark, extraction
start/completion, full policy/baseline configuration, Evidently version/effective
configuration, feature results, overall quality/drift, and suppression/exclusion
metadata. `report.json` is strict JSON: non-finite library values are represented
as `null`. `checksums.json` hashes the other stored artifacts. Neon retains the
same run configuration, artifact URIs/checksums, statuses, summary, and safe error
details. Evidently's presentation-only random HTML element IDs are replaced in
stable encounter order; this does not change report values and makes an identical
retry byte-for-byte compatible with immutable object writes.

To reproduce, use the recorded policy and baseline versions, fetch and verify the
reference, rerun the exact SQL criteria with the recorded cutoff/cursor/window,
apply the recorded deterministic limit, and use the recorded Evidently version
and configuration. Operations can supply the recorded cadence time to either:

```bash
python -m src.monitoring run --environment production \
  --model-version-id <exact-id> --as-of <scheduled-for>

modal run modal_app.py::run_monitoring --scheduled-for <scheduled-for>
```
