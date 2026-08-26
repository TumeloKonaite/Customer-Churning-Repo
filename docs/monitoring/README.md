# Monitoring operations

## The simple mental model

The drift calculation is the same reference-versus-current workflow shown in the
Evidently tutorial:

```python
import pandas as pd
from pathlib import Path

from src.monitoring.drift import MonitoringPolicy, run_drift_report

policy = MonitoringPolicy.model_validate_json(
    Path("configs/monitoring/policy-v1.0.0.json").read_text()
)
reference = pd.read_parquet("reference.parquet")
current = pd.read_parquet("current.parquet")

result = run_drift_report(reference, current, policy=policy)
result.snapshot  # displays the native Evidently report in Jupyter
```

`run_drift_report` builds `Report([DataDriftPreset(), DataSummaryPreset()])`, calls
`report.run(current_data=..., reference_data=...)`, and returns the native
Evidently snapshot together with HTML, JSON, and a small normalized drift summary.
The tutorial uses an older Evidently release where `show()` is called on the
report. With the pinned 0.7.21 API, `run()` returns a snapshot; evaluating
`result.snapshot` in Jupyter displays it.

Everything else makes that calculation safe and repeatable in production. It does
not change the underlying Evidently workflow:

```text
reference DataFrame + current DataFrame
                  |
                  v
          run_drift_report
                  |
                  v
     HTML + JSON + drift summary
```

The code is divided by responsibility:

- `drift/evidently.py` is the small statistical workflow above.
- `drift/service.py` selects data and saves a production run.
- `drift/selection.py` decides which production time window is current.
- `drift/quality.py` checks schema and deterministic rules before statistics.
- `drift/repository.py` contains database reads and run-status writes.

Start with `evidently.py` when learning drift. Read `service.py` when you need to
understand scheduling, retries, or reproducibility.

## What happens in production

Prediction requests do only model inference plus one atomic, privacy-safe append
to `prediction_events` through `src/database/prediction_events.py`. Caller
identifiers are excluded and public rows remain `monitoring_eligible=false`.

Every six hours, Modal calls `scheduled_monitoring` (`15 */6 * * *` UTC). Its
adapter constructs `src.monitoring.drift.service.MonitoringJob`, whose entry point:

1. resolves an immutable policy, extraction cutoff, watermark, and deterministic
   current window;
2. resolves the exact model baseline and verifies its SHA-256 and feature schema;
3. extracts only matching environment/model events, performs deterministic quality
   checks, and calls `run_drift_report` using the pinned Evidently 0.7.21 API;
4. immutably publishes the artifact bundle and persists separate operational,
   quality, and drift statuses.

The baseline is the newest active `monitoring_baselines` record for the exact model
version at the selected interval end. Reports are stored at:

```text
monitoring/{encoded-model-version}/{baseline-version}/drift/{run-id}/
  report.html
  report.json
  summary.json
  checksums.json
```

Inspect the latest run by querying `monitoring_runs` for the exact environment and
model, ordered by `scheduled_for DESC`; use its `artifact_uris` and verify the
recorded checksums. Reproduce it with the recorded schedule:

```bash
python -m src.monitoring run --environment production \
  --model-version-id <exact-model-version-id> --as-of <scheduled-for>
```

Outcome ingestion and delayed performance are separate capabilities. A protected
outcome API tokenizes trusted customer identity, the daily label job waits for the
prediction horizon, grace period, and source completeness, and the weekly
performance job evaluates matured labels. None of that machinery participates in
prediction-event persistence, drift extraction, or Evidently report generation.
Public predictions cannot enter this attribution path until a trusted identity
source is integrated and the label contract is approved.

Detailed guarantees remain in
[data-quality-drift-jobs-v1.md](data-quality-drift-jobs-v1.md), while privacy,
attribution, and performance contracts remain in
[production-monitoring-contract-v1.md](production-monitoring-contract-v1.md) and
[outcomes-labels-performance-v1.md](outcomes-labels-performance-v1.md).
