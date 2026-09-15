"""Platform-neutral aggregate parity checks for the Arize shadow period."""

from __future__ import annotations

from typing import Mapping

RECONCILIATION_METRICS = (
    "prediction_count", "actual_count", "positive_prediction_rate",
    "actual_churn_rate", "missing_value_rate", "prediction_score_mean",
)


def reconcile(neon: Mapping[str, float], arize: Mapping[str, float], *, count_tolerance: int = 0,
              rate_tolerance: float = 1e-6) -> dict[str, object]:
    """Compare sanitized aggregates; customer-level values never enter the result."""
    missing = [
        f"{source}.{metric}"
        for source, values in (("neon", neon), ("arize", arize))
        for metric in RECONCILIATION_METRICS if metric not in values
    ]
    if missing:
        raise ValueError(f"reconciliation aggregates are incomplete: {missing}")
    differences = {}
    passed = True
    for metric in RECONCILIATION_METRICS:
        values = [float(neon[metric]), float(arize[metric])]
        spread = max(values) - min(values)
        tolerance = count_tolerance if metric.endswith("count") else rate_tolerance
        differences[metric] = {"spread": spread, "tolerance": tolerance,
                               "passed": spread <= tolerance}
        passed = passed and spread <= tolerance
    return {"status": "passed" if passed else "failed", "metrics": differences}
