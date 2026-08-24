"""Deterministic hard schema checks and policy-driven data-quality checks."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.monitoring.models import MonitoringPolicy, ResultStatus


class MonitoringValidationError(ValueError):
    """Hard extraction/reference/schema failure, distinct from a statistical warning."""


def _type_matches(series: pd.Series, expected: str) -> bool:
    present = series.dropna()
    if present.empty:
        return True
    if expected == "integer":
        numeric = pd.to_numeric(present, errors="coerce")
        return bool(numeric.notna().all() and (numeric % 1 == 0).all())
    if expected in {"number", "float"}:
        return bool(pd.to_numeric(present, errors="coerce").notna().all())
    if expected in {"string", "category"}:
        return bool(present.map(lambda value: isinstance(value, str)).all())
    if expected == "boolean":
        return bool(present.isin([True, False, 0, 1]).all())
    raise MonitoringValidationError(f"unsupported policy data type: {expected}")


def validate_schema_compatibility(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    policy: MonitoringPolicy,
    baseline_schema_version: str,
    current_schema_versions: set[str],
) -> None:
    if current_schema_versions != {baseline_schema_version}:
        raise MonitoringValidationError(
            "current prediction schema version does not match the baseline"
        )
    expected = set(policy.feature_rules)
    for label, frame in (("reference", reference), ("current", current)):
        missing = sorted(expected - set(frame.columns))
        unexpected = sorted(
            set(frame.columns)
            - expected
            - {"prediction_id", "prediction_probability", "predicted_class"}
        )
        if missing:
            raise MonitoringValidationError(f"{label} data is missing columns: {missing}")
        if unexpected:
            raise MonitoringValidationError(
                f"{label} data has unexpected columns: {unexpected}"
            )
        for feature, rule in policy.feature_rules.items():
            if not _type_matches(frame[feature], rule.data_type):
                raise MonitoringValidationError(
                    f"{label} column {feature} is incompatible with {rule.data_type}"
                )


def data_quality_summary(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    policy: MonitoringPolicy,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, severity: str, **details: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": ResultStatus.PASS if passed else severity,
                **details,
            }
        )

    duplicate_count = int(current["prediction_id"].duplicated().sum())
    add("duplicate_prediction_ids", duplicate_count == 0, ResultStatus.FAIL, count=duplicate_count)
    probabilities = pd.to_numeric(current["prediction_probability"], errors="coerce")
    invalid_probability_count = int((probabilities.isna() | ~probabilities.between(0, 1)).sum())
    add(
        "prediction_probability_range",
        invalid_probability_count == 0,
        ResultStatus.FAIL,
        count=invalid_probability_count,
        expected={"minimum": 0.0, "maximum": 1.0},
    )
    reference_rows = max(len(reference), 1)
    row_ratio = len(current) / reference_rows
    row_delta = abs(row_ratio - 1.0)
    add(
        "unexpected_row_count_change",
        row_delta <= policy.row_count_change_warning_ratio,
        ResultStatus.WARNING,
        current_rows=len(current),
        reference_rows=len(reference),
        ratio=row_ratio,
        warning_ratio=policy.row_count_change_warning_ratio,
    )

    feature_results: dict[str, Any] = {}
    for feature, rule in policy.feature_rules.items():
        series = current[feature]
        missing_count = int(series.isna().sum())
        violations: Counter[str] = Counter()
        if missing_count and not rule.nullable:
            violations["missing"] = missing_count
        if rule.minimum is not None or rule.maximum is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            if rule.minimum is not None:
                violations["below_minimum"] = int((numeric < rule.minimum).sum())
            if rule.maximum is not None:
                violations["above_maximum"] = int((numeric > rule.maximum).sum())
        if rule.allowed_values is not None:
            violations["invalid_category"] = int(
                (~series.isna() & ~series.isin(rule.allowed_values)).sum()
            )
        violations = Counter({key: value for key, value in violations.items() if value})
        passed = not violations
        feature_results[feature] = {
            "status": ResultStatus.PASS if passed else ResultStatus.FAIL,
            "missing_count": missing_count,
            "violations": dict(violations),
            "rule": rule.model_dump(mode="json"),
            "suppressed": feature in policy.suppressed_features,
            "excluded": feature in policy.excluded_features,
        }
        add(
            f"feature_integrity:{feature}",
            passed,
            ResultStatus.FAIL,
            violations=dict(violations),
        )

    failures = [item for item in checks if item["status"] == ResultStatus.FAIL]
    warnings = [item for item in checks if item["status"] == ResultStatus.WARNING]
    status = ResultStatus.FAIL if failures else ResultStatus.WARNING if warnings else ResultStatus.PASS
    return {
        "status": status,
        "checks": checks,
        "feature_results": feature_results,
    }

