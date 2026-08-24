"""Reproducible metrics for matured, immutable prediction/label cohorts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import html
import math
from typing import Any, Iterable, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.monitoring.models import canonical_json_bytes, require_utc, timestamp


class PerformanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str
    prediction_timestamp: datetime
    horizon_end: datetime
    model_version_id: str
    deployment_id: str
    policy_version: str
    label_contract_version: str
    prediction_probability: float | None
    predicted_class: int | str | bool
    label_value: int
    label_revision_id: int = Field(ge=1)
    label_attribution_timestamp: datetime
    is_simulated: bool
    simulation_generator: str | None = None
    simulation_scenario_version: str | None = None
    segments: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "prediction_timestamp", "horizon_end", "label_attribution_timestamp"
    )
    @classmethod
    def timestamps_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)

    @field_validator("label_value")
    @classmethod
    def binary_label(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("label_value must be binary")
        return value

    @model_validator(mode="after")
    def simulation_metadata_is_consistent(self) -> "PerformanceRecord":
        if self.is_simulated and (
            not self.simulation_generator or not self.simulation_scenario_version
        ):
            raise ValueError("simulated records require generator and scenario version")
        if not self.is_simulated and (
            self.simulation_generator is not None
            or self.simulation_scenario_version is not None
        ):
            raise ValueError("real records cannot contain simulation metadata")
        return self


class CohortDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_start: datetime
    cohort_end: datetime
    selection_rule: str = "prediction_timestamp"
    horizon_days: int = Field(ge=1)
    grace_period_days: int = Field(ge=0)
    outcome_watermark: dict[str, Any]
    label_revision_watermark: int = Field(ge=0)
    label_contract_version: str
    model_version_id: str
    deployment_ids: tuple[str, ...] = Field(min_length=1)
    policy_version: str
    is_simulated: bool
    simulation_generator: str | None = None
    simulation_scenario_version: str | None = None
    classification_threshold: float = Field(ge=0, le=1)
    comparison_report: bool = False

    @field_validator("cohort_start", "cohort_end")
    @classmethod
    def dates_are_utc(cls, value: datetime, info) -> datetime:
        return require_utc(value, info.field_name)

    @model_validator(mode="after")
    def valid_interval(self) -> "CohortDefinition":
        if self.cohort_end <= self.cohort_start:
            raise ValueError("cohort_end must be after cohort_start")
        if self.selection_rule != "prediction_timestamp":
            raise ValueError("v1 performance cohorts use prediction_timestamp")
        if self.is_simulated and (
            not self.simulation_generator or not self.simulation_scenario_version
        ):
            raise ValueError("simulated cohort requires generator and scenario version")
        if not self.is_simulated and (
            self.simulation_generator is not None
            or self.simulation_scenario_version is not None
        ):
            raise ValueError("real cohort cannot contain simulation metadata")
        return self


def available(value: Any) -> dict[str, Any]:
    return {"available": True, "value": _finite(value)}


def unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "value": None, "reason": reason}


def construct_matured_cohort(
    records: Iterable[PerformanceRecord],
    *,
    definition: CohortDefinition,
    evaluated_at: datetime,
) -> tuple[tuple[PerformanceRecord, ...], dict[str, int]]:
    """Select [start, end) predictions under a fixed label revision snapshot."""
    evaluated_at = require_utc(evaluated_at, "evaluated_at")
    exclusions: Counter[str] = Counter()
    selected: list[PerformanceRecord] = []
    grace = timedelta(days=definition.grace_period_days)
    for record in records:
        reason: str | None = None
        if not (
            definition.cohort_start
            <= record.prediction_timestamp
            < definition.cohort_end
        ):
            reason = "outside_prediction_time_cohort"
        elif record.horizon_end + grace > evaluated_at:
            reason = "horizon_or_grace_not_elapsed"
        elif record.horizon_end != record.prediction_timestamp + timedelta(
            days=definition.horizon_days
        ):
            reason = "horizon_contract_mismatch"
        elif not _outcome_watermark_complete(
            definition.outcome_watermark, record.horizon_end
        ):
            reason = "outcome_source_incomplete"
        elif record.label_revision_id > definition.label_revision_watermark:
            reason = "after_label_revision_watermark"
        elif record.label_contract_version != definition.label_contract_version:
            reason = "label_contract_version_mismatch"
        elif record.is_simulated != definition.is_simulated:
            reason = "outcome_mode_mismatch"
        elif record.policy_version != definition.policy_version:
            reason = "policy_version_mismatch"
        elif not definition.comparison_report and record.model_version_id != definition.model_version_id:
            reason = "model_version_mismatch"
        elif definition.deployment_ids and record.deployment_id not in definition.deployment_ids:
            reason = "deployment_not_in_cohort"
        if reason:
            exclusions[reason] += 1
        else:
            selected.append(record)
    modes = {row.is_simulated for row in selected}
    models = {row.model_version_id for row in selected}
    if len(modes) > 1:
        raise ValueError("real and simulated outcomes cannot be mixed")
    if len(models) > 1 and not definition.comparison_report:
        raise ValueError("official report must contain one model version")
    return tuple(
        sorted(selected, key=lambda row: (row.prediction_timestamp, row.prediction_id))
    ), dict(sorted(exclusions.items()))


def classification_metrics(
    records: Iterable[PerformanceRecord], *, threshold: float
) -> dict[str, Any]:
    rows = tuple(records)
    if not rows:
        result = {
            name: unavailable("empty matured cohort")
            for name in (
                "confusion_matrix", "precision", "recall", "f1", "accuracy",
                "roc_auc", "pr_auc", "log_loss", "support_counts",
                "observed_positive_rate", "predicted_positive_rate",
            )
        }
        result["deployed_threshold"] = threshold_metrics(rows, threshold=threshold)
        return result
    y_true = np.asarray([row.label_value for row in rows], dtype=int)
    try:
        y_pred = np.asarray(
            [_binary_prediction(row.predicted_class) for row in rows], dtype=int
        )
        class_reason = None
    except ValueError:
        y_pred = None
        class_reason = "one or more predicted classes are not binary"
    probabilities, probability_reason = _probabilities(rows)
    observed_positives = int(y_true.sum())
    both_classes = len(set(y_true.tolist())) == 2
    result = {
        "support_counts": available(
            {"total": len(rows), "negative": int(len(rows) - observed_positives),
             "positive": observed_positives}
        ),
        "observed_positive_rate": available(observed_positives / len(rows)),
    }
    if y_pred is None:
        for name in (
            "confusion_matrix", "precision", "recall", "f1", "accuracy",
            "predicted_positive_rate",
        ):
            result[name] = unavailable(class_reason or "invalid predicted classes")
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        predicted_positives = int(y_pred.sum())
        result.update(
            {
                "confusion_matrix": available(
                    {"true_negative": int(tn), "false_positive": int(fp),
                     "false_negative": int(fn), "true_positive": int(tp)}
                ),
                "precision": (
                    available(precision_score(y_true, y_pred))
                    if predicted_positives else unavailable("no predicted-positive records")
                ),
                "recall": (
                    available(recall_score(y_true, y_pred))
                    if observed_positives else unavailable("no observed-positive records")
                ),
                "f1": (
                    available(f1_score(y_true, y_pred))
                    if predicted_positives and observed_positives
                    else unavailable("F1 requires observed and predicted positives")
                ),
                "accuracy": available(accuracy_score(y_true, y_pred)),
                "predicted_positive_rate": available(predicted_positives / len(rows)),
            }
        )
    if probabilities is None:
        result.update(
            {
                "roc_auc": unavailable(probability_reason),
                "pr_auc": unavailable(probability_reason),
                "log_loss": unavailable(probability_reason),
            }
        )
    else:
        result["roc_auc"] = (
            available(roc_auc_score(y_true, probabilities))
            if both_classes else unavailable("ROC AUC requires both observed classes")
        )
        result["pr_auc"] = (
            available(average_precision_score(y_true, probabilities))
            if both_classes else unavailable("PR AUC requires both observed classes")
        )
        result["log_loss"] = available(log_loss(y_true, probabilities, labels=[0, 1]))
    result["deployed_threshold"] = threshold_metrics(rows, threshold=threshold)
    return result


def threshold_metrics(
    records: Iterable[PerformanceRecord], *, threshold: float
) -> dict[str, Any]:
    rows = tuple(records)
    probabilities, reason = _probabilities(rows)
    if not rows:
        return {"threshold": threshold, "available": False, "reason": "empty matured cohort"}
    if probabilities is None:
        return {"threshold": threshold, "available": False, "reason": reason}
    truth = np.asarray([row.label_value for row in rows], dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "available": True,
        "predicted_positive_count": int(predicted.sum()),
        "true_positive_count": int(tp),
        "false_positive_count": int(fp),
        "recall": (
            available(tp / (tp + fn)) if tp + fn else unavailable("no observed-positive records")
        ),
        "precision": (
            available(tp / (tp + fp)) if tp + fp else unavailable("no predicted-positive records")
        ),
    }


def threshold_table(
    records: Iterable[PerformanceRecord], thresholds: Iterable[float]
) -> list[dict[str, Any]]:
    return [threshold_metrics(records, threshold=value) for value in thresholds]


def calibration_metrics(
    records: Iterable[PerformanceRecord],
    *,
    bin_count: int = 10,
    binning_strategy: str = "uniform",
    minimum_bin_volume: int = 20,
    expected_calibration_error_enabled: bool = True,
    minimum_regression_rows: int = 50,
) -> dict[str, Any]:
    rows = tuple(records)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    if minimum_bin_volume < 1:
        raise ValueError("minimum_bin_volume must be positive")
    probabilities, reason = _probabilities(rows)
    configuration = {
        "binning_strategy": binning_strategy,
        "requested_bin_count": bin_count,
        "minimum_bin_volume": minimum_bin_volume,
        "expected_calibration_error_enabled": expected_calibration_error_enabled,
        "minimum_regression_rows": minimum_regression_rows,
    }
    if probabilities is None:
        unavailable_result = unavailable(reason)
        return {
            "configuration": configuration,
            "brier_score": unavailable_result,
            "reliability": unavailable_result,
            "expected_calibration_error": unavailable_result,
            "calibration_intercept": unavailable_result,
            "calibration_slope": unavailable_result,
        }
    truth = np.asarray([row.label_value for row in rows], dtype=int)
    boundaries = _bin_boundaries(probabilities, bin_count, binning_strategy)
    indices = np.minimum(np.searchsorted(boundaries, probabilities, side="right") - 1, len(boundaries) - 2)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    ece_suppressed = False
    for index in range(len(boundaries) - 1):
        selected = indices == index
        count = int(selected.sum())
        suppressed = count < minimum_bin_volume
        if count and suppressed:
            ece_suppressed = True
        if count and not suppressed:
            observed = float(truth[selected].mean())
            predicted = float(probabilities[selected].mean())
            ece += (count / len(rows)) * abs(observed - predicted)
            bins.append(
                {"lower": float(boundaries[index]), "upper": float(boundaries[index + 1]),
                 "count": count, "observed_churn_rate": observed,
                 "mean_predicted_probability": predicted, "suppressed": False}
            )
        else:
            bins.append(
                {"lower": float(boundaries[index]), "upper": float(boundaries[index + 1]),
                 "count": count, "observed_churn_rate": None,
                 "mean_predicted_probability": None, "suppressed": suppressed,
                 "reason": ("below minimum bin volume" if suppressed else "empty bin")}
            )
    regression_reason = None
    intercept = slope = None
    if len(rows) < minimum_regression_rows:
        regression_reason = f"requires at least {minimum_regression_rows} records"
    elif len(set(truth.tolist())) < 2:
        regression_reason = "calibration regression requires both observed classes"
    else:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        regression = LogisticRegression(C=1e9, solver="lbfgs").fit(logits, truth)
        intercept = float(regression.intercept_[0])
        slope = float(regression.coef_[0][0])
    return {
        "configuration": {**configuration, "bin_boundaries": boundaries.tolist()},
        "brier_score": available(brier_score_loss(truth, probabilities)),
        "reliability": available(bins),
        "expected_calibration_error": (
            unavailable("disabled by monitoring contract")
            if not expected_calibration_error_enabled
            else (
                unavailable("one or more nonempty bins are below minimum volume")
                if ece_suppressed else available(ece)
            )
        ),
        "calibration_intercept": (
            available(intercept) if regression_reason is None else unavailable(regression_reason)
        ),
        "calibration_slope": (
            available(slope) if regression_reason is None else unavailable(regression_reason)
        ),
    }


def segment_performance(
    records: Iterable[PerformanceRecord],
    *,
    approved_segments: Mapping[str, str],
    minimum_privacy_size: int,
    threshold: float,
) -> dict[str, Any]:
    """Apply primary and complementary suppression before emitting any counts."""
    if minimum_privacy_size < 2:
        raise ValueError("minimum_privacy_size must be at least 2")
    rows = tuple(records)
    dimensions: dict[str, Any] = {}
    for field, definition_version in approved_segments.items():
        groups: dict[str, list[PerformanceRecord]] = defaultdict(list)
        for row in rows:
            value = row.segments.get(field)
            if value is not None:
                groups[value].append(row)
        primary = {name for name, values in groups.items() if len(values) < minimum_privacy_size}
        hidden = set(primary)
        # Complementary suppression prevents total-minus-visible reconstruction.
        visible_candidates = sorted(
            ((len(values), name) for name, values in groups.items() if name not in hidden),
            key=lambda item: (item[0], item[1]),
        )
        if hidden and visible_candidates:
            hidden.add(visible_candidates[0][1])
        published = []
        for name in sorted(set(groups) - hidden):
            values = groups[name]
            published.append(
                {
                    "segment": name,
                    "denominator": len(values),
                    "support": Counter(row.label_value for row in values),
                    "metrics": classification_metrics(values, threshold=threshold),
                }
            )
        dimensions[field] = {
            "segment_definition_version": definition_version,
            "minimum_privacy_size": minimum_privacy_size,
            "published": published,
            "suppression": {
                "applied": bool(hidden),
                "suppressed_group_count": len(hidden),
                "method": "primary_and_complementary",
                # Names and counts are intentionally omitted.
            },
        }
    return dimensions


def performance_report_configuration(
    *,
    approved_segments: Mapping[str, str] | None,
    minimum_privacy_size: int,
    calibration_options: Mapping[str, Any] | None,
    analysis_thresholds: Iterable[float] | None,
) -> dict[str, Any]:
    calibration = {
        "bin_count": 10,
        "binning_strategy": "uniform",
        "minimum_bin_volume": 20,
        "expected_calibration_error_enabled": True,
        "minimum_regression_rows": 50,
        **dict(calibration_options or {}),
    }
    return {
        "approved_segment_definitions": dict(sorted((approved_segments or {}).items())),
        "minimum_privacy_size": minimum_privacy_size,
        "calibration": calibration,
        "analysis_thresholds": (
            None if analysis_thresholds is None else list(analysis_thresholds)
        ),
        "automatic_threshold_changes": False,
    }


def performance_run_id(
    definition: CohortDefinition,
    report_configuration: Mapping[str, Any] | None = None,
) -> str:
    return "performance_" + hashlib.sha256(
        canonical_json_bytes(
            {
                "cohort": definition.model_dump(mode="json"),
                "report_configuration": dict(report_configuration or {}),
            }
        )
    ).hexdigest()


def build_performance_report(
    records: Iterable[PerformanceRecord],
    *,
    definition: CohortDefinition,
    evaluated_at: datetime,
    approved_segments: Mapping[str, str] | None = None,
    minimum_privacy_size: int = 20,
    calibration_options: Mapping[str, Any] | None = None,
    analysis_thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    cohort, exclusions = construct_matured_cohort(
        records, definition=definition, evaluated_at=evaluated_at
    )
    report_configuration = performance_report_configuration(
        approved_segments=approved_segments,
        minimum_privacy_size=minimum_privacy_size,
        calibration_options=calibration_options,
        analysis_thresholds=analysis_thresholds,
    )
    run_id = performance_run_id(definition, report_configuration)
    classification = classification_metrics(
        cohort, threshold=definition.classification_threshold
    )
    report = {
        "monitoring_run_id": run_id,
        "report_type": "performance",
        "official": not definition.is_simulated,
        "display_label": (
            "SIMULATED — NOT PRODUCTION PERFORMANCE"
            if definition.is_simulated else "PRODUCTION PERFORMANCE"
        ),
        "cohort": {
            **definition.model_dump(mode="json"),
            "eligible_prediction_count": len(cohort),
            "excluded_prediction_counts": exclusions,
            "maturity_evaluated_at": timestamp(evaluated_at),
        },
        "report_configuration": report_configuration,
        "classification": classification,
        "calibration": calibration_metrics(
            cohort, **report_configuration["calibration"]
        ),
        "segments": segment_performance(
            cohort,
            approved_segments=approved_segments or {},
            minimum_privacy_size=minimum_privacy_size,
            threshold=definition.classification_threshold,
        ),
    }
    if report_configuration["analysis_thresholds"] is not None:
        report["threshold_analysis"] = threshold_table(
            cohort, report_configuration["analysis_thresholds"]
        )
    return report


def performance_report_html(report: Mapping[str, Any]) -> bytes:
    label = html.escape(str(report["display_label"]))
    run_id = html.escape(str(report["monitoring_run_id"]))
    eligible = int(report["cohort"]["eligible_prediction_count"])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{label}</title></head><body><h1>{label}</h1>"
        f"<p>Monitoring run: {run_id}</p><p>Eligible predictions: {eligible}</p>"
        "<p>See report.json for complete reproducible metrics and metadata.</p>"
        "</body></html>"
    ).encode("utf-8")


def _probabilities(
    rows: tuple[PerformanceRecord, ...]
) -> tuple[np.ndarray | None, str]:
    if not rows:
        return None, "empty matured cohort"
    values = [row.prediction_probability for row in rows]
    if any(value is None for value in values):
        return None, "one or more prediction probabilities are unavailable"
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        return None, "prediction probabilities must be finite values in [0, 1]"
    return result, ""


def _binary_prediction(value: int | str | bool) -> int:
    if value in (1, True, "1", "true", "True"):
        return 1
    if value in (0, False, "0", "false", "False"):
        return 0
    raise ValueError(f"predicted_class is not binary: {value!r}")


def _bin_boundaries(values: np.ndarray, count: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, count + 1)
    if strategy == "quantile":
        boundaries = np.unique(np.quantile(values, np.linspace(0, 1, count + 1)))
        if len(boundaries) < 2:
            return np.asarray([0.0, 1.0])
        boundaries[0], boundaries[-1] = 0.0, 1.0
        return boundaries
    raise ValueError("binning_strategy must be uniform or quantile")


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _outcome_watermark_complete(
    watermark: Mapping[str, Any], horizon_end: datetime
) -> bool:
    required = tuple(watermark.get("required_sources", ()))
    completeness = watermark.get("source_complete_through")
    if not required or not isinstance(completeness, Mapping):
        return False
    for source in required:
        value = completeness.get(source)
        if value is None:
            return False
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return False
        if not isinstance(value, datetime) or value.tzinfo is None:
            return False
        if value.astimezone(horizon_end.tzinfo) < horizon_end:
            return False
    return True
