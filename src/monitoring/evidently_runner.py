"""A small compatibility boundary around Evidently's evolving public API."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd

from src.monitoring.models import MonitoringPolicy, ResultStatus


@dataclass(frozen=True, slots=True)
class EvidentlyOutput:
    html: bytes
    report: dict[str, Any]
    drift_summary: dict[str, Any]
    version: str
    configuration: dict[str, Any]


def _supported(factory: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return values
    if any(value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _old_api(policy: MonitoringPolicy, reference: pd.DataFrame, current: pd.DataFrame):
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    per_column_method = {
        feature: rule.drift_method
        for feature, rule in policy.feature_rules.items()
        if rule.drift_method and feature not in policy.excluded_features
    }
    per_column_method.update(
        {
            "prediction_probability": policy.prediction_probability_drift_method,
            "predicted_class": policy.predicted_class_drift_method,
        }
    )
    per_column_threshold = {
        feature: rule.drift_threshold
        for feature, rule in policy.feature_rules.items()
        if rule.drift_threshold and feature not in policy.excluded_features
    }
    per_column_threshold.update(
        {
            "prediction_probability": policy.prediction_probability_drift_threshold,
            "predicted_class": policy.predicted_class_drift_threshold,
        }
    )
    drift_config = {
        "stattest": None,
        "cat_stattest": policy.categorical_drift_method,
        "cat_stattest_threshold": policy.categorical_drift_threshold,
        "num_stattest": policy.numeric_drift_method,
        "num_stattest_threshold": policy.numeric_drift_threshold,
        "per_column_stattest": per_column_method,
        "per_column_stattest_threshold": per_column_threshold,
        "drift_share": policy.drift_dataset_share_threshold,
    }
    preset = DataDriftPreset(**_supported(DataDriftPreset, drift_config))
    report = Report(metrics=[preset, DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    return report, drift_config


def _new_api(policy: MonitoringPolicy, reference: pd.DataFrame, current: pd.DataFrame):
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    per_column_method = {
        **{
            feature: rule.drift_method
            for feature, rule in policy.feature_rules.items()
            if rule.drift_method and feature not in policy.excluded_features
        },
        "prediction_probability": policy.prediction_probability_drift_method,
        "predicted_class": policy.predicted_class_drift_method,
    }
    per_column_threshold = {
        **{
            feature: rule.drift_threshold
            for feature, rule in policy.feature_rules.items()
            if rule.drift_threshold and feature not in policy.excluded_features
        },
        "prediction_probability": policy.prediction_probability_drift_threshold,
        "predicted_class": policy.predicted_class_drift_threshold,
    }
    drift_config = {
        "drift_share": policy.drift_dataset_share_threshold,
        "cat_method": policy.categorical_drift_method,
        "cat_threshold": policy.categorical_drift_threshold,
        "num_method": policy.numeric_drift_method,
        "num_threshold": policy.numeric_drift_threshold,
        "per_column_method": per_column_method,
        "per_column_threshold": per_column_threshold,
    }
    preset = DataDriftPreset(**_supported(DataDriftPreset, drift_config))
    report = Report([preset, DataSummaryPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    return snapshot, drift_config


def _as_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "as_dict"):
        value = report.as_dict()
    elif hasattr(report, "dict"):
        value = report.dict()
    elif hasattr(report, "json"):
        value = json.loads(report.json())
    else:
        raise RuntimeError("Evidently report does not expose machine-readable output")
    if not isinstance(value, dict):
        raise RuntimeError("Evidently returned a non-object report")
    return value


def _stable_generated_ids(html: str) -> str:
    """Replace Evidently's presentation-only random IDs in encounter order."""
    patterns = (
        (re.compile(r"metric_[a-f0-9]{32}"), "metric_{:032x}"),
        (
            re.compile(
                r"\b[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-"
                r"[89ab][a-f0-9]{3}-[a-f0-9]{12}\b"
            ),
            "00000000-0000-7000-8000-{:012x}",
        ),
    )
    for pattern, template in patterns:
        replacements: dict[str, str] = {}

        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if value not in replacements:
                replacements[value] = template.format(len(replacements) + 1)
            return replacements[value]

        html = pattern.sub(replace, html)
    return html


def _save_html(report: Any) -> bytes:
    with tempfile.TemporaryDirectory(prefix="monitoring-report-") as temporary:
        path = Path(temporary) / "report.html"
        if hasattr(report, "save_html"):
            report.save_html(str(path))
        elif hasattr(report, "get_html"):
            path.write_text(report.get_html(), encoding="utf-8")
        else:
            raise RuntimeError("Evidently report does not expose HTML output")
        return _stable_generated_ids(path.read_text(encoding="utf-8")).encode("utf-8")


def _find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_find_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values(child, key))
    return found


def _normalize_drift(report: dict[str, Any], policy: MonitoringPolicy) -> dict[str, Any]:
    drifted = [value for value in _find_values(report, "number_of_drifted_columns") if isinstance(value, int)]
    shares = [value for value in _find_values(report, "share_of_drifted_columns") if isinstance(value, (int, float))]
    flags = [value for value in _find_values(report, "dataset_drift") if isinstance(value, bool)]
    feature_results: dict[str, Any] = {}
    for metric in report.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        config = metric.get("config", {})
        metric_type = str(config.get("type", ""))
        if metric_type.endswith("DriftedColumnsCount") and isinstance(metric.get("value"), dict):
            value = metric["value"]
            if isinstance(value.get("count"), (int, float)):
                drifted.insert(0, int(value["count"]))
            if isinstance(value.get("share"), (int, float)):
                shares.insert(0, float(value["share"]))
        if not metric_type.endswith("ValueDrift") or not config.get("column"):
            continue
        column = str(config["column"])
        score = metric.get("value")
        threshold = config.get("threshold")
        policy_method = (
            policy.prediction_probability_drift_method
            if column == "prediction_probability"
            else policy.predicted_class_drift_method
            if column == "predicted_class"
            else policy.feature_rules[column].drift_method
            or (
                policy.categorical_drift_method
                if policy.feature_rules[column].data_type in {"string", "category"}
                else policy.numeric_drift_method
            )
        )
        evidently_method = config.get("method")
        configured_method = str(evidently_method or policy_method)
        p_value_methods = {
            "ks", "chisquare", "z", "fisher_exact", "cramer_von_mises",
            "anderson", "mannw", "t_test", "es",
        }
        detected = None
        finite_score = (
            isinstance(score, (int, float)) and math.isfinite(float(score))
        )
        if finite_score and isinstance(threshold, (int, float)):
            method_key = configured_method.casefold().replace("-", "_").replace(" ", "_")
            detected = (
                float(score) < float(threshold)
                if configured_method in p_value_methods or "p_value" in method_key
                else float(score) > float(threshold)
            )
        feature_results[column] = {
            "status": (
                ResultStatus.WARNING if detected else ResultStatus.PASS
                if detected is not None else ResultStatus.NOT_EVALUATED
            ),
            "drift_detected": detected,
            "score": float(score) if finite_score else None,
            "method": configured_method,
            "policy_method": policy_method,
            "threshold": float(threshold) if isinstance(threshold, (int, float)) else None,
            "suppressed": column in policy.suppressed_features,
            "excluded": column in policy.excluded_features,
        }
    dataset_drift = flags[0] if flags else bool(shares and shares[0] >= policy.drift_dataset_share_threshold)
    return {
        "status": ResultStatus.WARNING if dataset_drift else ResultStatus.PASS,
        "dataset_drift": dataset_drift,
        "drifted_feature_count": drifted[0] if drifted else None,
        "drifted_feature_share": shares[0] if shares else None,
        "dataset_share_threshold": policy.drift_dataset_share_threshold,
        "feature_results": feature_results,
        "suppressed_features": list(policy.suppressed_features),
        "excluded_features": list(policy.excluded_features),
    }


def run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    policy: MonitoringPolicy,
) -> EvidentlyOutput:
    """Run drift and data-summary presets; imports stay out of inference startup."""
    import evidently

    columns = [
        feature
        for feature in policy.feature_rules
        if feature not in policy.excluded_features
    ] + ["prediction_probability", "predicted_class"]
    reference_input = reference[[column for column in columns if column in reference.columns]]
    current_input = current[[column for column in columns if column in current.columns]]

    try:
        report, effective_config = _new_api(policy, reference_input, current_input)
    except (ImportError, ModuleNotFoundError):
        report, effective_config = _old_api(policy, reference_input, current_input)
    machine_report = _as_dict(report)
    return EvidentlyOutput(
        html=_save_html(report),
        report=machine_report,
        drift_summary=_normalize_drift(machine_report, policy),
        version=str(getattr(evidently, "__version__", "unknown")),
        configuration={
            "numeric": {
                "method": policy.numeric_drift_method,
                "threshold": policy.numeric_drift_threshold,
            },
            "categorical": {
                "method": policy.categorical_drift_method,
                "threshold": policy.categorical_drift_threshold,
            },
            "prediction_probability": {
                "method": policy.prediction_probability_drift_method,
                "threshold": policy.prediction_probability_drift_threshold,
            },
            "predicted_class": {
                "method": policy.predicted_class_drift_method,
                "threshold": policy.predicted_class_drift_threshold,
            },
            "dataset_share_threshold": policy.drift_dataset_share_threshold,
            "effective_preset_arguments": effective_config,
        },
    )
