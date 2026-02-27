from __future__ import annotations

import math
import re
from typing import Any, Mapping


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PROBABILITY_KEYS = ("p_churn", "churn_probability", "probability")

DEFAULT_FROM_EMAIL = "no-reply@example.com"
DEFAULT_TONE_POLICY = "friendly-and-direct"
DEFAULT_SEND_MODE = "dry_run"


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _coerce_probability(row: Mapping[str, Any]) -> float | None:
    for key in _PROBABILITY_KEYS:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return None


def _normalize_email(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if not _EMAIL_RE.match(normalized):
        return None
    return normalized


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_index(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_results(batch_results: Any) -> list[Mapping[str, Any]]:
    if isinstance(batch_results, list):
        return [row for row in batch_results if isinstance(row, Mapping)]

    if isinstance(batch_results, Mapping):
        results = batch_results.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, Mapping)]

    raise ValueError("'batch_results' must be a list of objects or an envelope containing 'results'")


def select_targets(
    batch_results: Any,
    threshold: float,
    max_n: int,
    require_email: bool = True,
) -> list[dict[str, Any]]:
    """Select outreach targets from batch prediction results."""
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("'threshold' must be numeric") from exc

    try:
        max_targets = int(max_n)
    except (TypeError, ValueError) as exc:
        raise ValueError("'max_n' must be an integer") from exc
    if max_targets <= 0:
        return []

    rows = _coerce_results(batch_results)
    ranked: list[tuple[float, tuple[int, Any], dict[str, Any]]] = []
    for row_position, row in enumerate(rows):
        p_churn = _coerce_probability(row)
        if p_churn is None or p_churn < threshold_value:
            continue

        normalized_email = _normalize_email(row.get("email"))
        if require_email and normalized_email is None:
            continue

        index_value = _resolve_index(row.get("index"), row_position)
        raw_id = _normalize_optional_text(row.get("id"))
        if raw_id is None:
            target_id = f"idx-{index_value}"
            tie_key: tuple[int, Any] = (1, index_value)
        else:
            target_id = raw_id
            tie_key = (0, target_id)

        target: dict[str, Any] = {
            "id": target_id,
            "email": normalized_email,
            "metadata": {
                "index": index_value,
                "p_churn": p_churn,
            },
        }

        name = _normalize_optional_text(row.get("name"))
        if name is not None:
            target["name"] = name

        if isinstance(row.get("metadata"), Mapping):
            merged_metadata = dict(row["metadata"])
            merged_metadata.update(target["metadata"])
            target["metadata"] = merged_metadata

        ranked.append((-p_churn, tie_key, target))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max_targets]]


def render_prompt(template: str, **kwargs: Any) -> str:
    template_text = _require_non_empty_str(template, "prompt_template")
    try:
        rendered = template_text.format(**kwargs)
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise ValueError(f"prompt template is missing required variable '{missing_key}'") from exc
    rendered_text = rendered.strip()
    if not rendered_text:
        raise ValueError("'message_prompt' must be non-empty after rendering")
    return rendered_text


def _normalize_target(target: Any, fallback_index: int) -> dict[str, Any]:
    if isinstance(target, Mapping):
        source = target
    else:
        source = {
            "id": getattr(target, "id", None),
            "email": getattr(target, "email", None),
            "name": getattr(target, "name", None),
            "metadata": getattr(target, "metadata", None),
        }

    target_id = _normalize_optional_text(source.get("id")) or f"idx-{fallback_index}"
    normalized_email = _normalize_email(source.get("email"))
    normalized_name = _normalize_optional_text(source.get("name"))

    normalized_target: dict[str, Any] = {
        "id": target_id,
        "email": normalized_email,
    }
    if normalized_name is not None:
        normalized_target["name"] = normalized_name
    if isinstance(source.get("metadata"), Mapping):
        normalized_target["metadata"] = dict(source["metadata"])
    return normalized_target


def build_outreach_payload(
    targets: list[Any],
    from_name: str,
    company_name: str,
    prompt_template: str,
) -> dict[str, Any]:
    """Build a deterministic payload compatible with outreach contracts."""
    normalized_from_name = _require_non_empty_str(from_name, "from_name")
    normalized_company_name = _require_non_empty_str(company_name, "company_name")

    if not isinstance(targets, list) or not targets:
        raise ValueError("'targets' must be a non-empty list")

    normalized_targets = [_normalize_target(target, index) for index, target in enumerate(targets)]
    prompt_context = {
        "from_name": normalized_from_name,
        "company_name": normalized_company_name,
        "recipient_count": len(normalized_targets),
        "recipient_ids": ", ".join(target["id"] for target in normalized_targets),
    }
    message_prompt = render_prompt(prompt_template, **prompt_context)

    return {
        "company_name": normalized_company_name,
        "from_name": normalized_from_name,
        "recipients": normalized_targets,
        "message_prompt": message_prompt,
        "from_email": DEFAULT_FROM_EMAIL,
        "tone_policy": DEFAULT_TONE_POLICY,
        "send_mode": DEFAULT_SEND_MODE,
    }


__all__ = [
    "DEFAULT_FROM_EMAIL",
    "DEFAULT_SEND_MODE",
    "DEFAULT_TONE_POLICY",
    "build_outreach_payload",
    "render_prompt",
    "select_targets",
]
