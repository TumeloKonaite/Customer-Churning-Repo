from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Callable, Mapping, TypedDict

from src.agents.sales_manager import sales_manager as default_sales_manager
from src.schemas import OutreachRequest, Target
from src.services.outreach_service import (
    DEFAULT_FROM_EMAIL,
    DEFAULT_TONE_POLICY,
    build_outreach_payload,
    select_targets,
)


DEFAULT_PROMPT_TEMPLATE = (
    "Write a retention outreach email from {from_name} at {company_name} "
    "for {recipient_count} recipients: {recipient_ids}."
)
_ALLOWED_BATCH_STATUSES = {"success", "partial", "failed", "error"}
_SENT_LIKE_STATUSES = {"sent", "queued", "accepted", "ok", "success"}


class OutreachSummary(TypedDict):
    total_rows: int
    valid_predictions: int
    selected: int
    drafted: int
    sent: int


class OutreachReport(TypedDict):
    status: str
    selected_targets: list[dict[str, Any]]
    outreach_request: dict[str, Any] | None
    outreach_result: dict[str, Any] | None
    summary: OutreachSummary
    errors: list[Any]


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _new_report(
    *,
    status: str,
    selected_targets: list[dict[str, Any]],
    outreach_request: dict[str, Any] | None,
    outreach_result: dict[str, Any] | None,
    total_rows: int,
    valid_predictions: int,
    errors: list[Any],
) -> OutreachReport:
    drafted = 0
    sent = 0
    if selected_targets and outreach_result and str(outreach_result.get("status", "")).lower() != "error":
        drafted = len(selected_targets)
        send_status = str(outreach_result.get("send_status", "")).lower()
        send_mode = str(outreach_result.get("send_mode", "")).lower()
        if send_mode == "send" and send_status in _SENT_LIKE_STATUSES:
            sent = len(selected_targets)

    return {
        "status": status,
        "selected_targets": selected_targets,
        "outreach_request": outreach_request,
        "outreach_result": outreach_result,
        "summary": {
            "total_rows": total_rows,
            "valid_predictions": valid_predictions,
            "selected": len(selected_targets),
            "drafted": drafted,
            "sent": sent,
        },
        "errors": errors,
    }


def _coerce_probability(row: Mapping[str, Any]) -> float | None:
    raw = row.get("p_churn")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _is_usable_prediction(row: Mapping[str, Any]) -> bool:
    if _coerce_probability(row) is not None:
        return True
    action = row.get("recommended_action")
    if action is None:
        return False
    return bool(str(action).strip())


def _normalize_send_mode(dry_run: bool) -> str:
    return "dry_run" if dry_run else "send"


def _normalize_id_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _target_sort_key(target: Mapping[str, Any]) -> tuple[int, Any]:
    target_id = _normalize_id_text(target.get("id"))
    if target_id and not target_id.startswith("idx-"):
        return (0, target_id)

    metadata = target.get("metadata")
    index_value = None
    if isinstance(metadata, Mapping):
        index_value = _coerce_int(metadata.get("index"))
    if index_value is None and target_id.startswith("idx-"):
        index_value = _coerce_int(target_id.removeprefix("idx-"))
    if index_value is not None:
        return (1, index_value)
    if target_id:
        return (2, target_id)
    return (3, "")


def _resolve_selected_draft_key(drafts: Mapping[str, Any], selected_draft_text: str) -> str:
    selected = selected_draft_text.strip()
    for key in ("serious", "witty", "concise"):
        draft_value = drafts.get(key)
        if isinstance(draft_value, str) and draft_value.strip() == selected:
            return key
    return "serious"


def _normalize_sales_result(sales_result: Any, *, send_mode: str) -> dict[str, Any]:
    if not isinstance(sales_result, Mapping):
        return {
            "status": "error",
            "send_mode": send_mode,
            "selected_draft": "serious",
            "subject": "",
            "html": "",
            "send_status": None,
            "drafts": {},
            "raw": sales_result,
            "errors": ["Sales manager returned a non-dictionary result"],
        }

    drafts = sales_result.get("drafts")
    drafts_dict = dict(drafts) if isinstance(drafts, Mapping) else {}
    selected_draft_text = str(sales_result.get("selected_draft") or "")
    selected_draft_key = _resolve_selected_draft_key(drafts_dict, selected_draft_text)

    handoff = sales_result.get("handoff_result")
    handoff_dict = dict(handoff) if isinstance(handoff, Mapping) else {}
    errors = [str(err) for err in sales_result.get("errors", []) if err is not None] if isinstance(
        sales_result.get("errors"), list
    ) else []
    if isinstance(handoff_dict.get("errors"), list):
        errors.extend(str(err) for err in handoff_dict["errors"] if err is not None)

    return {
        "status": str(sales_result.get("status") or "error"),
        "send_mode": str(sales_result.get("send_mode") or send_mode),
        "selected_draft": selected_draft_key,
        "subject": str(handoff_dict.get("subject") or ""),
        "html": str(handoff_dict.get("html") or ""),
        "send_status": handoff_dict.get("send_status"),
        "drafts": drafts_dict,
        "raw": dict(sales_result),
        "errors": errors,
    }


def _require_config(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("'config' must be a mapping")
    return config


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _require_runner(config: Mapping[str, Any]) -> Callable[..., dict[str, Any]]:
    runner = config.get("sales_manager_runner")
    if runner is None:
        return default_sales_manager
    if not callable(runner):
        raise ValueError("'sales_manager_runner' must be callable when provided")
    return runner


def _safe_batch_errors(batch_response: Mapping[str, Any]) -> list[Any]:
    raw = batch_response.get("errors")
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _derive_status(*, batch_errors: bool, pipeline_errors: bool, selected: int) -> str:
    if pipeline_errors:
        return "partial" if selected > 0 else "error"
    if batch_errors:
        return "partial"
    return "ok"


def run_outreach_from_batch(batch_response: Any, config: Any) -> OutreachReport:
    base_errors: list[Any] = []

    envelope = _as_mapping(batch_response)
    if envelope is None:
        return _new_report(
            status="error",
            selected_targets=[],
            outreach_request=None,
            outreach_result=None,
            total_rows=0,
            valid_predictions=0,
            errors=["'batch_response' must be a mapping"],
        )

    batch_status = envelope.get("status")
    if not isinstance(batch_status, str) or batch_status not in _ALLOWED_BATCH_STATUSES:
        base_errors.append({"stage": "batch", "message": "'batch_response.status' is missing or unsupported"})

    raw_results = envelope.get("results")
    if not isinstance(raw_results, list):
        return _new_report(
            status="error",
            selected_targets=[],
            outreach_request=None,
            outreach_result=None,
            total_rows=0,
            valid_predictions=0,
            errors=base_errors + [{"stage": "batch", "message": "'batch_response.results' must be a list"}],
        )

    rows = [dict(row) for row in raw_results if isinstance(row, Mapping)]
    usable_rows = [row for row in rows if _is_usable_prediction(row)]
    total_rows = len(raw_results)
    valid_predictions = len(usable_rows)

    batch_errors = _safe_batch_errors(envelope)
    if batch_errors:
        base_errors.append({"stage": "batch", "errors": batch_errors})

    try:
        cfg = _require_config(config)
        company_name = _require_non_empty_string(cfg.get("company_name"), "company_name")
        from_name = _require_non_empty_string(cfg.get("from_name"), "from_name")
        from_email = str(cfg.get("from_email") or DEFAULT_FROM_EMAIL).strip()
        prompt_template = str(cfg.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE)
        tone_policy = str(cfg.get("tone_policy") or DEFAULT_TONE_POLICY).strip() or DEFAULT_TONE_POLICY

        threshold = float(cfg.get("threshold", 0.7))
        max_targets = int(cfg.get("max_targets", 50))
        require_email = bool(cfg.get("require_email", True))
        dry_run = bool(cfg.get("dry_run", True))
        send_mode = _normalize_send_mode(dry_run)
        sales_runner = _require_runner(cfg)

        allowed_actions_raw = cfg.get("recommended_actions")
        allowed_actions = None
        if isinstance(allowed_actions_raw, (list, tuple, set)):
            allowed_actions = {str(action).strip() for action in allowed_actions_raw if str(action).strip()}
        if allowed_actions:
            usable_rows = [row for row in usable_rows if str(row.get("recommended_action", "")).strip() in allowed_actions]
    except Exception as exc:
        return _new_report(
            status="error",
            selected_targets=[],
            outreach_request=None,
            outreach_result=None,
            total_rows=total_rows,
            valid_predictions=valid_predictions,
            errors=base_errors + [{"stage": "config", "message": str(exc)}],
        )

    try:
        selected_targets = select_targets(
            usable_rows,
            threshold=threshold,
            max_n=max_targets,
            require_email=require_email,
        )
    except Exception as exc:
        return _new_report(
            status="error",
            selected_targets=[],
            outreach_request=None,
            outreach_result=None,
            total_rows=total_rows,
            valid_predictions=valid_predictions,
            errors=base_errors + [{"stage": "target_selection", "message": str(exc)}],
        )

    selected_targets = sorted((dict(target) for target in selected_targets), key=_target_sort_key)
    if not selected_targets:
        report_status = _derive_status(
            batch_errors=bool(batch_errors),
            pipeline_errors=bool(base_errors),
            selected=0,
        )
        return _new_report(
            status=report_status,
            selected_targets=[],
            outreach_request=None,
            outreach_result=None,
            total_rows=total_rows,
            valid_predictions=valid_predictions,
            errors=base_errors,
        )

    pipeline_errors = list(base_errors)
    try:
        outreach_payload = build_outreach_payload(
            targets=selected_targets,
            from_name=from_name,
            company_name=company_name,
            prompt_template=prompt_template,
        )

        recipient_objects: list[Target] = []
        for recipient in outreach_payload["recipients"]:
            email = recipient.get("email")
            if not email:
                pipeline_errors.append(
                    {
                        "stage": "payload",
                        "message": "Skipping recipient without email",
                        "target_id": recipient.get("id"),
                    }
                )
                continue
            recipient_objects.append(
                Target(
                    id=str(recipient["id"]),
                    email=str(email),
                    name=recipient.get("name"),
                    metadata=recipient.get("metadata"),
                )
            )

        if not recipient_objects:
            raise ValueError("No recipients with valid email addresses remain after payload normalization")

        outreach_request_model = OutreachRequest(
            message_prompt=outreach_payload["message_prompt"],
            recipients=recipient_objects,
            from_name=outreach_payload["from_name"],
            from_email=from_email,
            company_name=outreach_payload["company_name"],
            tone_policy=tone_policy,
            send_mode=send_mode,
        )
    except Exception as exc:
        return _new_report(
            status=_derive_status(batch_errors=bool(batch_errors), pipeline_errors=True, selected=len(selected_targets)),
            selected_targets=selected_targets,
            outreach_request=None,
            outreach_result=None,
            total_rows=total_rows,
            valid_predictions=valid_predictions,
            errors=pipeline_errors + [{"stage": "request_build", "message": str(exc)}],
        )

    outreach_request = asdict(outreach_request_model)
    context = cfg.get("context")
    if context is not None and not isinstance(context, Mapping):
        pipeline_errors.append({"stage": "config", "message": "'context' must be a dictionary when provided"})
        context = None

    metadata = cfg.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        pipeline_errors.append({"stage": "config", "message": "'metadata' must be a dictionary when provided"})
        metadata = None

    if isinstance(metadata, Mapping):
        metadata_payload = dict(metadata)
    else:
        metadata_payload = {}
    metadata_payload["send_mode"] = send_mode
    metadata_payload["target_ids"] = [recipient.id for recipient in outreach_request_model.recipients]

    try:
        sales_result_raw = sales_runner(
            message_prompt=outreach_request_model.message_prompt,
            recipients=[recipient.email for recipient in outreach_request_model.recipients],
            context=dict(context) if isinstance(context, Mapping) else None,
            company_name=outreach_request_model.company_name,
            from_name=outreach_request_model.from_name,
            from_email=outreach_request_model.from_email,
            metadata=metadata_payload,
            send_mode=outreach_request_model.send_mode,
        )
    except Exception as exc:
        pipeline_errors.append({"stage": "sales_manager", "message": str(exc)})
        return _new_report(
            status=_derive_status(
                batch_errors=bool(batch_errors),
                pipeline_errors=True,
                selected=len(selected_targets),
            ),
            selected_targets=selected_targets,
            outreach_request=outreach_request,
            outreach_result=None,
            total_rows=total_rows,
            valid_predictions=valid_predictions,
            errors=pipeline_errors,
        )

    outreach_result = _normalize_sales_result(sales_result_raw, send_mode=send_mode)
    if outreach_result["status"].lower() == "error" or outreach_result["errors"]:
        pipeline_errors.append(
            {
                "stage": "sales_manager",
                "message": "Sales manager returned errors",
                "errors": outreach_result["errors"],
            }
        )

    report_status = _derive_status(
        batch_errors=bool(batch_errors),
        pipeline_errors=bool(pipeline_errors),
        selected=len(selected_targets),
    )
    return _new_report(
        status=report_status,
        selected_targets=selected_targets,
        outreach_request=outreach_request,
        outreach_result=outreach_result,
        total_rows=total_rows,
        valid_predictions=valid_predictions,
        errors=pipeline_errors,
    )


__all__ = [
    "OutreachReport",
    "OutreachSummary",
    "run_outreach_from_batch",
]
