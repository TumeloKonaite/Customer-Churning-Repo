from __future__ import annotations

import re
from typing import Any

try:
    from agents import function_tool
except Exception:
    def function_tool(func):
        return func

from src.agents.email_manager import emailer_agent
from src.agents.picker import pick_best_sales_email
from src.agents.retention_writers import (
    write_retention_email_concise,
    write_retention_email_serious,
    write_retention_email_witty,
)

handoff_description = "Write three draft emails, pick the best one, and hand off to email manager"
STRICT_INSTRUCTION = (
    "Deterministic pipeline: call each writer tool exactly once, "
    "call picker exactly once, and call emailer_agent exactly once."
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SUBJECT_PREFIX_RE = re.compile(r"^\s*subject\s*:", re.IGNORECASE)


def _sanitize_plain_text_body(text: str) -> str:
    if not isinstance(text, str):
        return ""

    without_html = _HTML_TAG_RE.sub("", text)
    lines: list[str] = []
    for raw_line in without_html.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if _SUBJECT_PREFIX_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _require_recipients(recipients: list[str]) -> list[str]:
    if not isinstance(recipients, list) or not recipients:
        raise ValueError("'recipients' must be a non-empty list of email strings")

    normalized: list[str] = []
    for i, recipient in enumerate(recipients):
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError(f"'recipients[{i}]' must be a non-empty string")
        normalized.append(recipient.strip())
    return normalized


def _merge_context(
    context: dict[str, Any] | None,
    *,
    company_name: str | None = None,
    from_name: str | None = None,
    from_email: str | None = None,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if context is not None:
        if not isinstance(context, dict):
            raise ValueError("'context' must be a dictionary when provided")
        merged.update(context)

    optionals = {
        "company_name": company_name,
        "from_name": from_name,
        "from_email": from_email,
    }
    for key, value in optionals.items():
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            merged[key] = value_text

    return merged or None


def _generate_drafts(message_prompt: str, writer_context: dict[str, Any] | None) -> tuple[str, str, str]:
    draft_serious = _sanitize_plain_text_body(
        write_retention_email_serious(prompt=message_prompt, context=writer_context)
    )
    draft_witty = _sanitize_plain_text_body(
        write_retention_email_witty(prompt=message_prompt, context=writer_context)
    )
    draft_concise = _sanitize_plain_text_body(
        write_retention_email_concise(prompt=message_prompt, context=writer_context)
    )

    drafts = (draft_serious, draft_witty, draft_concise)
    if any(not draft for draft in drafts):
        raise RuntimeError("All writer tools must return non-empty plain-text draft bodies")
    return drafts


@function_tool
def sales_manager(
    message_prompt: str,
    recipients: list[str],
    context: dict[str, Any] | None = None,
    company_name: str | None = None,
    from_name: str | None = None,
    from_email: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_text = _require_non_empty_str(message_prompt, "message_prompt")
    recipient_list = _require_recipients(recipients)
    writer_context = _merge_context(
        context,
        company_name=company_name,
        from_name=from_name,
        from_email=from_email,
    )

    draft_serious = ""
    draft_witty = ""
    draft_concise = ""
    selected_draft = ""

    try:
        draft_serious, draft_witty, draft_concise = _generate_drafts(prompt_text, writer_context)
        selected_draft = pick_best_sales_email(
            drafts=[draft_serious, draft_witty, draft_concise],
            context=writer_context,
        )
        selected_draft = _sanitize_plain_text_body(selected_draft)
        if not selected_draft:
            raise RuntimeError("Picker returned an empty draft")
    except Exception as exc:
        return {
            "status": "error",
            "selected_draft": selected_draft,
            "drafts": {
                "serious": draft_serious,
                "witty": draft_witty,
                "concise": draft_concise,
            },
            "recipients": recipient_list,
            "handoff_result": None,
            "errors": [str(exc)],
        }

    try:
        handoff_result = emailer_agent(
            body_text=selected_draft,
            recipients=recipient_list,
            context=writer_context,
            from_name=from_name,
            from_email=from_email,
            metadata=metadata,
        )
    except Exception as exc:
        return {
            "status": "error",
            "selected_draft": selected_draft,
            "drafts": {
                "serious": draft_serious,
                "witty": draft_witty,
                "concise": draft_concise,
            },
            "recipients": recipient_list,
            "handoff_result": None,
            "errors": [str(exc)],
        }

    handoff_status = None
    if isinstance(handoff_result, dict):
        handoff_status = handoff_result.get("status")

    return {
        "status": str(handoff_status or "sent"),
        "selected_draft": selected_draft,
        "drafts": {
            "serious": draft_serious,
            "witty": draft_witty,
            "concise": draft_concise,
        },
        "recipients": recipient_list,
        "handoff_result": handoff_result,
        "errors": [],
    }


__all__ = [
    "handoff_description",
    "STRICT_INSTRUCTION",
    "sales_manager",
]
