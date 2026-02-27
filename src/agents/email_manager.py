from __future__ import annotations

from typing import Any

try:
    from agents import function_tool
except Exception:
    def function_tool(func):
        return func

from src.agents.formatter_agents import html_tool as _formatter_html_tool
from src.agents.formatter_agents import subject_tool as _formatter_subject_tool
from src.agents.tools_email import send_email_html as _send_email_html_tool

handoff_description = "Convert an email to HTML and send it"
STRICT_INSTRUCTION = (
    "You must execute exactly one send attempt per invocation. "
    "Flow is deterministic and fixed: subject_tool once, html_tool once, send_email_html once. "
    "Never retry send_email_html."
)


@function_tool
def subject_tool(body_text: str, recipients: list[str], context: dict[str, Any] | None = None) -> str:  # noqa: ARG001
    return _formatter_subject_tool(body_text)


@function_tool
def html_tool(
    subject: str,  # noqa: ARG001
    body_text: str,
    recipients: list[str],  # noqa: ARG001
    context: dict[str, Any] | None = None,  # noqa: ARG001
) -> str:
    return _formatter_html_tool(body_text)


@function_tool
def send_email_html(
    subject: str,
    html: str,
    recipients: list[str],
    from_name: str | None = None,  # noqa: ARG001
    from_email: str | None = None,
    metadata: dict[str, Any] | None = None,  # noqa: ARG001
):
    return _send_email_html_tool(
        subject=subject,
        body_html=html,
        to_emails=recipients,
        from_email=from_email,
    )


def _derive_send_status(send_result: Any) -> str | None:
    if send_result is None:
        return None

    if isinstance(send_result, dict):
        status = send_result.get("status")
        if status is not None:
            return str(status)

        if "ok" in send_result:
            return "sent" if bool(send_result["ok"]) else "error"

        status_code = send_result.get("status_code")
        if status_code is not None:
            try:
                return "sent" if 200 <= int(status_code) < 300 else "error"
            except (TypeError, ValueError):
                return str(status_code)

    return "sent"


def emailer_agent(
    body_text: str,
    recipients: list[str],
    context: dict[str, Any] | None = None,
    from_name: str | None = None,
    from_email: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subject = ""
    html = ""
    send_result: Any = None
    send_calls = 0
    recipient_list = list(recipients)

    try:
        subject = subject_tool(
            body_text=body_text,
            recipients=recipient_list,
            context=context,
        )
        html = html_tool(
            subject=subject,
            body_text=body_text,
            recipients=recipient_list,
            context=context,
        )
    except Exception as exc:
        return {
            "status": "error",
            "subject": subject,
            "html": html,
            "recipients": recipient_list,
            "send_status": None,
            "send_result": None,
            "errors": [str(exc)],
        }

    try:
        if send_calls >= 1:
            raise RuntimeError("send_email_html may only be called once per invocation")
        send_calls += 1
        send_result = send_email_html(
            subject=subject,
            html=html,
            recipients=recipient_list,
            from_name=from_name,
            from_email=from_email,
            metadata=metadata,
        )
    except Exception as exc:
        return {
            "status": "error",
            "subject": subject,
            "html": html,
            "recipients": recipient_list,
            "send_status": _derive_send_status(send_result),
            "send_result": send_result,
            "errors": [str(exc)],
        }

    return {
        "status": "sent",
        "subject": subject,
        "html": html,
        "recipients": recipient_list,
        "send_status": _derive_send_status(send_result),
        "send_result": send_result,
        "errors": [],
    }


__all__ = [
    "handoff_description",
    "STRICT_INSTRUCTION",
    "subject_tool",
    "html_tool",
    "send_email_html",
    "emailer_agent",
]
