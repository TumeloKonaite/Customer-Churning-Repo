from __future__ import annotations

from src.adapters.email_sendgrid import SendgridEmailClient

try:
    from agents import function_tool
except Exception:
    def function_tool(func):
        return func


@function_tool
def send_email_text(
    subject: str,
    body_text: str,
    to_emails: list[str],
    from_email: str | None = None,
):
    client = SendgridEmailClient()
    return client.send_text(
        subject=subject,
        body_text=body_text,
        to_emails=to_emails,
        from_email=from_email,
    )


@function_tool
def send_email_html(
    subject: str,
    body_html: str,
    to_emails: list[str],
    from_email: str | None = None,
):
    client = SendgridEmailClient()
    return client.send_html(
        subject=subject,
        body_html=body_html,
        to_emails=to_emails,
        from_email=from_email,
    )
