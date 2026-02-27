from __future__ import annotations

import os
import re
from typing import Any

import requests


_SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _validate_email(value: str, field_name: str) -> str:
    email = _require_non_empty_str(value, field_name)
    if not _EMAIL_RE.match(email):
        raise ValueError(f"'{field_name}' must be a valid email address")
    return email


class SendgridEmailClient:
    """Standalone SendGrid mail sender with env-driven defaults."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        verified_sender: str | None = None,
        api_url: str = _SENDGRID_API_URL,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("SENDGRID_API_KEY", "")).strip()
        self.verified_sender = (
            verified_sender if verified_sender is not None else os.getenv("SENDGRID_VERIFIED_SENDER", "")
        ).strip()
        self.api_url = api_url

    def send_text(
        self,
        subject: str,
        body_text: str,
        to_emails: list[str],
        from_email: str | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(
            subject=subject,
            body=body_text,
            to_emails=to_emails,
            content_type="text/plain",
            from_email=from_email,
        )
        return self._send_payload(payload)

    def send_html(
        self,
        subject: str,
        body_html: str,
        to_emails: list[str],
        from_email: str | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(
            subject=subject,
            body=body_html,
            to_emails=to_emails,
            content_type="text/html",
            from_email=from_email,
        )
        return self._send_payload(payload)

    def _normalize_recipients(self, to_emails: list[str]) -> list[str]:
        if not isinstance(to_emails, list) or not to_emails:
            raise ValueError("'to_emails' must be a non-empty list of email addresses")

        normalized: list[str] = []
        for idx, raw in enumerate(to_emails):
            normalized.append(_validate_email(raw, f"to_emails[{idx}]"))
        return normalized

    def _resolve_sender(self, from_email: str | None) -> str:
        sender = from_email if from_email else self.verified_sender
        if not sender:
            raise ValueError("A sender email is required (provide from_email or SENDGRID_VERIFIED_SENDER)")
        return _validate_email(sender, "from_email")

    def _build_payload(
        self,
        *,
        subject: str,
        body: str,
        to_emails: list[str],
        content_type: str,
        from_email: str | None,
    ) -> dict[str, Any]:
        normalized_subject = _require_non_empty_str(subject, "subject")
        normalized_body = _require_non_empty_str(body, "body")
        normalized_to = self._normalize_recipients(to_emails)
        sender = self._resolve_sender(from_email)

        return {
            "from": {"email": sender},
            "personalizations": [
                {
                    "to": [{"email": email} for email in normalized_to],
                }
            ],
            "subject": normalized_subject,
            "content": [
                {
                    "type": content_type,
                    "value": normalized_body,
                }
            ],
        }

    def _send_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("SENDGRID_API_KEY is required")

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"SendGrid send failed ({response.status_code}): {response.text}")

        return {
            "status_code": response.status_code,
            "ok": True,
            "body": response.text,
        }
