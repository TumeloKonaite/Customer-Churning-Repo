from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DRAFT_KEYS = {"serious", "witty", "concise"}


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _validate_email(value: str, field_name: str) -> str:
    normalized = _require_non_empty_str(value, field_name)
    if not _EMAIL_RE.match(normalized):
        raise ValueError(f"'{field_name}' must be a valid email address")
    return normalized


@dataclass(slots=True)
class Target:
    id: str
    email: str
    name: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "id")
        self.email = _validate_email(self.email, "email")
        if self.name is not None:
            self.name = _require_non_empty_str(self.name, "name")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("'metadata' must be a dictionary when provided")


@dataclass(slots=True)
class DraftSet:
    serious: str
    witty: str
    concise: str

    def __post_init__(self) -> None:
        self.serious = _require_non_empty_str(self.serious, "serious")
        self.witty = _require_non_empty_str(self.witty, "witty")
        self.concise = _require_non_empty_str(self.concise, "concise")


@dataclass(slots=True)
class OutreachRequest:
    message_prompt: str
    recipients: list[Target]
    from_name: str
    from_email: str
    company_name: str
    tone_policy: str
    send_mode: str

    def __post_init__(self) -> None:
        self.message_prompt = _require_non_empty_str(self.message_prompt, "message_prompt")
        if not isinstance(self.recipients, list) or not self.recipients:
            raise ValueError("'recipients' must be a non-empty list of Target")
        if not all(isinstance(recipient, Target) for recipient in self.recipients):
            raise ValueError("'recipients' must contain only Target objects")
        self.from_name = _require_non_empty_str(self.from_name, "from_name")
        self.from_email = _validate_email(self.from_email, "from_email")
        self.company_name = _require_non_empty_str(self.company_name, "company_name")
        self.tone_policy = _require_non_empty_str(self.tone_policy, "tone_policy")
        self.send_mode = _require_non_empty_str(self.send_mode, "send_mode")


@dataclass(slots=True)
class OutreachResult:
    status: str
    selected_draft: str
    subject: str
    html: str
    send_status: str | None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = _require_non_empty_str(self.status, "status")
        self.selected_draft = _require_non_empty_str(self.selected_draft, "selected_draft")
        if self.selected_draft not in _DRAFT_KEYS:
            raise ValueError("'selected_draft' must be one of: serious, witty, concise")
        self.subject = _require_non_empty_str(self.subject, "subject")
        self.html = _require_non_empty_str(self.html, "html")
        if self.send_status is not None:
            self.send_status = _require_non_empty_str(self.send_status, "send_status")
        if not isinstance(self.errors, list) or not all(isinstance(error, str) for error in self.errors):
            raise ValueError("'errors' must be a list of strings")


__all__ = [
    "Target",
    "DraftSet",
    "OutreachRequest",
    "OutreachResult",
]
