# Shared services package for API/business logic extraction.

from .outreach_service import build_outreach_payload, render_prompt, select_targets

__all__ = [
    "build_outreach_payload",
    "render_prompt",
    "select_targets",
]
