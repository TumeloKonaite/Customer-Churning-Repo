from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from agents import Agent, Runner, function_tool
except Exception:
    @dataclass(slots=True)
    class Agent:  # type: ignore[override]
        name: str
        instructions: str

    @dataclass(slots=True)
    class _RunnerResult:
        final_output: str

    class Runner:  # type: ignore[override]
        @staticmethod
        def run_sync(agent: Agent, input: str) -> _RunnerResult:  # noqa: ARG004
            return _RunnerResult(final_output=input)

    def function_tool(func):
        return func


TOOL_NAME = "pick_best_sales_email"
TOOL_DESCRIPTION = "Pick the best cold sales/retention email draft"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SUBJECT_PREFIX_RE = re.compile(r"^\s*subject\s*:", re.IGNORECASE)
_DRAFT_INDEX_RE = re.compile(r"\b(?:draft|option)?\s*([123])\b", re.IGNORECASE)


def _final_output_to_text(result: Any) -> str:
    final_output = getattr(result, "final_output", result)
    if final_output is None:
        return ""
    return str(final_output).strip()


def _validate_and_normalize_draft(value: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'drafts[{index}]' must be a non-empty string")
    if "<" in value or ">" in value or _HTML_TAG_RE.search(value):
        raise ValueError(f"'drafts[{index}]' must not contain HTML")

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"'drafts[{index}]' must be a non-empty string")
    if any(_SUBJECT_PREFIX_RE.match(line) for line in lines):
        raise ValueError(f"'drafts[{index}]' must not contain a subject line")

    normalized: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            if normalized and normalized[-1]:
                normalized.append("")
            continue
        normalized.append(line)

    return "\n".join(normalized).strip()


def _validate_and_normalize_drafts(drafts: list[str]) -> list[str]:
    if not isinstance(drafts, list) or len(drafts) != 3:
        raise ValueError("'drafts' must contain exactly 3 draft bodies")
    return [_validate_and_normalize_draft(draft, i) for i, draft in enumerate(drafts)]


def _normalize_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("'context' must be a dictionary when provided")
    normalized: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            normalized[key_text] = value_text
    return normalized or None


def _compose_picker_input(drafts: list[str], context: dict[str, Any] | None) -> str:
    parts = [
        "Pick the strongest draft for cold sales/retention outreach.",
        "Return exactly one full draft body from the options below with no extra text.",
        "",
        f"Draft 1:\n{drafts[0]}",
        "",
        f"Draft 2:\n{drafts[1]}",
        "",
        f"Draft 3:\n{drafts[2]}",
    ]

    if context:
        context_lines = [f"{key}: {value}" for key, value in context.items()]
        parts.extend(["", "Context:", *context_lines])

    return "\n".join(parts)


def _resolve_selected_draft(raw_output: str, drafts: list[str]) -> str:
    candidate = raw_output.strip()
    if candidate in drafts:
        return candidate

    match = _DRAFT_INDEX_RE.search(candidate)
    if match:
        return drafts[int(match.group(1)) - 1]

    lowered = candidate.lower()
    if "first" in lowered:
        return drafts[0]
    if "second" in lowered:
        return drafts[1]
    if "third" in lowered:
        return drafts[2]

    return drafts[0]


sales_picker = Agent(
    name="Sales Picker",
    instructions=(
        "Choose the best cold sales/retention email body from three draft options. "
        "Return only one full draft body with no explanation, no JSON, no markdown, no label, and no subject."
    ),
)


@function_tool
def pick_best_sales_email(drafts: list[str], context: dict[str, Any] | None = None) -> str:
    normalized_drafts = _validate_and_normalize_drafts(drafts)
    normalized_context = _normalize_context(context)
    picker_input = _compose_picker_input(normalized_drafts, normalized_context)
    result = Runner.run_sync(sales_picker, picker_input)
    raw_output = _final_output_to_text(result)
    return _resolve_selected_draft(raw_output, normalized_drafts)


for attr, value in (
    ("__name__", TOOL_NAME),
    ("__qualname__", TOOL_NAME),
    ("tool_name", TOOL_NAME),
    ("tool_description", TOOL_DESCRIPTION),
):
    try:
        setattr(pick_best_sales_email, attr, value)
    except Exception:
        pass


__all__ = [
    "TOOL_NAME",
    "TOOL_DESCRIPTION",
    "sales_picker",
    "pick_best_sales_email",
]
