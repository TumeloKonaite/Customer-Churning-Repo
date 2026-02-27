from __future__ import annotations

import html
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


_SUBJECT_PREFIX_RE = re.compile(r"^\s*subject\s*:\s*", re.IGNORECASE)


def _final_output_to_text(result: Any) -> str:
    final_output = getattr(result, "final_output", result)
    if final_output is None:
        return ""
    return str(final_output).strip()


def _extract_subject(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[0] if lines else text.strip()
    candidate = _SUBJECT_PREFIX_RE.sub("", candidate).strip()
    return re.sub(r"<[^>]+>", "", candidate).strip()


def _extract_html(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    first_tag_index = normalized.find("<")
    if first_tag_index >= 0:
        return normalized[first_tag_index:].strip()

    return f"<p>{html.escape(normalized)}</p>"


subject_writer = Agent(
    name="Subject Writer",
    instructions=(
        "Write one outreach email subject line from the input draft. "
        "Return only subject text, with no quotes, no markdown, and no HTML."
    ),
)


html_converter = Agent(
    name="HTML Converter",
    instructions=(
        "Convert the input outreach draft to email-ready HTML. "
        "Return only the HTML body content and do not include a subject line."
    ),
)


@function_tool
def subject_tool(draft_text: str) -> str:
    result = Runner.run_sync(subject_writer, draft_text)
    return _extract_subject(_final_output_to_text(result))


@function_tool
def html_tool(draft_text: str) -> str:
    result = Runner.run_sync(html_converter, draft_text)
    return _extract_html(_final_output_to_text(result))


__all__ = [
    "subject_writer",
    "subject_tool",
    "html_converter",
    "html_tool",
]
