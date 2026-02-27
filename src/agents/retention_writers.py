from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

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


TOOL_DESCRIPTION = "Write a cold sales/retention email"

_SUBJECT_PREFIX_RE = re.compile(r"^\s*subject\s*:\s*", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _final_output_to_text(result: Any) -> str:
    final_output = getattr(result, "final_output", result)
    if final_output is None:
        return ""
    return str(final_output).strip()


def _sanitize_plain_text_body(text: str) -> str:
    if not text:
        return ""

    without_html = _HTML_TAG_RE.sub("", text)
    lines: list[str] = []
    first_content_line = True

    for raw_line in without_html.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if first_content_line:
            line = _SUBJECT_PREFIX_RE.sub("", line).strip()
            first_content_line = False
        lines.append(line)

    return "\n".join(lines).strip()


def _normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    return prompt.strip()


def _merge_context(
    context: dict[str, Any] | None,
    *,
    company_name: str | None = None,
    from_name: str | None = None,
    from_email: str | None = None,
    recipient_name: str | None = None,
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
        "recipient_name": recipient_name,
    }
    for key, value in optionals.items():
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            merged[key] = value_text

    return merged or None


def _compose_input(prompt: str, context: dict[str, Any] | None) -> str:
    normalized_prompt = _normalize_prompt(prompt)
    if not context:
        return normalized_prompt

    lines: list[str] = []
    for key, value in context.items():
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            lines.append(f"{key}: {value_text}")

    if not lines:
        return normalized_prompt

    return f"{normalized_prompt}\n\nContext:\n" + "\n".join(lines)


def _set_tool_metadata(tool: Callable[..., str], *, name: str, description: str) -> Callable[..., str]:
    for attr, value in (
        ("__name__", name),
        ("__qualname__", name),
        ("__doc__", description),
        ("tool_name", name),
        ("tool_description", description),
    ):
        try:
            setattr(tool, attr, value)
        except Exception:
            pass
    return tool


class _BaseRetentionWriter:
    agent_name: str = ""
    persona_instruction: str = ""
    default_tool_name: str = ""

    def __init__(self) -> None:
        self.agent = Agent(name=self.agent_name, instructions=self._instructions())

    def _instructions(self) -> str:
        return (
            "You write cold sales and customer-retention outreach emails. "
            "Return exactly one plain-text email body and nothing else. "
            "Do not include a subject line, HTML, markdown, metadata labels, or commentary. "
            "Do not mention churn scoring or AI scoring unless explicitly requested in the input. "
            f"{self.persona_instruction}"
        )

    def write(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        composed_input = _compose_input(prompt, context)
        result = Runner.run_sync(self.agent, composed_input)
        body = _sanitize_plain_text_body(_final_output_to_text(result))
        if not body:
            raise RuntimeError("Writer returned an empty email body")
        return body

    def as_tool(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str = TOOL_DESCRIPTION,
    ) -> Callable[..., str]:
        writer = self
        resolved_name = tool_name or self.default_tool_name

        @function_tool
        def _tool(
            prompt: str,
            context: dict[str, Any] | None = None,
            company_name: str | None = None,
            from_name: str | None = None,
            from_email: str | None = None,
            recipient_name: str | None = None,
        ) -> str:
            merged_context = _merge_context(
                context,
                company_name=company_name,
                from_name=from_name,
                from_email=from_email,
                recipient_name=recipient_name,
            )
            return writer.write(prompt=prompt, context=merged_context)

        return _set_tool_metadata(
            _tool,
            name=resolved_name,
            description=tool_description,
        )


class SeriousRetentionWriter(_BaseRetentionWriter):
    agent_name = "Serious Retention Writer"
    persona_instruction = "Tone: professional, empathetic, and provide one clear next step."
    default_tool_name = "write_retention_email_serious"


class WittyRetentionWriter(_BaseRetentionWriter):
    agent_name = "Witty Retention Writer"
    persona_instruction = "Tone: warm and lightly witty while remaining professional and practical."
    default_tool_name = "write_retention_email_witty"


class ConciseRetentionWriter(_BaseRetentionWriter):
    agent_name = "Concise Retention Writer"
    persona_instruction = "Tone: short, direct, helpful, and action-oriented."
    default_tool_name = "write_retention_email_concise"


write_retention_email_serious = SeriousRetentionWriter().as_tool()
write_retention_email_witty = WittyRetentionWriter().as_tool()
write_retention_email_concise = ConciseRetentionWriter().as_tool()


__all__ = [
    "TOOL_DESCRIPTION",
    "SeriousRetentionWriter",
    "WittyRetentionWriter",
    "ConciseRetentionWriter",
    "write_retention_email_serious",
    "write_retention_email_witty",
    "write_retention_email_concise",
]
