from __future__ import annotations

import src.agents.picker as picker


class _StubResult:
    def __init__(self, final_output: str) -> None:
        self.final_output = final_output


def test_picker_returns_one_plain_text_body_without_explanation(monkeypatch):
    drafts = [
        "Hi Alex,\nThanks for being with us. Reply with your top blocker and I will send a plan.",
        "Hi Alex,\nNo confetti cannon, just a quick check-in. Share one issue and I will help resolve it.",
        "Hi Alex,\nQuick check-in. Reply with one issue and I will send next steps today.",
    ]

    class StubRunner:
        @staticmethod
        def run_sync(agent, input_text: str):  # noqa: ARG001
            return _StubResult("Selected: Draft 2\nBecause it is clearer.")

    monkeypatch.setattr(picker, "Runner", StubRunner)

    body = picker.pick_best_sales_email(
        drafts=drafts,
        context={"company_name": "Example Co"},
    )

    assert isinstance(body, str)
    assert body == drafts[1]
    assert body.strip()
    assert "<" not in body and ">" not in body
    assert "Subject:" not in body
    assert "Selected:" not in body
    assert "Because" not in body
