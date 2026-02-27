from __future__ import annotations

import pytest

import src.agents.picker as picker


class _StubResult:
    def __init__(self, final_output: str) -> None:
        self.final_output = final_output


def test_picker_receives_three_drafts_in_expected_shape(monkeypatch):
    captured: dict[str, str] = {}
    drafts = [
        "Hi Alex,\nThanks for partnering with us. Reply with your top blocker and I will send a plan.",
        "Hi Alex,\nNo confetti cannon, just a quick check-in. Share one issue and I will help resolve it.",
        "Hi Alex,\nQuick check-in: reply with one issue and I will send next steps today.",
    ]

    class StubRunner:
        @staticmethod
        def run_sync(agent, input_text: str):
            assert agent is picker.sales_picker
            captured["input"] = input_text
            return _StubResult(drafts[1])

    monkeypatch.setattr(picker, "Runner", StubRunner)

    selected = picker.pick_best_sales_email(
        drafts=drafts,
        context={"company_name": "Example Co", "from_name": "Customer Success"},
    )

    assert selected == drafts[1]
    prompt = captured["input"]
    assert "Draft 1:" in prompt
    assert "Draft 2:" in prompt
    assert "Draft 3:" in prompt
    assert drafts[0] in prompt
    assert drafts[1] in prompt
    assert drafts[2] in prompt


def test_picker_rejects_non_three_draft_input():
    with pytest.raises(ValueError, match="exactly 3"):
        picker.pick_best_sales_email(drafts=["one", "two"])


def test_picker_rejects_html_drafts():
    with pytest.raises(ValueError, match="must not contain HTML"):
        picker.pick_best_sales_email(
            drafts=[
                "<p>Hi Alex</p>",
                "Hi Alex, checking in with a quick note.",
                "Hi Alex, quick follow-up from our side.",
            ]
        )


def test_picker_rejects_subject_lines():
    with pytest.raises(ValueError, match="subject line"):
        picker.pick_best_sales_email(
            drafts=[
                "Subject: Quick follow-up\nHi Alex, wanted to check in.",
                "Hi Alex, checking in with a quick note.",
                "Hi Alex, quick follow-up from our side.",
            ]
        )
