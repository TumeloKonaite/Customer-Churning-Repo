from __future__ import annotations

import importlib

sales_manager_module = importlib.import_module("src.agents.sales_manager")


def test_sales_manager_calls_writers_picker_and_handoff_once(monkeypatch):
    call_order: list[str] = []
    call_counts = {
        "serious": 0,
        "witty": 0,
        "concise": 0,
        "picker": 0,
        "handoff": 0,
    }
    recipients = ["a@example.com", "b@example.com"]
    draft_serious = "Hi Alex,\nThanks for partnering with us. Share your top blocker and I will send a plan."
    draft_witty = "Hi Alex,\nNo confetti cannon, just a quick check-in. Share one issue and I will help."
    draft_concise = "Hi Alex,\nQuick check-in. Reply with one issue and I will send next steps."

    def stub_serious(prompt, context=None):  # noqa: ANN001
        call_order.append("serious")
        call_counts["serious"] += 1
        assert prompt == "Write a retention outreach email"
        assert context == {
            "tenant": "acme",
            "company_name": "Example Co",
            "from_name": "Customer Success",
            "from_email": "success@example.com",
        }
        return draft_serious

    def stub_witty(prompt, context=None):  # noqa: ANN001
        call_order.append("witty")
        call_counts["witty"] += 1
        assert prompt == "Write a retention outreach email"
        assert context["tenant"] == "acme"
        return draft_witty

    def stub_concise(prompt, context=None):  # noqa: ANN001
        call_order.append("concise")
        call_counts["concise"] += 1
        assert prompt == "Write a retention outreach email"
        assert context["tenant"] == "acme"
        return draft_concise

    def stub_picker(drafts, context=None):  # noqa: ANN001
        call_order.append("picker")
        call_counts["picker"] += 1
        assert drafts == [draft_serious, draft_witty, draft_concise]
        assert context["company_name"] == "Example Co"
        return draft_witty

    def stub_emailer_agent(
        body_text,
        recipients,
        context=None,
        from_name=None,
        from_email=None,
        metadata=None,
    ):  # noqa: ANN001
        call_order.append("handoff")
        call_counts["handoff"] += 1
        assert body_text == draft_witty
        assert recipients == ["a@example.com", "b@example.com"]
        assert context == {
            "tenant": "acme",
            "company_name": "Example Co",
            "from_name": "Customer Success",
            "from_email": "success@example.com",
        }
        assert from_name == "Customer Success"
        assert from_email == "success@example.com"
        assert metadata == {"campaign": "q1"}
        return {"status": "sent", "provider": "stub"}

    monkeypatch.setattr(sales_manager_module, "write_retention_email_serious", stub_serious)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_witty", stub_witty)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_concise", stub_concise)
    monkeypatch.setattr(sales_manager_module, "pick_best_sales_email", stub_picker)
    monkeypatch.setattr(sales_manager_module, "emailer_agent", stub_emailer_agent)

    result = sales_manager_module.sales_manager(
        message_prompt="Write a retention outreach email",
        recipients=recipients,
        context={"tenant": "acme"},
        company_name="Example Co",
        from_name="Customer Success",
        from_email="success@example.com",
        metadata={"campaign": "q1"},
    )

    assert call_counts["serious"] == 1
    assert call_counts["witty"] == 1
    assert call_counts["concise"] == 1
    assert call_counts["picker"] == 1
    assert call_counts["handoff"] == 1
    assert call_order == ["serious", "witty", "concise", "picker", "handoff"]

    assert result["status"] == "sent"
    assert result["selected_draft"] == draft_witty
    assert result["recipients"] == recipients
    assert result["drafts"]["serious"] == draft_serious
    assert result["drafts"]["witty"] == draft_witty
    assert result["drafts"]["concise"] == draft_concise
    assert result["handoff_result"]["provider"] == "stub"
    assert result["errors"] == []


def test_sales_manager_passes_plain_text_selected_draft_to_handoff(monkeypatch):
    seen: dict[str, object] = {}

    def stub_serious(prompt, context=None):  # noqa: ANN001, ARG001
        return "Subject: Ignore this\n<p>Serious body line.</p>"

    def stub_witty(prompt, context=None):  # noqa: ANN001, ARG001
        return "Subject: Ignore this too\n<p>Witty body line.</p>"

    def stub_concise(prompt, context=None):  # noqa: ANN001, ARG001
        return "Subject: Ignore this one too\n<p>Concise body line.</p>"

    def stub_picker(drafts, context=None):  # noqa: ANN001, ARG001
        seen["drafts"] = drafts
        return "Subject: Ignore this too\n<p>Witty body line.</p>"

    def stub_emailer_agent(
        body_text,
        recipients,
        context=None,
        from_name=None,
        from_email=None,
        metadata=None,
    ):  # noqa: ANN001, ARG001
        seen["body_text"] = body_text
        seen["recipients"] = recipients
        return {"status": "sent"}

    monkeypatch.setattr(sales_manager_module, "write_retention_email_serious", stub_serious)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_witty", stub_witty)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_concise", stub_concise)
    monkeypatch.setattr(sales_manager_module, "pick_best_sales_email", stub_picker)
    monkeypatch.setattr(sales_manager_module, "emailer_agent", stub_emailer_agent)

    result = sales_manager_module.sales_manager(
        message_prompt="Write a retention outreach email",
        recipients=["a@example.com"],
    )

    assert seen["drafts"] == ["Serious body line.", "Witty body line.", "Concise body line."]
    assert seen["body_text"] == "Witty body line."
    assert seen["recipients"] == ["a@example.com"]
    assert result["status"] == "sent"
    assert result["selected_draft"] == "Witty body line."
