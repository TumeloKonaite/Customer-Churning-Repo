from src.agents import email_manager


def test_email_manager_runs_subject_html_send_once_in_order(monkeypatch):
    call_order: list[str] = []
    call_counts = {"subject": 0, "html": 0, "send": 0}
    expected_recipients = ["a@example.com", "b@example.com"]

    def stub_subject_tool(body_text, recipients, context=None):
        call_order.append("subject_tool")
        call_counts["subject"] += 1
        assert body_text == "Draft body text"
        assert recipients == expected_recipients
        assert context == {"tenant": "acme"}
        return "Retention check-in"

    def stub_html_tool(subject, body_text, recipients, context=None):
        call_order.append("html_tool")
        call_counts["html"] += 1
        assert subject == "Retention check-in"
        assert body_text == "Draft body text"
        assert recipients == expected_recipients
        assert context == {"tenant": "acme"}
        return "<p>Hi team</p>"

    def stub_send_email_html(subject, html, recipients, from_name=None, from_email=None, metadata=None):
        call_order.append("send_email_html")
        call_counts["send"] += 1
        assert subject == "Retention check-in"
        assert html == "<p>Hi team</p>"
        assert recipients == expected_recipients
        assert from_name == "Customer Success"
        assert from_email == "success@example.com"
        assert metadata == {"campaign": "q1"}
        return {"status_code": 202, "ok": True, "provider": "stub"}

    monkeypatch.setattr(email_manager, "subject_tool", stub_subject_tool)
    monkeypatch.setattr(email_manager, "html_tool", stub_html_tool)
    monkeypatch.setattr(email_manager, "send_email_html", stub_send_email_html)

    result = email_manager.emailer_agent(
        body_text="Draft body text",
        recipients=expected_recipients,
        context={"tenant": "acme"},
        from_name="Customer Success",
        from_email="success@example.com",
        metadata={"campaign": "q1"},
    )

    assert call_counts["subject"] == 1
    assert call_counts["html"] == 1
    assert call_counts["send"] == 1
    assert call_order == ["subject_tool", "html_tool", "send_email_html"]
    assert result["status"] == "sent"
    assert result["subject"] == "Retention check-in"
    assert result["html"] == "<p>Hi team</p>"
    assert result["recipients"] == expected_recipients
    assert result["send_status"] == "sent"
    assert result["send_result"]["provider"] == "stub"
    assert result["errors"] == []


def test_email_manager_returns_error_without_sending_on_pre_send_failure(monkeypatch):
    call_counts = {"subject": 0, "html": 0, "send": 0}

    def stub_subject_tool(body_text, recipients, context=None):  # noqa: ARG001
        call_counts["subject"] += 1
        raise RuntimeError("subject generation failed")

    def stub_html_tool(subject, body_text, recipients, context=None):  # noqa: ARG001
        call_counts["html"] += 1
        return "<p>unused</p>"

    def stub_send_email_html(subject, html, recipients, from_name=None, from_email=None, metadata=None):  # noqa: ARG001
        call_counts["send"] += 1
        return {"status_code": 202, "ok": True}

    monkeypatch.setattr(email_manager, "subject_tool", stub_subject_tool)
    monkeypatch.setattr(email_manager, "html_tool", stub_html_tool)
    monkeypatch.setattr(email_manager, "send_email_html", stub_send_email_html)

    result = email_manager.emailer_agent(
        body_text="Draft body text",
        recipients=["a@example.com"],
    )

    assert call_counts["subject"] == 1
    assert call_counts["html"] == 0
    assert call_counts["send"] == 0
    assert result["status"] == "error"
    assert result["subject"] == ""
    assert result["html"] == ""
    assert result["send_status"] is None
    assert result["send_result"] is None
    assert result["errors"] == ["subject generation failed"]


def test_email_manager_handoff_description():
    assert email_manager.handoff_description == "Convert an email to HTML and send it"
