from src.agents import tools_email


def test_send_email_text_tool_forwards_arguments(monkeypatch):
    monkeypatch.setenv("SENDGRID_VERIFIED_SENDER", "verified@example.com")
    calls = {}

    class StubClient:
        def send_text(self, subject, body_text, to_emails, from_email=None):
            calls["subject"] = subject
            calls["body_text"] = body_text
            calls["to_emails"] = to_emails
            calls["from_email"] = from_email
            return {"status_code": 202}

    monkeypatch.setattr(tools_email, "SendgridEmailClient", lambda: StubClient())

    result = tools_email.send_email_text(
        subject="S1",
        body_text="Body",
        to_emails=["a@example.com", "b@example.com"],
    )

    assert result["status_code"] == 202
    assert calls["subject"] == "S1"
    assert calls["body_text"] == "Body"
    assert calls["to_emails"] == ["a@example.com", "b@example.com"]
    assert calls["from_email"] is None


def test_send_email_html_tool_forwards_explicit_sender(monkeypatch):
    calls = {}

    class StubClient:
        def send_html(self, subject, body_html, to_emails, from_email=None):
            calls["subject"] = subject
            calls["body_html"] = body_html
            calls["to_emails"] = to_emails
            calls["from_email"] = from_email
            return {"status_code": 202}

    monkeypatch.setattr(tools_email, "SendgridEmailClient", lambda: StubClient())

    result = tools_email.send_email_html(
        subject="S2",
        body_html="<p>Body</p>",
        to_emails=["a@example.com"],
        from_email="sender@example.com",
    )

    assert result["status_code"] == 202
    assert calls["subject"] == "S2"
    assert calls["body_html"] == "<p>Body</p>"
    assert calls["to_emails"] == ["a@example.com"]
    assert calls["from_email"] == "sender@example.com"
