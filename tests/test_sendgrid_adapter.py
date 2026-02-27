from src.adapters.email_sendgrid import SendgridEmailClient


def test_send_text_builds_payload_with_multiple_recipients(monkeypatch):
    captured = {}

    def fake_send_payload(self, payload):
        captured["payload"] = payload
        return {"status_code": 202, "ok": True, "body": ""}

    monkeypatch.setenv("SENDGRID_API_KEY", "sg.test-key")
    monkeypatch.setenv("SENDGRID_VERIFIED_SENDER", "verified@example.com")
    monkeypatch.setattr(SendgridEmailClient, "_send_payload", fake_send_payload)

    client = SendgridEmailClient()
    response = client.send_text(
        subject="Retention Offer",
        body_text="Hello there",
        to_emails=["a@example.com", "b@example.com"],
    )

    assert response["status_code"] == 202
    payload = captured["payload"]
    assert payload["from"]["email"] == "verified@example.com"
    assert payload["subject"] == "Retention Offer"
    assert payload["content"][0]["type"] == "text/plain"
    assert payload["content"][0]["value"] == "Hello there"
    assert payload["personalizations"][0]["to"] == [
        {"email": "a@example.com"},
        {"email": "b@example.com"},
    ]


def test_send_html_uses_explicit_sender_over_default(monkeypatch):
    captured = {}

    def fake_send_payload(self, payload):
        captured["payload"] = payload
        return {"status_code": 202, "ok": True, "body": ""}

    monkeypatch.setenv("SENDGRID_API_KEY", "sg.test-key")
    monkeypatch.setenv("SENDGRID_VERIFIED_SENDER", "verified@example.com")
    monkeypatch.setattr(SendgridEmailClient, "_send_payload", fake_send_payload)

    client = SendgridEmailClient()
    client.send_html(
        subject="Retention Offer",
        body_html="<p>Hello there</p>",
        to_emails=["a@example.com"],
        from_email="agent@example.com",
    )

    payload = captured["payload"]
    assert payload["from"]["email"] == "agent@example.com"
    assert payload["content"][0]["type"] == "text/html"
    assert payload["content"][0]["value"] == "<p>Hello there</p>"
    assert payload["personalizations"][0]["to"] == [{"email": "a@example.com"}]


def test_send_text_requires_sender_when_no_env_default(monkeypatch):
    monkeypatch.setenv("SENDGRID_API_KEY", "sg.test-key")
    monkeypatch.delenv("SENDGRID_VERIFIED_SENDER", raising=False)
    client = SendgridEmailClient()

    try:
        client.send_text(
            subject="Retention Offer",
            body_text="Hello there",
            to_emails=["a@example.com"],
        )
    except ValueError as exc:
        assert "sender email" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing sender")
