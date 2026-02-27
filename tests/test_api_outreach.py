import application


def valid_record(customer_id: str, email: str):
    return {
        "customer_id": customer_id,
        "email": email,
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
    }


def test_outreach_contract_validation_failure_short_circuits_pipeline(monkeypatch):
    calls = {"predict": 0, "send": 0}

    def stub_predict_batch_records(records, options):  # noqa: ANN001, ARG001
        calls["predict"] += 1
        return {}

    def stub_send_email_text(**kwargs):  # noqa: ANN003, ARG001
        calls["send"] += 1
        return {"status_code": 202, "ok": True}

    monkeypatch.setattr(application, "predict_batch_records", stub_predict_batch_records)
    monkeypatch.setattr(application, "send_email_text", stub_send_email_text)

    client = application.app.test_client()
    payload = {
        "contract_version": "v2",
        "records": [],
        "outreach_options": {
            "threshold": 2,
            "max_emails": 0,
            "dry_run": "yes",
            "tone": "playful",
        },
        "context": {},
    }

    response = client.post("/api/outreach", json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["contract_version"] == "v1"
    assert body["status"] == "error"
    assert body["selected"] == []
    assert body["send"] == {"attempted": False, "sent": 0, "results": []}
    assert body["summary"]["n_records"] == 0
    assert body["errors"]
    assert calls["predict"] == 0
    assert calls["send"] == 0


def test_outreach_dry_run_returns_drafts_and_never_sends(monkeypatch):
    predict_calls = {"count": 0}
    send_calls = {"count": 0}

    def stub_predict_batch_records(records, options):  # noqa: ANN001, ARG001
        predict_calls["count"] += 1
        return {
            "status": "success",
            "results": [
                {"index": 1, "id": "C002", "p_churn": 0.72},
                {"index": 0, "id": "C001", "p_churn": 0.91},
                {"index": 2, "id": "C003", "p_churn": 0.30},
            ],
            "errors": None,
            "summary": {"valid_records": 3, "invalid_records": 0},
        }

    def stub_witty_writer(prompt, context=None):  # noqa: ANN001, ARG001
        return "Quick check-in from your retention team."

    def stub_subject_tool(draft_text):  # noqa: ANN001
        return "We'd love to keep you"

    def stub_send_email_text(**kwargs):  # noqa: ANN003, ARG001
        send_calls["count"] += 1
        return {"status_code": 202, "ok": True}

    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(application, "predict_batch_records", stub_predict_batch_records)
    monkeypatch.setattr(application, "write_retention_email_witty", stub_witty_writer)
    monkeypatch.setattr(application, "outreach_subject_tool", stub_subject_tool)
    monkeypatch.setattr(application, "send_email_text", stub_send_email_text)

    client = application.app.test_client()
    payload = {
        "contract_version": "v1",
        "records": [
            valid_record("C001", "one@example.com"),
            valid_record("C002", "two@example.com"),
            valid_record("C003", "three@example.com"),
        ],
        "outreach_options": {
            "threshold": 0.65,
            "max_emails": 50,
            "dry_run": True,
            "tone": "witty",
        },
        "context": {
            "company_name": "Example Co",
            "from_name": "Retention Team",
            "from_email": "retention@example.com",
        },
    }

    response = client.post("/api/outreach", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["summary"]["n_records"] == 3
    assert body["summary"]["n_valid"] == 3
    assert body["summary"]["n_invalid"] == 0
    assert body["summary"]["n_selected"] == 2
    assert [item["id"] for item in body["selected"]] == ["C001", "C002"]
    assert all("draft" in item for item in body["selected"])
    assert all(set(item["draft"]) == {"subject", "body_text"} for item in body["selected"])
    assert body["send"] == {"attempted": False, "sent": 0, "results": []}
    assert body["errors"] == []
    assert predict_calls["count"] == 1
    assert send_calls["count"] == 0


def test_outreach_send_mode_caps_and_sends_exactly_max_emails(monkeypatch):
    predict_calls = {"count": 0}
    send_calls = {"count": 0}
    sent_recipients = []

    def stub_predict_batch_records(records, options):  # noqa: ANN001, ARG001
        predict_calls["count"] += 1
        return {
            "status": "success",
            "results": [
                {"index": 0, "id": "C001", "p_churn": 0.99},
                {"index": 1, "id": "C002", "p_churn": 0.87},
                {"index": 2, "id": "C003", "p_churn": 0.79},
            ],
            "errors": None,
            "summary": {"valid_records": 3, "invalid_records": 0},
        }

    def stub_serious_writer(prompt, context=None):  # noqa: ANN001, ARG001
        return "We value your business and want to help."

    def stub_subject_tool(draft_text):  # noqa: ANN001
        return "A quick support check-in"

    def stub_send_email_text(subject, body_text, to_emails, from_email=None):  # noqa: ANN001
        send_calls["count"] += 1
        sent_recipients.extend(to_emails)
        return {"status_code": 202, "ok": True, "subject": subject, "from_email": from_email}

    monkeypatch.setenv("SENDGRID_API_KEY", "sg.test-key")
    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(application, "predict_batch_records", stub_predict_batch_records)
    monkeypatch.setattr(application, "write_retention_email_serious", stub_serious_writer)
    monkeypatch.setattr(application, "outreach_subject_tool", stub_subject_tool)
    monkeypatch.setattr(application, "send_email_text", stub_send_email_text)

    client = application.app.test_client()
    payload = {
        "contract_version": "v1",
        "records": [
            valid_record("C001", "one@example.com"),
            valid_record("C002", "two@example.com"),
            valid_record("C003", "three@example.com"),
        ],
        "outreach_options": {
            "threshold": 0.7,
            "max_emails": 2,
            "dry_run": False,
            "tone": "serious",
        },
        "context": {
            "company_name": "Example Co",
            "from_name": "Retention Team",
            "from_email": "retention@example.com",
        },
    }

    response = client.post("/api/outreach", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["summary"]["n_selected"] == 2
    assert [item["id"] for item in body["selected"]] == ["C001", "C002"]
    assert body["send"]["attempted"] is True
    assert body["send"]["sent"] == 2
    assert len(body["send"]["results"]) == 2
    assert predict_calls["count"] == 1
    assert send_calls["count"] == 2
    assert sent_recipients == ["one@example.com", "two@example.com"]


def test_outreach_response_contract_shape_is_stable_across_modes(monkeypatch):
    def stub_predict_batch_records(records, options):  # noqa: ANN001, ARG001
        return {
            "status": "success",
            "results": [{"index": 0, "id": "C001", "p_churn": 0.91}],
            "errors": None,
            "summary": {"valid_records": 1, "invalid_records": 0},
        }

    def stub_writer(prompt, context=None):  # noqa: ANN001, ARG001
        return "We appreciate your business."

    def stub_subject_tool(draft_text):  # noqa: ANN001, ARG001
        return "Checking in"

    def stub_send_email_text(subject, body_text, to_emails, from_email=None):  # noqa: ANN001, ARG001
        return {"status_code": 202, "ok": True}

    monkeypatch.setenv("SENDGRID_API_KEY", "sg.test-key")
    monkeypatch.setattr(application, "artifacts_ready", lambda: True)
    monkeypatch.setattr(application, "predict_batch_records", stub_predict_batch_records)
    monkeypatch.setattr(application, "write_retention_email_concise", stub_writer)
    monkeypatch.setattr(application, "outreach_subject_tool", stub_subject_tool)
    monkeypatch.setattr(application, "send_email_text", stub_send_email_text)

    client = application.app.test_client()
    base_payload = {
        "contract_version": "v1",
        "records": [valid_record("C001", "one@example.com")],
        "outreach_options": {
            "threshold": 0.65,
            "max_emails": 1,
            "tone": "concise",
        },
        "context": {
            "company_name": "Example Co",
            "from_name": "Retention Team",
            "from_email": "retention@example.com",
        },
    }

    dry_run_payload = dict(base_payload)
    dry_run_payload["outreach_options"] = dict(base_payload["outreach_options"])
    dry_run_payload["outreach_options"]["dry_run"] = True

    send_payload = dict(base_payload)
    send_payload["outreach_options"] = dict(base_payload["outreach_options"])
    send_payload["outreach_options"]["dry_run"] = False

    dry_run_response = client.post("/api/outreach", json=dry_run_payload)
    send_response = client.post("/api/outreach", json=send_payload)

    assert dry_run_response.status_code == 200
    assert send_response.status_code == 200

    dry_body = dry_run_response.get_json()
    send_body = send_response.get_json()

    expected_top_keys = {"contract_version", "status", "summary", "selected", "send", "errors", "timestamp"}
    expected_summary_keys = {"n_records", "n_valid", "n_invalid", "n_selected", "threshold", "max_emails", "dry_run"}
    expected_send_keys = {"attempted", "sent", "results"}
    expected_selected_keys = {"id", "index", "email", "p_churn", "draft"}
    expected_draft_keys = {"subject", "body_text"}

    assert set(dry_body) == expected_top_keys
    assert set(send_body) == expected_top_keys
    assert set(dry_body["summary"]) == expected_summary_keys
    assert set(send_body["summary"]) == expected_summary_keys
    assert set(dry_body["send"]) == expected_send_keys
    assert set(send_body["send"]) == expected_send_keys
    assert set(dry_body["selected"][0]) == expected_selected_keys
    assert set(send_body["selected"][0]) == expected_selected_keys
    assert set(dry_body["selected"][0]["draft"]) == expected_draft_keys
    assert set(send_body["selected"][0]["draft"]) == expected_draft_keys
