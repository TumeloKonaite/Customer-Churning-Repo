import json

import application


def test_outreach_form_get_renders():
    client = application.app.test_client()
    response = client.get("/outreach")

    assert response.status_code == 200
    assert b"Outreach Pipeline" in response.data
    assert b"outreach_request_json" in response.data


def test_outreach_form_post_invalid_json_shows_error():
    client = application.app.test_client()
    response = client.post("/outreach", data={"outreach_request_json": "{"})

    assert response.status_code == 200
    assert b"Request Error" in response.data
    assert b"Invalid request JSON" in response.data


def test_outreach_form_post_valid_json_executes_request(monkeypatch):
    calls = {}

    def stub_execute_outreach_request(body):  # noqa: ANN001
        calls["body"] = body
        return (
            {
                "contract_version": "v1",
                "status": "ok",
                "summary": {
                    "n_records": 1,
                    "n_valid": 1,
                    "n_invalid": 0,
                    "n_selected": 1,
                    "threshold": 0.65,
                    "max_emails": 1,
                    "dry_run": True,
                },
                "selected": [
                    {
                        "id": "C001",
                        "index": 0,
                        "email": "one@example.com",
                        "p_churn": 0.9,
                        "draft": {"subject": "Checking in", "body_text": "Hello from support."},
                    }
                ],
                "send": {"attempted": False, "sent": 0, "results": []},
                "errors": [],
                "timestamp": "2026-02-27T00:00:00+00:00",
            },
            200,
        )

    monkeypatch.setattr(application, "execute_outreach_request", stub_execute_outreach_request)

    payload = {
        "contract_version": "v1",
        "records": [{}],
        "outreach_options": {},
        "context": {},
    }

    client = application.app.test_client()
    response = client.post("/outreach", data={"outreach_request_json": json.dumps(payload)})

    assert response.status_code == 200
    assert calls["body"] == payload
    assert b"Outreach Response" in response.data
    assert b"HTTP 200" in response.data
