from __future__ import annotations

import importlib

from src.pipeline import run_outreach_from_batch


def _stub_sales_result(send_mode: str) -> dict:
    return {
        "status": "dry_run" if send_mode == "dry_run" else "sent",
        "send_mode": send_mode,
        "selected_draft": "Serious draft body.",
        "drafts": {
            "serious": "Serious draft body.",
            "witty": "Witty draft body.",
            "concise": "Concise draft body.",
        },
        "handoff_result": {
            "status": "dry_run" if send_mode == "dry_run" else "sent",
            "subject": "Retention check-in",
            "html": "<p>Serious draft body.</p>",
            "send_status": "skipped" if send_mode == "dry_run" else "sent",
        },
        "errors": [],
    }


def test_outreach_pipeline_dry_run_selects_expected_targets():
    captured: dict[str, object] = {}

    def stub_sales_manager(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _stub_sales_result(str(kwargs["send_mode"]))

    batch_response = {
        "status": "success",
        "results": [
            {"id": "cust-c", "email": "c@example.com", "p_churn": 0.95},
            {"id": "cust-a", "email": "a@example.com", "p_churn": 0.90},
            {"id": "cust-b", "email": "b@example.com", "p_churn": 0.30},
            {"id": "cust-d", "email": "d@example.com", "p_churn": None},
            {"id": "cust-e", "p_churn": 0.99},
        ],
        "errors": None,
    }
    config = {
        "company_name": "Example Co",
        "from_name": "Customer Success",
        "from_email": "success@example.com",
        "threshold": 0.8,
        "max_targets": 10,
        "dry_run": True,
        "sales_manager_runner": stub_sales_manager,
    }

    report = run_outreach_from_batch(batch_response, config)

    assert report["status"] == "ok"
    assert len(report["selected_targets"]) == 2
    assert [item["id"] for item in report["selected_targets"]] == ["cust-a", "cust-c"]
    assert captured["recipients"] == ["a@example.com", "c@example.com"]


def test_outreach_pipeline_passes_recipients_identity_and_send_mode():
    captured: dict[str, object] = {}

    def stub_sales_manager(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _stub_sales_result(str(kwargs["send_mode"]))

    batch_response = {
        "status": "success",
        "results": [
            {"id": "cust-2", "email": "two@example.com", "p_churn": 0.88},
            {"id": "cust-1", "email": "one@example.com", "p_churn": 0.99},
        ],
        "errors": None,
    }
    config = {
        "company_name": "Northwind",
        "from_name": "Retention Team",
        "from_email": "retention@northwind.com",
        "prompt_template": "Write outreach from {from_name} at {company_name} for {recipient_ids}.",
        "threshold": 0.8,
        "dry_run": True,
        "sales_manager_runner": stub_sales_manager,
    }

    report = run_outreach_from_batch(batch_response, config)

    assert report["status"] == "ok"
    assert captured["company_name"] == "Northwind"
    assert captured["from_name"] == "Retention Team"
    assert captured["from_email"] == "retention@northwind.com"
    assert captured["send_mode"] == "dry_run"
    assert captured["recipients"] == ["one@example.com", "two@example.com"]

    outreach_request = report["outreach_request"]
    assert outreach_request is not None
    assert outreach_request["send_mode"] == "dry_run"
    assert [item["id"] for item in outreach_request["recipients"]] == ["cust-1", "cust-2"]
    assert outreach_request["company_name"] == "Northwind"


def test_outreach_pipeline_dry_run_never_calls_sendgrid(monkeypatch):
    sales_manager_module = importlib.import_module("src.agents.sales_manager")
    email_manager_module = importlib.import_module("src.agents.email_manager")
    send_calls = {"count": 0}

    def stub_serious(prompt, context=None):  # noqa: ANN001, ARG001
        return "Serious draft body."

    def stub_witty(prompt, context=None):  # noqa: ANN001, ARG001
        return "Witty draft body."

    def stub_concise(prompt, context=None):  # noqa: ANN001, ARG001
        return "Concise draft body."

    def stub_picker(drafts, context=None):  # noqa: ANN001, ARG001
        return drafts[0]

    def stub_subject_tool(body_text, recipients, context=None):  # noqa: ANN001, ARG001
        return "Retention check-in"

    def stub_html_tool(subject, body_text, recipients, context=None):  # noqa: ANN001, ARG001
        return "<p>Serious draft body.</p>"

    def stub_send_email_html(subject, html, recipients, from_name=None, from_email=None, metadata=None):  # noqa: ANN001, ARG001
        send_calls["count"] += 1
        return {"status_code": 202, "ok": True}

    monkeypatch.setattr(sales_manager_module, "write_retention_email_serious", stub_serious)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_witty", stub_witty)
    monkeypatch.setattr(sales_manager_module, "write_retention_email_concise", stub_concise)
    monkeypatch.setattr(sales_manager_module, "pick_best_sales_email", stub_picker)
    monkeypatch.setattr(sales_manager_module, "email_subject_tool", stub_subject_tool)
    monkeypatch.setattr(sales_manager_module, "email_html_tool", stub_html_tool)
    monkeypatch.setattr(email_manager_module, "send_email_html", stub_send_email_html)

    batch_response = {
        "status": "success",
        "results": [
            {"id": "cust-1", "email": "one@example.com", "p_churn": 0.95},
        ],
        "errors": None,
    }
    config = {
        "company_name": "Example Co",
        "from_name": "Customer Success",
        "from_email": "success@example.com",
        "threshold": 0.8,
        "dry_run": True,
    }

    report = run_outreach_from_batch(batch_response, config)

    assert report["status"] == "ok"
    assert send_calls["count"] == 0
    assert report["outreach_result"] is not None
    assert report["outreach_result"]["send_status"] == "skipped"


def test_outreach_pipeline_partial_batch_still_returns_targets_and_carries_batch_errors():
    captured: dict[str, object] = {}

    def stub_sales_manager(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _stub_sales_result(str(kwargs["send_mode"]))

    batch_errors = [
        {"row_index": 3, "field": "Age", "message": "Missing required field: Age"},
    ]
    batch_response = {
        "status": "partial",
        "results": [
            {"id": "cust-1", "email": "one@example.com", "p_churn": 0.91},
            {"id": "cust-2", "email": "two@example.com", "p_churn": 0.10},
        ],
        "errors": batch_errors,
    }
    config = {
        "company_name": "Example Co",
        "from_name": "Customer Success",
        "from_email": "success@example.com",
        "threshold": 0.8,
        "dry_run": True,
        "sales_manager_runner": stub_sales_manager,
    }

    report = run_outreach_from_batch(batch_response, config)

    assert report["status"] == "partial"
    assert len(report["selected_targets"]) == 1
    assert captured["recipients"] == ["one@example.com"]
    assert any(item.get("stage") == "batch" for item in report["errors"] if isinstance(item, dict))
