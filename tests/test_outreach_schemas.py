import pytest

from src.schemas import DraftSet, OutreachRequest, OutreachResult, Target


def test_target_constructs_with_optional_fields():
    target = Target(
        id="cust-001",
        email="person@example.com",
        name="Alex",
        metadata={"segment": "high-risk"},
    )

    assert target.id == "cust-001"
    assert target.email == "person@example.com"
    assert target.name == "Alex"
    assert target.metadata == {"segment": "high-risk"}


def test_target_rejects_invalid_email():
    with pytest.raises(ValueError, match="email"):
        Target(id="cust-001", email="not-an-email")


def test_draft_set_constructs():
    drafts = DraftSet(
        serious="Serious draft text.",
        witty="Witty draft text.",
        concise="Concise draft text.",
    )

    assert drafts.serious
    assert drafts.witty
    assert drafts.concise


def test_outreach_request_constructs():
    request = OutreachRequest(
        message_prompt="Write a retention outreach email.",
        recipients=[Target(id="cust-001", email="person@example.com")],
        from_name="Customer Success",
        from_email="success@example.com",
        company_name="Example Co",
        tone_policy="friendly-and-direct",
        send_mode="dry_run",
    )

    assert request.message_prompt.startswith("Write")
    assert len(request.recipients) == 1
    assert request.from_email == "success@example.com"


def test_outreach_request_requires_non_empty_recipients():
    with pytest.raises(ValueError, match="recipients"):
        OutreachRequest(
            message_prompt="Write a retention outreach email.",
            recipients=[],
            from_name="Customer Success",
            from_email="success@example.com",
            company_name="Example Co",
            tone_policy="friendly-and-direct",
            send_mode="dry_run",
        )


def test_outreach_result_constructs():
    result = OutreachResult(
        status="ok",
        selected_draft="serious",
        subject="We can help you get more value",
        html="<p>Hello</p>",
        send_status="queued",
        errors=[],
    )

    assert result.status == "ok"
    assert result.selected_draft == "serious"
    assert result.send_status == "queued"


def test_outreach_result_rejects_invalid_selected_draft():
    with pytest.raises(ValueError, match="selected_draft"):
        OutreachResult(
            status="ok",
            selected_draft="playful",
            subject="Subject",
            html="<p>Hello</p>",
        )
