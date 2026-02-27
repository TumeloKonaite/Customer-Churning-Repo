from src.services.outreach_service import build_outreach_payload, select_targets


def test_select_targets_filters_by_threshold():
    batch_results = [
        {"id": "cust-1", "email": "a@example.com", "p_churn": 0.91},
        {"id": "cust-2", "email": "b@example.com", "p_churn": 0.49},
        {"id": "cust-3", "email": "c@example.com", "p_churn": None},
        {"id": "cust-4", "email": "d@example.com", "p_churn": 0.50},
    ]

    targets = select_targets(batch_results, threshold=0.5, max_n=10)

    assert [target["id"] for target in targets] == ["cust-1", "cust-4"]
    assert all(target["metadata"]["p_churn"] >= 0.5 for target in targets)


def test_select_targets_missing_email_handling():
    batch_results = [
        {"id": "cust-1", "email": "valid@example.com", "p_churn": 0.95},
        {"id": "cust-2", "email": " ", "p_churn": 0.90},
        {"id": "cust-3", "p_churn": 0.85},
        {"id": "cust-4", "email": "not-an-email", "p_churn": 0.80},
    ]

    require_email = select_targets(batch_results, threshold=0.7, max_n=10, require_email=True)
    allow_missing_email = select_targets(batch_results, threshold=0.7, max_n=10, require_email=False)

    assert [target["id"] for target in require_email] == ["cust-1"]
    assert [target["id"] for target in allow_missing_email] == ["cust-1", "cust-2", "cust-3", "cust-4"]
    assert [target["email"] for target in allow_missing_email] == ["valid@example.com", None, None, None]


def test_select_targets_enforces_max_n():
    batch_results = [
        {"id": "cust-1", "email": "one@example.com", "p_churn": 0.99},
        {"id": "cust-2", "email": "two@example.com", "p_churn": 0.88},
        {"id": "cust-3", "email": "three@example.com", "p_churn": 0.77},
        {"id": "cust-4", "email": "four@example.com", "p_churn": 0.66},
    ]

    targets = select_targets(batch_results, threshold=0.5, max_n=2)

    assert len(targets) == 2
    assert [target["id"] for target in targets] == ["cust-1", "cust-2"]


def test_select_targets_stable_ordering_with_tie_breaker():
    batch_results = [
        {"id": "cust-2", "email": "two@example.com", "p_churn": 0.8},
        {"id": "cust-1", "email": "one@example.com", "p_churn": 0.8},
        {"index": 10, "email": "ten@example.com", "p_churn": 0.8},
        {"index": 2, "email": "twoidx@example.com", "p_churn": 0.8},
    ]

    first = select_targets(batch_results, threshold=0.7, max_n=10)
    second = select_targets(batch_results, threshold=0.7, max_n=10)

    assert first == second
    assert [target["id"] for target in first] == ["cust-1", "cust-2", "idx-2", "idx-10"]


def test_build_outreach_payload_is_deterministic_and_contract_shaped():
    targets = [
        {"id": "cust-1", "email": "one@example.com", "name": "Taylor"},
        {"id": "cust-2", "email": "two@example.com"},
    ]
    prompt_template = (
        "Write a retention email from {from_name} at {company_name} "
        "for {recipient_count} recipients: {recipient_ids}."
    )

    payload_one = build_outreach_payload(
        targets=targets,
        from_name="Customer Success",
        company_name="Example Co",
        prompt_template=prompt_template,
    )
    payload_two = build_outreach_payload(
        targets=targets,
        from_name="Customer Success",
        company_name="Example Co",
        prompt_template=prompt_template,
    )

    assert payload_one == payload_two
    assert payload_one["company_name"] == "Example Co"
    assert payload_one["from_name"] == "Customer Success"
    assert payload_one["message_prompt"] == (
        "Write a retention email from Customer Success at Example Co "
        "for 2 recipients: cust-1, cust-2."
    )
    assert [recipient["id"] for recipient in payload_one["recipients"]] == ["cust-1", "cust-2"]
    assert set(payload_one) == {
        "company_name",
        "from_name",
        "recipients",
        "message_prompt",
        "from_email",
        "tone_policy",
        "send_mode",
    }
