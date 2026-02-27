from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.agents import retention_writers


class _StubResult:
    def __init__(self, final_output: str) -> None:
        self.final_output = final_output


class _StubRunner:
    _OUTPUTS = {
        "Serious Retention Writer": (
            "Subject: Retention check-in\n"
            "<p>Hi there, I wanted to check in and make sure your team is getting full value. "
            "If you share your top blocker, I can send a concrete plan by Friday.</p>"
        ),
        "Witty Retention Writer": (
            "Subject: Quick reality check\n"
            "<p>Hi there, no confetti cannon here, just a quick check-in. "
            "If one thing is making your week harder, tell me and I will fix what I can.</p>"
        ),
        "Concise Retention Writer": (
            "Subject: Fast follow-up\n"
            "<p>Hi there, quick check-in. Reply with your top issue and I will send next steps today.</p>"
        ),
    }

    @staticmethod
    def run_sync(agent, input_text: str):
        assert "Write" in input_text
        return _StubResult(_StubRunner._OUTPUTS[agent.name])


def test_retention_writer_tools_construct_with_stable_names():
    serious_tool = retention_writers.SeriousRetentionWriter().as_tool()
    witty_tool = retention_writers.WittyRetentionWriter().as_tool()
    concise_tool = retention_writers.ConciseRetentionWriter().as_tool()

    assert callable(serious_tool)
    assert callable(witty_tool)
    assert callable(concise_tool)
    assert serious_tool.__name__ == "write_retention_email_serious"
    assert witty_tool.__name__ == "write_retention_email_witty"
    assert concise_tool.__name__ == "write_retention_email_concise"


def test_retention_writer_tools_return_plain_text(monkeypatch):
    monkeypatch.setattr(retention_writers, "Runner", _StubRunner)
    tools = [
        retention_writers.write_retention_email_serious,
        retention_writers.write_retention_email_witty,
        retention_writers.write_retention_email_concise,
    ]

    outputs = [
        tool(
            prompt="Write a retention email for at-risk customers",
            company_name="Example Co",
            from_name="Customer Success",
            from_email="success@example.com",
        )
        for tool in tools
    ]

    for body in outputs:
        assert isinstance(body, str)
        assert body.strip()
        assert "<" not in body and ">" not in body

    assert len(set(outputs)) == 3


def test_retention_writer_tools_share_exact_description():
    tools = [
        retention_writers.write_retention_email_serious,
        retention_writers.write_retention_email_witty,
        retention_writers.write_retention_email_concise,
    ]

    descriptions = {getattr(tool, "tool_description", None) for tool in tools}

    assert descriptions == {retention_writers.TOOL_DESCRIPTION}
    assert retention_writers.TOOL_DESCRIPTION == "Write a cold sales/retention email"


def test_retention_writer_tools_can_run_in_parallel(monkeypatch):
    monkeypatch.setattr(retention_writers, "Runner", _StubRunner)
    tools = [
        retention_writers.write_retention_email_serious,
        retention_writers.write_retention_email_witty,
        retention_writers.write_retention_email_concise,
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                tool,
                prompt="Write a retention email for at-risk customers",
                context={"company_name": "Example Co", "from_name": "Customer Success"},
            )
            for tool in tools
        ]
        outputs = [future.result() for future in futures]

    assert len(outputs) == 3
    assert all(isinstance(body, str) and body.strip() for body in outputs)
