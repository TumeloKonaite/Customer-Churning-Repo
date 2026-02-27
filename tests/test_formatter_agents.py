from src.agents import formatter_agents


def test_formatter_tools_exist_with_expected_names():
    assert callable(formatter_agents.subject_tool)
    assert callable(formatter_agents.html_tool)
    assert formatter_agents.subject_tool.__name__ == "subject_tool"
    assert formatter_agents.html_tool.__name__ == "html_tool"


def test_subject_tool_returns_subject_only_with_stub_runner(monkeypatch):
    class StubResult:
        def __init__(self, final_output):
            self.final_output = final_output

    class StubRunner:
        @staticmethod
        def run_sync(agent, input_text):
            assert agent is formatter_agents.subject_writer
            assert input_text == "Draft body text"
            return StubResult("Subject: Save at-risk accounts now\n<p>Ignored body</p>")

    monkeypatch.setattr(formatter_agents, "Runner", StubRunner)

    result = formatter_agents.subject_tool("Draft body text")

    assert result == "Save at-risk accounts now"


def test_html_tool_returns_html_only_with_stub_runner(monkeypatch):
    class StubResult:
        def __init__(self, final_output):
            self.final_output = final_output

    class StubRunner:
        @staticmethod
        def run_sync(agent, input_text):
            assert agent is formatter_agents.html_converter
            assert input_text == "Draft body text"
            return StubResult("Subject: Ignore this line\n<p>Hello team</p>")

    monkeypatch.setattr(formatter_agents, "Runner", StubRunner)

    result = formatter_agents.html_tool("Draft body text")

    assert result == "<p>Hello team</p>"
