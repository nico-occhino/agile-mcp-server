"""Instrumentation must never emit request data or exception messages."""

import pytest

from workflow import instrumentation


class CapturingLog:
    def __init__(self):
        self.events = []

    def info(self, event, **fields):
        self.events.append((event, fields))

    def error(self, event, **fields):
        self.events.append((event, fields))


def test_tool_arguments_and_exception_messages_are_not_logged(monkeypatch):
    capture = CapturingLog()
    monkeypatch.setattr(instrumentation, "log", capture)

    @instrumentation.instrumented("test_tool")
    def failing_tool(*, token):
        raise ValueError(f"Rejected {token}")

    with pytest.raises(ValueError):
        failing_tool(token="secret-bearer-value")

    assert [event for event, _ in capture.events] == [
        "tool_call_start",
        "tool_call_error",
    ]
    assert "secret-bearer-value" not in repr(capture.events)
    assert capture.events[-1][1]["error_type"] == "ValueError"

