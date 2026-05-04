from pydantic import BaseModel

from workflow import llm_client


class StructuredFixture(BaseModel):
    name: str
    items: list[str]


def test_call_llm_structured_retries_after_validation_error(monkeypatch):
    responses = iter([
        '{"properties": {"name": {"type": "string"}}}',
        '{"name": "valid", "items": ["one"]}',
    ])
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(llm_client, "call_llm", fake_call_llm)

    result = llm_client.call_llm_structured(
        system="Extract structured data.",
        user="Input text.",
        schema=StructuredFixture,
    )

    assert result == StructuredFixture(name="valid", items=["one"])
    assert len(calls) == 2
    assert "INSTANCE" in calls[0]["system"]
    assert "failed Pydantic validation" in calls[1]["user"]
