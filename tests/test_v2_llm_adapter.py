"""Provider adapter and parser fallback stay deterministic under test."""

import httpx
import pytest

from v2.llm import OpenAICompatibleGenerator
from v2.parser import ParseFailure, parse_query


@pytest.mark.asyncio
async def test_openai_compatible_adapter_posts_model_and_bearer():
    observed = {}

    def respond(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["authorization"] = request.headers.get("authorization")
        observed["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Synthetic summary"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAICompatibleGenerator(client, "https://provider.example/v1", "test-key", "model-x")
        assert await adapter.generate("system", "user") == "Synthetic summary"
    assert observed["path"] == "/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-key"
    assert '"model":"model-x"' in observed["body"]


@pytest.mark.asyncio
async def test_llm_parser_fallback_is_typed_and_rejects_unknown_intent():
    class Reply:
        def __init__(self, value):
            self.value = value

        async def generate(self, system: str, user: str) -> str:
            return self.value

    parsed = await parse_query("Show record", Reply('{"intent":"patient_status","patient_id":"101"}'))
    assert parsed.patient_id == "101"
    with pytest.raises(ParseFailure):
        await parse_query("Show record", Reply('{"intent":"delete_all","patient_id":"101"}'))

