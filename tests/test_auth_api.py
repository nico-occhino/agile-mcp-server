from fastapi.testclient import TestClient

from api_demo import app


def test_auth_decode_endpoint_returns_anonymous_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    client = TestClient(app)

    response = client.post("/auth/decode", json={"token": None})

    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["subject"] == "anonymous"


def test_auth_authorize_tool_endpoint_allows_guardrail_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    client = TestClient(app)

    response = client.post(
        "/auth/authorize-tool",
        json={"tool_name": "evaluate_input_prompt_guardrail", "token": None},
    )

    assert response.status_code == 200
    assert response.json()["authorized"] is True
