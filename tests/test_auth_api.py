import time

import jwt
from fastapi.testclient import TestClient

from api_demo import app


def _configure_hs256(monkeypatch, secret: str = "test-secret-that-is-long-enough!") -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("JWT_ISSUER", "aria")
    monkeypatch.setenv("JWT_AUDIENCE", "agile-mcp-server")


def _token(secret: str = "test-secret-that-is-long-enough!") -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": "user-123",
            "username": "nico",
            "role": "doctor",
            "permissions": ["patient:read"],
            "iss": "aria",
            "aud": "agile-mcp-server",
            "iat": now,
            "exp": now + 3600,
        },
        secret,
        algorithm="HS256",
    )


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


def test_auth_decode_endpoint_accepts_authorization_header(monkeypatch):
    _configure_hs256(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/auth/decode",
        json={"token": None},
        headers={"Authorization": f"Bearer {_token()}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["context"]["subject"] == "user-123"
    assert payload["context"]["permissions"] == ["patient:read"]


def test_auth_authorize_tool_endpoint_accepts_authorization_header(monkeypatch):
    _configure_hs256(monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/auth/authorize-tool",
        json={"tool_name": "get_patient_status", "token": None},
        headers={"authorization": f"Bearer {_token()}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authorized"] is True
    assert payload["subject"] == "user-123"
