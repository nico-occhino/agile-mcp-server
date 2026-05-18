import time

import jwt

from server import authorize_tool_access, current_jwt_auth_context, decode_jwt_auth_context


def _configure_hs256(monkeypatch, secret: str = "test-secret-that-is-long-enough!") -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("JWT_ISSUER", "aria")
    monkeypatch.setenv("JWT_AUDIENCE", "agile-mcp-server")


def _token(
    secret: str = "test-secret-that-is-long-enough!",
    permissions: list[str] | None = None,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": "user-123",
            "username": "nico",
            "role": "doctor",
            "permissions": permissions or ["patient:read"],
            "iss": "aria",
            "aud": "agile-mcp-server",
            "iat": now,
            "exp": now + 3600,
        },
        secret,
        algorithm="HS256",
    )


def test_decode_demo_tool_returns_anonymous_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    result = decode_jwt_auth_context()

    assert result["auth_enabled"] is False
    assert result["context"]["subject"] == "anonymous"


def test_decode_demo_tool_decodes_provided_token(monkeypatch):
    _configure_hs256(monkeypatch)

    result = decode_jwt_auth_context(_token())

    assert result["context"]["subject"] == "user-123"
    assert result["context"]["permissions"] == ["patient:read"]


def test_authorize_tool_access_returns_structured_denial(monkeypatch):
    _configure_hs256(monkeypatch)

    result = authorize_tool_access(
        "get_patient_discharge_draft",
        token=_token(permissions=["patient:read"]),
    )

    assert result["authorized"] is False
    assert result["error"] == "Unauthorized: missing required permissions."
    assert result["required_permissions"] == ["patient:read", "discharge:generate"]


def test_authorize_tool_access_allows_guardrail_without_token_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    result = authorize_tool_access("evaluate_input_prompt_guardrail")

    assert result["authorized"] is True
    assert result["subject"] == "anonymous"


def test_current_jwt_auth_context_returns_anonymous_without_http_request_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    result = current_jwt_auth_context()

    assert result["auth_enabled"] is False
    assert result["context"]["subject"] == "anonymous"
