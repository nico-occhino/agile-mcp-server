import time

import jwt
import pytest

from auth.context import anonymous_context
from auth.jwt import MissingTokenError
from auth.request_auth import (
    auth_context_from_bearer_header,
    extract_bearer_token_from_headers,
)


def _configure_hs256(monkeypatch, secret: str = "test-secret-that-is-long-enough!") -> None:
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


def test_authorization_bearer_token_is_extracted():
    token = extract_bearer_token_from_headers({"Authorization": "Bearer abc.def.ghi"})

    assert token == "abc.def.ghi"


def test_lowercase_authorization_header_works():
    token = extract_bearer_token_from_headers({"authorization": "Bearer abc.def.ghi"})

    assert token == "abc.def.ghi"


def test_extra_spaces_work():
    token = extract_bearer_token_from_headers({"Authorization": "  Bearer   abc.def.ghi  "})

    assert token == "abc.def.ghi"


def test_missing_authorization_returns_none():
    assert extract_bearer_token_from_headers({}) is None


def test_basic_auth_returns_none():
    token = extract_bearer_token_from_headers({"Authorization": "Basic abc"})

    assert token is None


def test_bearer_without_token_returns_none():
    assert extract_bearer_token_from_headers({"Authorization": "Bearer"}) is None
    assert extract_bearer_token_from_headers({"Authorization": "Bearer   "}) is None


def test_auth_disabled_missing_header_returns_anonymous(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    context = auth_context_from_bearer_header(None)

    assert context == anonymous_context()


def test_auth_enabled_missing_header_raises(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")

    with pytest.raises(MissingTokenError):
        auth_context_from_bearer_header(None)


def test_valid_hs256_token_in_header_returns_auth_context(monkeypatch):
    _configure_hs256(monkeypatch)

    context = auth_context_from_bearer_header(
        {"Authorization": f"Bearer {_token()}"},
    )

    assert context.subject == "user-123"
    assert context.username == "nico"
    assert context.permissions == ["patient:read"]
