import time

import jwt
import pytest

from auth.context import anonymous_context
from auth.jwt import (
    InvalidTokenError,
    MissingTokenError,
    auth_context_from_optional_token,
    decode_and_verify_jwt,
)


def _configure_hs256(monkeypatch, secret: str = "test-secret-that-is-long-enough!") -> None:
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("JWT_ISSUER", "aria")
    monkeypatch.setenv("JWT_AUDIENCE", "agile-mcp-server")


def _token(secret: str = "test-secret-that-is-long-enough!", **overrides) -> str:
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "username": "nico",
        "role": "doctor",
        "department": "orthopedics",
        "permissions": ["patient:read"],
        "iss": "aria",
        "aud": "agile-mcp-server",
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return jwt.encode(claims, secret, algorithm="HS256")


def test_valid_hs256_token_decodes_into_auth_context(monkeypatch):
    _configure_hs256(monkeypatch)

    context = decode_and_verify_jwt(_token())

    assert context.subject == "user-123"
    assert context.username == "nico"
    assert context.role == "doctor"
    assert context.department == "orthopedics"
    assert context.permissions == ["patient:read"]
    assert context.issuer == "aria"
    assert context.audience == "agile-mcp-server"
    assert context.raw_claims["sub"] == "user-123"


def test_scope_string_is_split_into_permissions(monkeypatch):
    _configure_hs256(monkeypatch)

    context = decode_and_verify_jwt(_token(permissions=None, scope="patient:read cohort:read"))

    assert context.permissions == ["patient:read", "cohort:read"]


def test_permissions_list_works(monkeypatch):
    _configure_hs256(monkeypatch)

    context = decode_and_verify_jwt(
        _token(permissions=["patient:read", "summary:generate"])
    )

    assert context.permissions == ["patient:read", "summary:generate"]


def test_permissions_and_scope_combine_and_deduplicate(monkeypatch):
    _configure_hs256(monkeypatch)

    context = decode_and_verify_jwt(
        _token(permissions=["patient:read"], scope="patient:read cohort:read")
    )

    assert context.permissions == ["patient:read", "cohort:read"]


def test_expired_token_raises_invalid_token(monkeypatch):
    _configure_hs256(monkeypatch)

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt(_token(exp=int(time.time()) - 10))


def test_invalid_signature_raises_invalid_token(monkeypatch):
    _configure_hs256(monkeypatch, secret="expected-secret-that-is-long-enough")

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt(_token(secret="wrong-secret-that-is-long-enough"))


def test_missing_token_with_auth_enabled_raises(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")

    with pytest.raises(MissingTokenError):
        auth_context_from_optional_token(None)


def test_auth_disabled_missing_token_returns_anonymous(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    context = auth_context_from_optional_token(None)

    assert context == anonymous_context()
