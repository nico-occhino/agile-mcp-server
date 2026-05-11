"""JWT verification and AuthContext extraction for Phase 2 auth."""

from __future__ import annotations

import os
from typing import Any

import jwt
from jwt import PyJWTError

from auth.context import AuthContext, anonymous_context


class AuthError(Exception):
    """Base class for authentication failures."""


class MissingTokenError(AuthError):
    """Raised when auth is enabled and no token was supplied."""


class InvalidTokenError(AuthError):
    """Raised when a token cannot be safely verified."""


def is_auth_enabled() -> bool:
    """Return whether JWT enforcement is enabled by environment."""
    return os.getenv("AUTH_ENABLED", "false").lower() in {"true", "1", "yes", "on"}


def decode_and_verify_jwt(token: str) -> AuthContext:
    """Verify a JWT and map its claims into an AuthContext."""
    if not token or not token.strip():
        raise MissingTokenError("Missing authentication token.")

    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    key = _verification_key_for_algorithm(algorithm)
    options: dict[str, Any] = {}
    issuer = os.getenv("JWT_ISSUER") or None
    audience = os.getenv("JWT_AUDIENCE", "agile-mcp-server") or None

    try:
        claims = jwt.decode(
            token.strip(),
            key=key,
            algorithms=[algorithm],
            issuer=issuer,
            audience=audience,
            options=options,
        )
    except PyJWTError as exc:
        raise InvalidTokenError(_safe_invalid_token_message(exc)) from exc

    return AuthContext(
        subject=str(claims.get("sub", "")),
        username=claims.get("username") or claims.get("preferred_username"),
        role=claims.get("role"),
        department=claims.get("department"),
        permissions=_extract_permissions(claims),
        issuer=claims.get("iss"),
        audience=claims.get("aud"),
        issued_at=claims.get("iat"),
        expires_at=claims.get("exp"),
        raw_claims=claims,
    )


def auth_context_from_optional_token(token: str | None) -> AuthContext:
    """Return anonymous context when auth is disabled, otherwise verify token."""
    if not is_auth_enabled():
        return anonymous_context()
    if not token:
        raise MissingTokenError("Missing authentication token.")
    return decode_and_verify_jwt(token)


def _verification_key_for_algorithm(algorithm: str) -> str:
    if algorithm == "HS256":
        secret = os.getenv("JWT_SECRET", "")
        if not secret:
            raise InvalidTokenError("JWT secret is not configured.")
        return secret
    if algorithm == "RS256":
        public_key = os.getenv("JWT_PUBLIC_KEY", "")
        if not public_key:
            raise InvalidTokenError("JWT public key is not configured.")
        return public_key
    raise InvalidTokenError("Unsupported JWT algorithm.")


def _extract_permissions(claims: dict[str, Any]) -> list[str]:
    values: list[str] = []

    permissions = claims.get("permissions")
    if isinstance(permissions, list):
        values.extend(str(permission) for permission in permissions)

    scope = claims.get("scope")
    if isinstance(scope, str):
        values.extend(scope.split())

    return list(dict.fromkeys(permission for permission in values if permission))


def _safe_invalid_token_message(exc: Exception) -> str:
    message = exc.__class__.__name__.replace("Error", "")
    return f"Invalid token: {message}."
