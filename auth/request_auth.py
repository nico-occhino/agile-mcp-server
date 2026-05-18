"""HTTP header based auth helpers for FastMCP/SSE requests."""

from __future__ import annotations

from collections.abc import Mapping

from auth.context import AuthContext, anonymous_context
from auth.jwt import MissingTokenError, decode_and_verify_jwt, is_auth_enabled


def extract_bearer_token_from_headers(headers: Mapping[str, str] | None) -> str | None:
    """Extract a Bearer token from case-insensitive HTTP headers."""
    if not headers:
        return None

    authorization = None
    for name, value in headers.items():
        if name.lower() == "authorization":
            authorization = value
            break

    if not authorization:
        return None

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def auth_context_from_bearer_header(
    headers: Mapping[str, str] | None,
) -> AuthContext:
    """Build AuthContext from an Authorization Bearer header."""
    token = extract_bearer_token_from_headers(headers)
    if token:
        return decode_and_verify_jwt(token)
    if is_auth_enabled():
        raise MissingTokenError("Missing authentication token.")
    return anonymous_context()


def get_current_auth_context() -> AuthContext:
    """Read current FastMCP HTTP headers and return the caller AuthContext."""
    from fastmcp.server.dependencies import get_http_headers

    headers = get_http_headers(include={"authorization"})
    return auth_context_from_bearer_header(headers)
