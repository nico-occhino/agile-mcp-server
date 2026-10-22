"""JWT verification and authorization shared by MCP and synthetic REST API."""

from __future__ import annotations

from dataclasses import dataclass

import jwt

from v2.settings import Settings


class AccessDenied(Exception):
    """Safe authentication or authorization failure."""


@dataclass(frozen=True)
class Caller:
    subject: str
    permissions: frozenset[str]
    token: str

    def require(self, permission: str) -> None:
        if permission not in self.permissions:
            raise AccessDenied("Permission denied")


def bearer_token(authorization: str | None) -> str:
    parts = (authorization or "").split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise AccessDenied("Bearer token required")
    return parts[1]


def verify_token(token: str, settings: Settings) -> Caller:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["sub", "exp", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise AccessDenied("Invalid bearer token") from exc
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AccessDenied("Invalid bearer token")
    raw = claims.get("permissions", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise AccessDenied("Invalid permissions claim")
    return Caller(subject, frozenset(raw), token)


def authorize(authorization: str | None, settings: Settings, permission: str) -> Caller:
    caller = verify_token(bearer_token(authorization), settings)
    caller.require(permission)
    return caller

