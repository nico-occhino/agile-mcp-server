"""Typed authentication context extracted from JWT claims."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    subject: str
    username: str | None = None
    role: str | None = None
    department: str | None = None
    permissions: list[str] = Field(default_factory=list)
    issuer: str | None = None
    audience: str | list[str] | None = None
    issued_at: int | None = None
    expires_at: int | None = None
    raw_claims: dict[str, Any] = Field(default_factory=dict)


def anonymous_context() -> AuthContext:
    """Return an unauthenticated Phase 1 context."""
    return AuthContext(
        subject="anonymous",
        role="anonymous",
        permissions=[],
    )
