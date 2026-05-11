"""Phase 2 JWT authentication and authorization helpers."""

from auth.context import AuthContext, anonymous_context
from auth.jwt import (
    AuthError,
    InvalidTokenError,
    MissingTokenError,
    auth_context_from_optional_token,
    decode_and_verify_jwt,
    is_auth_enabled,
)
from auth.permissions import (
    AuthorizationError,
    authorize_tool_call,
    has_required_permissions,
    required_permissions_for_tool,
)

__all__ = [
    "AuthContext",
    "AuthError",
    "AuthorizationError",
    "InvalidTokenError",
    "MissingTokenError",
    "anonymous_context",
    "auth_context_from_optional_token",
    "authorize_tool_call",
    "decode_and_verify_jwt",
    "has_required_permissions",
    "is_auth_enabled",
    "required_permissions_for_tool",
]
