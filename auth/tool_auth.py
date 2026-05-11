"""Structured authorization helper for future MCP tool wrappers."""

from __future__ import annotations

from auth.context import AuthContext
from auth.permissions import (
    AuthorizationError,
    authorize_tool_call,
    required_permissions_for_tool,
)


def authorize_or_error(context: AuthContext, tool_name: str) -> dict | None:
    """Return None when authorized, otherwise a client-safe error payload."""
    try:
        authorize_tool_call(context, tool_name)
    except AuthorizationError:
        return {
            "authorized": False,
            "error": "Unauthorized: missing required permissions.",
            "tool": tool_name,
            "required_permissions": required_permissions_for_tool(tool_name),
            "subject": context.subject,
            "role": context.role,
        }
    return None
