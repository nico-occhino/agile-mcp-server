"""Permission policy for MCP tool authorization."""

from __future__ import annotations

from auth.context import AuthContext


class AuthorizationError(Exception):
    """Raised when a caller lacks required tool permissions."""


TOOL_PERMISSION_POLICY: dict[str, list[str]] = {
    "get_patient_age": ["patient:read"],
    "get_patient_status": ["patient:read"],
    "get_admission_history": ["patient:read"],
    "get_patient_summary": ["patient:read", "summary:generate"],
    "get_patient_discharge_draft": ["patient:read", "discharge:generate"],
    "get_patients_by_diagnosis": ["cohort:read"],
    "get_cohort_summary": ["cohort:read", "summary:generate"],
    "get_recently_admitted": ["cohort:read"],
    "evaluate_clinical_output_guardrail": [],
    "evaluate_input_prompt_guardrail": [],
    "decode_jwt_auth_context": [],
    "authorize_tool_access": [],
    "current_jwt_auth_context": [],
}


def required_permissions_for_tool(tool_name: str) -> list[str]:
    """Return required permissions, defaulting unknown tools to admin only."""
    return TOOL_PERMISSION_POLICY.get(tool_name, ["admin"])


def has_required_permissions(context: AuthContext, tool_name: str) -> bool:
    """Return whether context can call the named tool."""
    required = required_permissions_for_tool(tool_name)
    if not required:
        return True
    if context.role == "admin":
        return True
    permissions = set(context.permissions)
    return all(permission in permissions for permission in required)


def authorize_tool_call(context: AuthContext, tool_name: str) -> None:
    """Raise AuthorizationError if the context cannot call the tool."""
    if not has_required_permissions(context, tool_name):
        raise AuthorizationError("Unauthorized: missing required permissions.")
