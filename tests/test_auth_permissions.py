import pytest

from auth.context import AuthContext
from auth.permissions import (
    AuthorizationError,
    authorize_tool_call,
    has_required_permissions,
    required_permissions_for_tool,
)


def _context(role: str = "doctor", permissions: list[str] | None = None) -> AuthContext:
    return AuthContext(
        subject="user-123",
        role=role,
        permissions=permissions or [],
    )


def test_admin_role_can_access_all_tools():
    context = _context(role="admin")

    assert has_required_permissions(context, "unknown_future_tool") is True
    authorize_tool_call(context, "get_patient_discharge_draft")


def test_doctor_with_patient_read_can_access_patient_status():
    context = _context(permissions=["patient:read"])

    assert has_required_permissions(context, "get_patient_status") is True


def test_doctor_without_discharge_generate_cannot_access_discharge_draft():
    context = _context(permissions=["patient:read"])

    assert has_required_permissions(context, "get_patient_discharge_draft") is False
    with pytest.raises(AuthorizationError):
        authorize_tool_call(context, "get_patient_discharge_draft")


def test_guardrail_tools_require_no_permissions():
    context = _context()

    assert required_permissions_for_tool("evaluate_input_prompt_guardrail") == []
    assert has_required_permissions(context, "evaluate_input_prompt_guardrail") is True
    assert has_required_permissions(context, "evaluate_clinical_output_guardrail") is True


def test_unknown_tool_requires_admin_permission_by_default():
    context = _context(permissions=["patient:read", "summary:generate"])

    assert required_permissions_for_tool("unknown_future_tool") == ["admin"]
    assert has_required_permissions(context, "unknown_future_tool") is False
