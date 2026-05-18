"""
server.py
---------
FastMCP server entrypoint. Wires all features as MCP tools.

HOW MCP WORKS (read this before the MCP docs, it will orient you)
-----------------------------------------------------------------
MCP (Model Context Protocol) is a client-server protocol. The server (you)
exposes "tools" — typed, documented functions. The client (Agile's gateway,
or Claude Desktop, or mcp-cli during development) discovers the tool list
and decides which one to call based on the tool's name and docstring.

The protocol sits on top of JSON-RPC 2.0. Under the hood:
  - The client opens a connection (stdio, HTTP, or SSE).
  - It sends: initialize → tools/list → tools/call {name, arguments}
  - The server responds with the tool's return value.

FastMCP handles all of this. You just decorate functions with @mcp.tool()
and it auto-generates the JSON schema from the Python type hints and the
OpenAPI-style docstring.

WHAT THE CLIENT SEES
--------------------
For each @mcp.tool(), the client gets:
  {
    "name": "get_patient_age",
    "description": "Return the age of a patient...",   ← from the docstring
    "inputSchema": {                                    ← from type hints
      "type": "object",
      "properties": {
        "patient_id": {"type": "string", "description": "..."}
      },
      "required": ["patient_id"]
    }
  }

The description IS the docstring. The model reads it to decide whether
this tool is the right one for the user's query. Write docstrings for
tools as if you're writing instructions for a slightly confused AI
colleague, not for a developer.

HOW TO RUN
----------
Development (with stdio transport — for testing with mcp-cli):
    python server.py

Development (with SSE transport — for HTTP-based clients):
    fastmcp run server.py --transport sse --port 8000

Connect Claude Desktop:
    Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "agile-hospital": {
          "command": "python",
          "args": ["/absolute/path/to/server.py"]
        }
      }
    }

Test with mcp-cli:
    pip install mcp-cli
    mcp-cli --server "python server.py" tools list
    mcp-cli --server "python server.py" tools call get_patient_age '{"patient_id": "P001"}'
"""

from dotenv import load_dotenv
load_dotenv()  # must happen before any module that reads env vars is imported

from workflow.logging import configure_logging
configure_logging()

from workflow.instrumentation import instrumented

from auth.context import anonymous_context
from auth.jwt import (
    AuthError,
    MissingTokenError,
    decode_and_verify_jwt,
    is_auth_enabled,
)
from auth.permissions import has_required_permissions, required_permissions_for_tool
from auth.request_auth import get_current_auth_context
from fastmcp import FastMCP
from guardrails.decision import evaluate_guardrail
from guardrails.input_guardrail import (
    evaluate_input_prompt_guardrail as run_input_prompt_guardrail,
)

# Import all feature functions
from features.patient_lookup import (
    get_patient_age,
    get_patient_status,
    get_admission_history,
)
from features.patient_summary import (
    get_patient_summary,
    get_patient_discharge_draft,
)
from features.cohort import (
    get_patients_by_diagnosis,
    get_cohort_summary,
    get_recently_admitted,
)


def evaluate_clinical_output_guardrail(
    task_type: str,
    confidence: float | None = None,
) -> dict:
    """
    Evaluate whether a generated clinical output should be accepted, reviewed,
    or rejected.

    This tool does not access patient data and does not call the LLM. It only
    applies the uncertainty-aware guardrail policy to a task type and confidence
    score. The confidence value is a semantic self-consistency score from
    repeated LLM sampling; it is not a guarantee of clinical truth.

    Args:
        task_type: Clinical task category, e.g. patient_summary,
            discharge_draft, medication_or_therapy, administrative_summary.
        confidence: Optional semantic self-consistency score in [0, 1].
    """
    return evaluate_guardrail(
        task_type=task_type,
        confidence=confidence,
    ).model_dump()


def evaluate_input_prompt_guardrail(
    query: str,
    use_llm_classifier: bool = False,
) -> dict:
    """
    Evaluate a user query before NL2API parsing, routing, or execution.

    This input guardrail blocks obvious prompt injection and unsafe requests
    with deterministic static filtering. Optionally, an LLM classifier can
    produce structured risk scores for ambiguous prompts, but the final
    ALLOW / CLARIFY / REQUIRES_REVIEW / BLOCK decision is always made by
    deterministic policy rules. This tool does not access patient data and
    does not execute clinical tools.

    Args:
        query: User request to evaluate as untrusted input.
        use_llm_classifier: Whether to use optional structured LLM scoring
            after the static filter passes.
    """
    return run_input_prompt_guardrail(
        query=query,
        use_llm_classifier=use_llm_classifier,
    ).model_dump()


def decode_jwt_auth_context(token: str | None = None) -> dict:
    """
    Decode and verify a JWT into a typed Phase 2 caller context.

    This demo tool supports future Aria integration work. If a token is
    provided, it is verified and mapped into AuthContext regardless of
    AUTH_ENABLED. If no token is provided and AUTH_ENABLED is false, an
    anonymous context is returned. It does not access patient data and does not
    execute clinical tools.

    Args:
        token: Optional JWT string to decode and verify.
    """
    try:
        if token:
            return {
                "auth_enabled": is_auth_enabled(),
                "context": decode_and_verify_jwt(token).model_dump(),
            }
        if not is_auth_enabled():
            return {
                "auth_enabled": False,
                "context": anonymous_context().model_dump(),
            }
        raise MissingTokenError("Missing authentication token.")
    except AuthError as exc:
        return _auth_error_payload(exc, tool_name="decode_jwt_auth_context")


def authorize_tool_access(tool_name: str, token: str | None = None) -> dict:
    """
    Check whether a JWT/auth context can access a given MCP tool.

    This Phase 2 demo tool verifies the optional JWT, extracts caller context,
    and evaluates the deterministic tool permission policy. It only checks
    access; it does not execute the requested tool or access patient data.

    Args:
        tool_name: MCP tool name to check.
        token: Optional JWT string to decode and verify.
    """
    context_result = _context_for_auth_demo(token, tool_name)
    if "error" in context_result:
        return context_result

    context = context_result["context"]
    required = required_permissions_for_tool(tool_name)
    authorized = has_required_permissions(context, tool_name)
    return {
        "authorized": authorized,
        "tool": tool_name,
        "subject": context.subject,
        "role": context.role,
        "required_permissions": required,
        "permissions": context.permissions,
        "error": None if authorized else "Unauthorized: missing required permissions.",
    }


def current_jwt_auth_context() -> dict:
    """
    Return the JWT AuthContext from the current HTTP Authorization header.

    This Phase 2 demo tool tests real header-based JWT extraction for MCP/SSE
    calls. It reads Authorization: Bearer <token> from the current FastMCP HTTP
    request context, verifies the token when present, and returns the caller
    context. It does not access patient data and does not execute clinical
    tools.
    """
    try:
        return {
            "auth_enabled": is_auth_enabled(),
            "context": get_current_auth_context().model_dump(),
        }
    except AuthError as exc:
        return _auth_error_payload(exc, tool_name="current_jwt_auth_context")


def _context_for_auth_demo(token: str | None, tool_name: str) -> dict:
    try:
        if token:
            return {"context": decode_and_verify_jwt(token)}
        if not is_auth_enabled():
            return {"context": anonymous_context()}
        raise MissingTokenError("Missing authentication token.")
    except AuthError as exc:
        return _auth_error_payload(exc, tool_name=tool_name)


def _auth_error_payload(exc: AuthError, tool_name: str) -> dict:
    return {
        "authorized": False,
        "tool": tool_name,
        "error": str(exc),
        "auth_enabled": is_auth_enabled(),
    }

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
# The name "agile-hospital" is what Agile's gateway will see when it connects.
# It identifies this MCP server among others in the Agile ecosystem.

mcp = FastMCP(
    name="agile-hospital",
    instructions=(
        "This MCP server provides clinical data tools for hospital management. "
        "All patient data is currently synthetic (Phase 1 POC). "
        "Tools that involve LLM processing return a 'confidence' score (0–1) and "
        "a 'confidence_level' (HIGH/MEDIUM/LOW). For LOW confidence results, "
        "surface a clarification request to the clinician rather than displaying "
        "the result directly."
    ),
)

# ---------------------------------------------------------------------------
# Register tools
# The @mcp.tool() decorator is all you need.
# FastMCP reads the function signature and docstring automatically.
#
# Naming convention: verb_noun or verb_noun_modifier
#   get_patient_age          ← simple lookup
#   get_patient_summary      ← LLM-based, returns UncertainResult
#   get_cohort_summary       ← multi-step workflow
# ---------------------------------------------------------------------------

# --- Deterministic lookups (no LLM) ---
mcp.tool()(instrumented("get_patient_age")(get_patient_age))
mcp.tool()(instrumented("get_patient_status")(get_patient_status))
mcp.tool()(instrumented("get_admission_history")(get_admission_history))

# --- LLM-based, uncertainty-aware ---
mcp.tool()(instrumented("get_patient_summary")(get_patient_summary))
mcp.tool()(instrumented("get_patient_discharge_draft")(get_patient_discharge_draft))

# --- Multi-step workflows ---
mcp.tool()(instrumented("get_patients_by_diagnosis")(get_patients_by_diagnosis))
mcp.tool()(instrumented("get_cohort_summary")(get_cohort_summary))
mcp.tool()(instrumented("get_recently_admitted")(get_recently_admitted))

# --- Guardrail/evaluation tools (no patient data access) ---
mcp.tool(
    annotations={
        # This tool is deterministic, local, and policy-only. These hints help
        # MCP clients such as Inspector or Aria understand that calling it does
        # not read patient data, mutate state, call external services, or depend
        # on open-world side effects.
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)(instrumented("evaluate_clinical_output_guardrail")(evaluate_clinical_output_guardrail))

mcp.tool(
    annotations={
        # Intended semantics: local policy evaluation only; no patient data
        # access, mutation, clinical execution, or open-world side effects.
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)(instrumented("evaluate_input_prompt_guardrail")(evaluate_input_prompt_guardrail))

mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)(instrumented("decode_jwt_auth_context")(decode_jwt_auth_context))

mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)(instrumented("authorize_tool_access")(authorize_tool_access))

mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)(instrumented("current_jwt_auth_context")(current_jwt_auth_context))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # stdio transport is the default and works with mcp-cli and Claude Desktop.
    # Switch to "sse" when Agile wants to connect over HTTP.
    mcp.run(transport="stdio")
