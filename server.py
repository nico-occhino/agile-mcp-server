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

from fastmcp import FastMCP
from guardrails.decision import evaluate_guardrail

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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # stdio transport is the default and works with mcp-cli and Claude Desktop.
    # Switch to "sse" when Agile wants to connect over HTTP.
    mcp.run(transport="stdio")
