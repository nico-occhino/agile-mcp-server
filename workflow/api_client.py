"""
workflow/api_client.py
----------------------
Low-level HTTP client placeholder for Agile's hospital APIs.

CURRENT STATUS: Phase 1 stub; not used by clinical features yet.

In Phase 2, when Agile provides the real API contracts, this module should stay
focused on HTTP mechanics: base URL handling, timeouts, headers, status-code
translation, and JSON parsing. It should not be imported directly by clinical
feature modules.

Intended architecture:

    features -> data.repository -> AgileApiRepository -> workflow.api_client -> Agile API

A future data/agile_api_repository.py should implement PatientRepository using
this client. Repository selection should happen centrally via
data.repository.get_repository() / set_repository(), environment configuration,
or application startup wiring. Feature modules should keep depending on the
repository abstraction, not on httpx or raw Agile API functions.

Downstream JWT forwarding is expected for Aria-originated calls, but the exact
Bearer-token contract is pending Agile API confirmation.

EXPECTED LOW-LEVEL SHAPE (fill in when API contracts arrive)
------------------------------------------------------------
async def request_json(
    method: str,
    path: str,
    *,
    bearer_token: str | None = None,
    ...
) -> dict: ...

async def get_json(
    path: str,
    *,
    bearer_token: str | None = None,
    ...
) -> dict: ...

WHY ASYNC
---------
Real HTTP calls should be async so the MCP server can handle concurrent
requests without blocking. httpx.AsyncClient is the right tool. For Phase 1 the
mock repository is synchronous and that is fine.
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

AGILE_API_BASE_URL = os.getenv("AGILE_API_BASE_URL", "")
AGILE_API_KEY = os.getenv("AGILE_API_KEY", "")


# ---------------------------------------------------------------------------
# Placeholder; implement when Agile sends API contracts.
# ---------------------------------------------------------------------------

async def get_patient(patient_id: str) -> dict | None:
    """Placeholder only; implement through AgileApiRepository in Phase 2."""
    raise NotImplementedError(
        "Phase 2 not started. Implement data.agile_api_repository.AgileApiRepository."
    )


async def list_patients_by_diagnosis(diagnosis_code_prefix: str) -> list[dict]:
    """Placeholder only; implement through PatientRepository in Phase 2."""
    raise NotImplementedError(
        "Phase 2 not started. Implement data.agile_api_repository.AgileApiRepository."
    )
