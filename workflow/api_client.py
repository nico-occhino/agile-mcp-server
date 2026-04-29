"""
workflow/api_client.py
----------------------
HTTP client for Agile's hospital APIs.

CURRENT STATUS: Phase 1 stub — not used yet.

In Phase 2, when Nocita provides the real API contracts, this module
will replace direct mock_store lookups in the feature files. The feature
code itself shouldn't change — only the import at the bottom of each
feature file switches from:

    from data.mock_store import get_patient

to:

    from workflow.api_client import get_patient

EXPECTED INTERFACE (fill in when API contracts arrive)
------------------------------------------------------
async def get_patient(patient_id: str) -> dict | None: ...
async def list_patients_by_diagnosis(icd10_prefix: str) -> list[dict]: ...
async def get_recently_admitted(days: int) -> list[dict]: ...

WHY ASYNC
---------
Real HTTP calls should be async so the MCP server can handle concurrent
requests without blocking. httpx.AsyncClient is the right tool.
For Phase 1 the mock store is synchronous and that's fine.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

AGILE_API_BASE_URL = os.getenv("AGILE_API_BASE_URL", "")
AGILE_API_KEY = os.getenv("AGILE_API_KEY", "")


# ---------------------------------------------------------------------------
# Placeholder — implement when Nocita sends API contracts
# ---------------------------------------------------------------------------

async def get_patient(patient_id: str) -> dict | None:
    """Fetch a single patient record from Agile's API."""
    raise NotImplementedError(
        "Phase 2 not started. Use data.mock_store.get_patient() for now."
    )


async def list_patients_by_diagnosis(icd10_prefix: str) -> list[dict]:
    """Fetch patients by ICD-10 diagnosis prefix."""
    raise NotImplementedError(
        "Phase 2 not started. Use data.mock_store.list_patients_by_diagnosis() for now."
    )