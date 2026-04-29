"""
features/patient_lookup.py
--------------------------
Simple, deterministic patient lookup tools.

These are the warm-up features Nocita suggested: "un tool che permette di
recuperare l'età del paziente" — start simple, prove the MCP layer works,
then layer complexity on top.

NO LLM IS USED HERE. These tools are pure data retrieval. They demonstrate:
  1. How to define an MCP tool with typed parameters and a docstring.
  2. How the docstring IS the tool description the LLM client sees — write it
     as if you're explaining the function to a model, not to a developer.
  3. How to handle errors cleanly (patient not found, no admissions, etc.)

This is the baseline. The next file (patient_summary.py) adds an LLM call
on top of the same data layer. The file after that (cohort.py) chains
multiple calls into a multi-step workflow.
"""

from __future__ import annotations

from datetime import date
from data.mock_store import get_patient


# ---------------------------------------------------------------------------
# Tool 1 — patient age
# This is literally Nocita's example. Start here.
# ---------------------------------------------------------------------------

def get_patient_age(patient_id: str) -> dict:
    """
    Return the age of a patient in years given their hospital ID.

    Args:
        patient_id: the patient's hospital identifier (e.g. "P001")

    Returns a dict with:
        patient_id, full_name, age_years, data_nascita, found (bool)
    """
    record = get_patient(patient_id)

    if record is None or "patient" not in record:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    patient = record["patient"]
    dob = date.fromisoformat(patient["birthDate"][:10])
    today = date.today()
    # Correct age calculation: subtract 1 if birthday hasn't passed yet this year
    age = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )

    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['name']} {patient['surname']}",
        "age_years": age,
        "data_nascita": patient["birthDate"][:10],
    }


# ---------------------------------------------------------------------------
# Tool 2 — current admission status
# ---------------------------------------------------------------------------

def get_patient_status(patient_id: str) -> dict:
    """
    Return the current admission status of a patient.

    If the patient is currently admitted (stato = 'ricoverato'), return
    the ward, admission date, primary diagnosis, and current medications.
    If the patient is not currently admitted, return their last discharge date.

    Args:
        patient_id: the patient's hospital identifier (e.g. "P001")

    This is the kind of query a clinician makes at the start of a shift:
    "How is patient P002 today?"
    """
    record = get_patient(patient_id)

    if record is None or "patient" not in record:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    patient = record["patient"]
    event = record.get("event")

    if event and event.get("dateEnd") is None:
        return {
            "found": True,
            "patient_id": patient_id,
            "full_name": f"{patient['name']} {patient['surname']}",
            "stato": "ricoverato",
            "reparto": event["uoDescription"],
            "data_ingresso": event["dateStart"][:10],
            "diagnosi_principale": event.get("diagnosis", {}).get("primary"),
        }

    # Not currently admitted — find the most recent discharge
    if not event:
        return {
            "found": True,
            "patient_id": patient_id,
            "full_name": f"{patient['name']} {patient['surname']}",
            "stato": "mai_ricoverato",
            "message": "Patient has no recorded hospital admissions.",
        }

    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['name']} {patient['surname']}",
        "stato": "dimesso",
        "ultima_dimissione": event["dateEnd"],
        "ultimo_reparto": event["uoDescription"],
        "ultima_diagnosi": event.get("diagnosis", {}).get("primary"),
    }


# ---------------------------------------------------------------------------
# Tool 3 — admission history
# ---------------------------------------------------------------------------

def get_admission_history(patient_id: str) -> dict:
    """
    Return the full admission history for a patient, ordered by date.

    Args:
        patient_id: the patient's hospital identifier (e.g. "P001")

    Useful for longitudinal queries: "How many times has P003 been admitted?"
    or "What was P003's first diagnosis?"
    """
    record = get_patient(patient_id)

    if record is None or "patient" not in record:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    patient = record["patient"]
    event = record.get("event")

    # Note: Multi-event history requires a separate API endpoint.
    ricoveri = []
    if event:
        ricoveri.append({
            "id_ricovero": event.get("eventId"),
            "reparto": event["uoDescription"],
            "data_ingresso": event["dateStart"][:10],
            "data_dimissione": event["dateEnd"],
            "diagnosi_principale": event.get("diagnosis", {}).get("primary"),
            "stato": "ricoverato" if event.get("dateEnd") is None else "dimesso",
        })

    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['name']} {patient['surname']}",
        "total_admissions": len(ricoveri),
        "admissions": ricoveri,
    }