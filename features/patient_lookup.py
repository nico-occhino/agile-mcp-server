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
from data.mock_store import get_patient, get_active_ricovero


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
    patient = get_patient(patient_id)

    if patient is None:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    dob = date.fromisoformat(patient["data_nascita"])
    today = date.today()
    # Correct age calculation: subtract 1 if birthday hasn't passed yet this year
    age = today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )

    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['nome']} {patient['cognome']}",
        "age_years": age,
        "data_nascita": patient["data_nascita"],
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
    patient = get_patient(patient_id)

    if patient is None:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    active = get_active_ricovero(patient_id)

    if active:
        return {
            "found": True,
            "patient_id": patient_id,
            "full_name": f"{patient['nome']} {patient['cognome']}",
            "stato": "ricoverato",
            "reparto": active["reparto"],
            "data_ingresso": active["data_ingresso"],
            "diagnosi_principale": active["diagnosi_principale"],
            "farmaci": active["farmaci"],
        }

    # Not currently admitted — find the most recent discharge
    ricoveri = patient.get("ricoveri", [])
    if not ricoveri:
        return {
            "found": True,
            "patient_id": patient_id,
            "full_name": f"{patient['nome']} {patient['cognome']}",
            "stato": "mai_ricoverato",
            "message": "Patient has no recorded hospital admissions.",
        }

    latest = max(ricoveri, key=lambda r: r["data_dimissione"] or "")
    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['nome']} {patient['cognome']}",
        "stato": "dimesso",
        "ultima_dimissione": latest["data_dimissione"],
        "ultimo_reparto": latest["reparto"],
        "ultima_diagnosi": latest["diagnosi_principale"],
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
    patient = get_patient(patient_id)

    if patient is None:
        return {
            "found": False,
            "patient_id": patient_id,
            "error": f"No patient found with ID '{patient_id}'.",
        }

    ricoveri = sorted(
        patient.get("ricoveri", []),
        key=lambda r: r["data_ingresso"],
    )

    return {
        "found": True,
        "patient_id": patient_id,
        "full_name": f"{patient['nome']} {patient['cognome']}",
        "total_admissions": len(ricoveri),
        "admissions": [
            {
                "id_ricovero": r["id_ricovero"],
                "reparto": r["reparto"],
                "data_ingresso": r["data_ingresso"],
                "data_dimissione": r["data_dimissione"],
                "diagnosi_principale": r["diagnosi_principale"],
                "stato": r["stato"],
            }
            for r in ricoveri
        ],
    }