"""
features/patient_lookup.py
--------------------------
Deterministic patient lookup tools — no LLM, pure data retrieval.
Schema: Nocita's real API shape (event + patient blocks, numeric IDs).
"Currently admitted" = event["dateEnd"] is None.
"""
from __future__ import annotations
from datetime import date
from data.mock_store import get_patient

def get_patient_age(patient_id: str) -> dict:
    """
    Return the age of a patient in years given their hospital ID.

    Args:
        patient_id: the patient's numeric hospital identifier (e.g. "45")
    """
    record = get_patient(patient_id)
    if record is None or "patient" not in record:
        return {"found": False, "patient_id": patient_id, "error": f"No patient found with ID '{patient_id}'."}
    patient = record["patient"]
    dob = date.fromisoformat(patient["birthDate"][:10])
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return {"found": True, "patient_id": patient_id, "full_name": f"{patient['name']} {patient['surname']}", "age_years": age, "data_nascita": patient["birthDate"][:10]}

def get_patient_status(patient_id: str) -> dict:
    """
    Return the current admission status of a patient.

    If dateEnd is None the patient is currently admitted.
    If dateEnd is set the patient has been discharged.

    Args:
        patient_id: the patient's numeric hospital identifier (e.g. "45")
    """
    record = get_patient(patient_id)
    if record is None or "patient" not in record:
        return {"found": False, "patient_id": patient_id, "error": f"No patient found with ID '{patient_id}'."}
    patient = record["patient"]
    event = record.get("event")
    if event and event.get("dateEnd") is None:
        return {"found": True, "patient_id": patient_id, "full_name": f"{patient['name']} {patient['surname']}", "stato": "ricoverato", "reparto": event["uoDescription"], "data_ingresso": event["dateStart"][:10], "diagnosi_principale": event.get("diagnosis", {}).get("primary")}
    if not event:
        return {"found": True, "patient_id": patient_id, "full_name": f"{patient['name']} {patient['surname']}", "stato": "mai_ricoverato", "message": "Patient has no recorded hospital admissions."}
    return {"found": True, "patient_id": patient_id, "full_name": f"{patient['name']} {patient['surname']}", "stato": "dimesso", "ultima_dimissione": event["dateEnd"], "ultimo_reparto": event["uoDescription"], "ultima_diagnosi": event.get("diagnosis", {}).get("primary")}

def get_admission_history(patient_id: str) -> dict:
    """
    Return the admission record(s) for a patient.

    NOTE: Agile's current API returns one event per request.
    Multi-event history requires a dedicated history endpoint (Phase 2).

    Args:
        patient_id: the patient's numeric hospital identifier (e.g. "45")
    """
    record = get_patient(patient_id)
    if record is None or "patient" not in record:
        return {"found": False, "patient_id": patient_id, "error": f"No patient found with ID '{patient_id}'."}
    patient = record["patient"]
    event = record.get("event")
    ricoveri = []
    if event:
        ricoveri.append({"id_ricovero": event.get("eventId"), "reparto": event["uoDescription"], "data_ingresso": event["dateStart"][:10], "data_dimissione": event["dateEnd"], "diagnosi_principale": event.get("diagnosis", {}).get("primary"), "stato": "ricoverato" if event.get("dateEnd") is None else "dimesso"})
    return {"found": True, "patient_id": patient_id, "full_name": f"{patient['name']} {patient['surname']}", "total_admissions": len(ricoveri), "admissions": ricoveri}