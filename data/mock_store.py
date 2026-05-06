"""
data/mock_store.py
------------------
Synthetic patient records shaped after Agile's real API response structure.

SCHEMA SOURCE
-------------
This schema was derived from the real API response provided by Ing. Nocita
on 2026-04-29. Field names, nesting, and value formats mirror production.
Fake content, real shape.

TOP-LEVEL STRUCTURE
--------------------
Each record has two keys:
  "event"   — the hospital admission (ricovero) data
  "patient" — the patient's demographic and clinical data

This differs from the SDO flat structure we assumed initially.
Key differences from our first mock:
  - Patient IDs are numeric strings ("45"), not "P001"-style
  - Diagnoses use Italian ministerial codes ("0123"), not ICD-10 ("I21.9")
  - Allergy field is on the patient object — safety-critical
  - eventType: "ORDINARIO" | "DAY_SURGERY" | "URGENTE"
  - eventSource: "OSP" (ospedaliero) | "PS" (pronto soccorso)
  - dateStart/dateEnd: "YYYY-MM-DD HH:MM:SS" format

DIAGNOSIS CODES
---------------
Nocita's system uses the Italian ministerial coding system (ICD-9-CM adapted),
not ICD-10. "0123" is a placeholder — real codes are 3-5 digit numeric strings.
Update this mapping as you learn the domain from Nocita.

EXTENSION POINT
---------------
When Nocita's real API is accessible, replace get_patient() with a call to
workflow/api_client.py. The access functions below (get_patient, list_patients,
etc.) are the only interface the feature code uses — swap the backend without
touching features.
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Raw data — mirrors Nocita's real API response structure
# ---------------------------------------------------------------------------

PATIENTS: dict[str, dict] = {
    "45": {
        "event": {
            "diagnosis": {
                "primary": "0123",       # Italian ministerial code, not ICD-10
                "secondary": {},
            },
            "surgeries": {
                "primary": None,
                "secondary": {},
            },
            "drg": {
                "code": "210",
                "type": None,
                "value": None,
                "mdc": "08",             # Major Diagnostic Category 08 = musculoskeletal
                "weight": "2.345",
            },
            "sdo": {
                "eventTicketPrice": None,
                "eventPaccCode": None,
                "eventExemptionCode": None,
                "eventPriorityClass": "C",        # C = Urgent
                "eventBookingDate": "2026-01-01",
                "eventPatientOrigin": "10",       # origin code (territory)
                "eventHospitalBurden": "A",
                "eventDischargeMode": "02",       # 02 = ordinary discharge
            },
            "adt": {
                "eventFirstUoId": "1",
                "eventLastUoId": "1",
                "eventLastTransferId": "183",
            },
            "patientId": "45",
            "eventType": "ORDINARIO",            # ORDINARIO | DAY_SURGERY | URGENTE
            "eventSource": "OSP",                # OSP = ospedaliero, PS = pronto soccorso
            "eventCode": "GSO_2026000001",
            "eventId": "129",
            "dateStart": "2026-04-28 15:07:00",
            "dateArchived": "2026-04-28 15:08:40",
            "uoDescription": "Reparto Ortopedia",
            "eventReason": (
                "Paziente ricoverato per dolore acuto e limitazione funzionale "
                "dell'arto inferiore destro per trauma accidentale"
            ),
            "isDaySurgery": False,
            "uoCode": "3601",
            "uoId": "1",
            "regionalCode": "190207",
            "namePresidio": "Presidio A",
            "dateEnd": "2026-04-28 15:08:00",
            "isPreOsp": False,
        },
        "patient": {
            "id": "678928fec16f503b2715e9ea",     # MongoDB ObjectId from Agile's DB
            "internalId": "45",
            "name": "Mario",
            "surname": "Rossi",
            "fiscalCode": "RSSMRA91A01C351R",
            "birthDate": "1991-01-01 00:00:00",
            "allergy": "Arachidi",               # SAFETY-CRITICAL — always surface this
        },
    },
    "46": {
        "event": {
            "diagnosis": {
                "primary": "4280",               # Heart failure
                "secondary": {"1": "2500"},      # Diabetes mellitus
            },
            "surgeries": {
                "primary": None,
                "secondary": {},
            },
            "drg": {
                "code": "127",
                "type": None,
                "value": None,
                "mdc": "05",                     # MDC 05 = cardiovascular
                "weight": "1.890",
            },
            "sdo": {
                "eventTicketPrice": None,
                "eventPaccCode": None,
                "eventExemptionCode": None,
                "eventPriorityClass": "U",        # U = Emergency
                "eventBookingDate": None,
                "eventPatientOrigin": "10",
                "eventHospitalBurden": "A",
                "eventDischargeMode": None,       # None = still admitted
            },
            "adt": {
                "eventFirstUoId": "3",
                "eventLastUoId": "3",
                "eventLastTransferId": "201",
            },
            "patientId": "46",
            "eventType": "URGENTE",
            "eventSource": "PS",                  # came via pronto soccorso
            "eventCode": "GSO_2026000002",
            "eventId": "130",
            "dateStart": "2026-04-27 09:15:00",
            "dateArchived": None,                 # None = still admitted
            "uoDescription": "Reparto Cardiologia",
            "eventReason": (
                "Paziente giunta per dispnea ingravescente e edemi declivi bilaterali. "
                "Scompenso cardiaco acuto su cardiopatia ischemica nota."
            ),
            "isDaySurgery": False,
            "uoCode": "3201",
            "uoId": "3",
            "regionalCode": "190207",
            "namePresidio": "Presidio A",
            "dateEnd": None,                      # None = still admitted
            "isPreOsp": False,
        },
        "patient": {
            "id": "678928fec16f503b2715e9eb",
            "internalId": "46",
            "name": "Giovanna",
            "surname": "Ferrara",
            "fiscalCode": "FRRGNN72P70C351K",
            "birthDate": "1972-09-30 00:00:00",
            "allergy": None,                      # no known allergies
        },
    },
    "47": {
        "event": {
            "diagnosis": {
                "primary": "4349",               # Cerebrovascular accident
                "secondary": {"1": "4011", "2": "2720"},
            },
            "surgeries": {
                "primary": None,
                "secondary": {},
            },
            "drg": {
                "code": "014",
                "type": None,
                "value": None,
                "mdc": "01",                     # MDC 01 = nervous system
                "weight": "1.654",
            },
            "sdo": {
                "eventTicketPrice": None,
                "eventPaccCode": None,
                "eventExemptionCode": None,
                "eventPriorityClass": "U",
                "eventBookingDate": None,
                "eventPatientOrigin": "10",
                "eventHospitalBurden": "A",
                "eventDischargeMode": "02",
            },
            "adt": {
                "eventFirstUoId": "5",
                "eventLastUoId": "5",
                "eventLastTransferId": "220",
            },
            "patientId": "47",
            "eventType": "URGENTE",
            "eventSource": "PS",
            "eventCode": "GSO_2026000003",
            "eventId": "131",
            "dateStart": "2026-04-20 03:42:00",
            "dateArchived": "2026-04-29 10:00:00",
            "uoDescription": "Reparto Neurologia",
            "eventReason": (
                "Paziente con emiplegia sinistra a insorgenza acuta. "
                "TC cranio: lesione ischemica territorio ACM destra."
            ),
            "isDaySurgery": False,
            "uoCode": "3501",
            "uoId": "5",
            "regionalCode": "190207",
            "namePresidio": "Presidio A",
            "dateEnd": "2026-04-29 10:00:00",
            "isPreOsp": False,
        },
        "patient": {
            "id": "678928fec16f503b2715e9ec",
            "internalId": "47",
            "name": "Luca",
            "surname": "Bianchi",
            "fiscalCode": "BNCLCU45S05C351T",
            "birthDate": "1945-11-05 00:00:00",
            "allergy": "Penicillina",            # SAFETY-CRITICAL
        },
    },
    "48": {
        # Patient with no current event — tests edge case handling
        "event": None,
        "patient": {
            "id": "678928fec16f503b2715e9ed",
            "internalId": "48",
            "name": "Sara",
            "surname": "Esposito",
            "fiscalCode": "SPSSRA90L62C351P",
            "birthDate": "1990-07-22 00:00:00",
            "allergy": None,
        },
    },
}


# ---------------------------------------------------------------------------
# Access functions — ALL feature code goes through these, never PATIENTS directly
# ---------------------------------------------------------------------------

def get_patient(patient_id: str) -> Optional[dict]:
    """
    Return the full patient record (event + patient), or None if not found.
    Sanitizes input: strips whitespace, handles both "45" and " 45 ".
    """
    clean_id = patient_id.strip()
    return PATIENTS.get(clean_id)


def get_patient_demographics(patient_id: str) -> Optional[dict]:
    """Return only the patient demographics block, or None."""
    record = get_patient(patient_id)
    if record is None:
        return None
    return record.get("patient")


def get_patient_event(patient_id: str) -> Optional[dict]:
    """Return only the current event (admission) block, or None."""
    record = get_patient(patient_id)
    if record is None:
        return None
    return record.get("event")


def is_currently_admitted(patient_id: str) -> bool:
    """
    Return True if the patient is currently admitted (dateEnd is None).
    This is the real-API pattern: dateEnd=None means still in hospital.
    """
    event = get_patient_event(patient_id)
    if event is None:
        return False
    return event.get("dateEnd") is None


def list_patients() -> list[dict]:
    """Return all patient demographic records (no event detail)."""
    return [
        record["patient"]
        for record in PATIENTS.values()
        if record.get("patient") is not None
    ]


def list_patients_by_diagnosis(diagnosis_code_prefix: str) -> list[dict]:
    """
    Return full records for patients whose primary diagnosis code
    starts with the given prefix.

    NOTE: Agile uses Italian ministerial numeric codes ("4280", "0123"),
    not ICD-10 alphanumeric codes ("I21.9"). Prefix search still works:
    list_patients_by_diagnosis("428") finds all heart failure variants.
    """
    clean_prefix = diagnosis_code_prefix.strip()
    results = []
    for record in PATIENTS.values():
        event = record.get("event")
        if event is None:
            continue
        primary = event.get("diagnosis", {}).get("primary") or ""
        if primary.startswith(clean_prefix):
            results.append(record)
    return results


def list_currently_admitted() -> list[dict]:
    """Return full records for all currently admitted patients."""
    return [
        record for record in PATIENTS.values()
        if record.get("event") is not None
        and record["event"].get("dateEnd") is None
    ]


def get_patient_allergy(patient_id: str) -> Optional[str]:
    """
    Return the patient's allergy string, or None if no known allergies.

    SAFETY NOTE: This field must ALWAYS be checked before any medication
    recommendation. The cohort confidence gate exists partly because of
    this field — a LOW-confidence summary might have missed an allergy.
    """
    demo = get_patient_demographics(patient_id)
    if demo is None:
        return None
    return demo.get("allergy")
