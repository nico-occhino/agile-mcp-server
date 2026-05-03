"""Route validated IR objects to existing local clinical feature functions."""

from __future__ import annotations

from features.cohort import get_patients_by_diagnosis, get_recently_admitted
from features.patient_lookup import get_patient_status
from features.patient_summary import get_patient_summary
from orchestrator.ir_schema import (
    IR,
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)


def route_and_execute(ir: IR) -> dict:
    """Execute the curated tool selected by a validated IR object."""
    if isinstance(ir, UnknownIR):
        return {
            "found": False,
            "error": "UnknownIR non puo essere instradato a un tool clinico.",
        }

    if isinstance(ir, GetPatientStatusIR):
        return get_patient_status(ir.patient_id)

    if isinstance(ir, GetPatientSummaryIR):
        return get_patient_summary(ir.patient_id)

    if isinstance(ir, GetPatientsByDiagnosisIR):
        return get_patients_by_diagnosis(ir.diagnosis_code_prefix)

    if isinstance(ir, GetRecentlyAdmittedIR):
        return get_recently_admitted(days=ir.days)

    return {"found": False, "error": "Intento IR non supportato dal router."}
