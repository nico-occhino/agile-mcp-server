"""
client/router.py
----------------
route(ir: IR) -> tuple[str, dict] | None

Pure lookup table — maps IR classes to (mcp_tool_name, kwargs_dict).
Zero LLM calls. Zero business logic. Zero imports from workflow/.
"""

from __future__ import annotations

from client.ir_schema import (
    IR,
    GetPatientAgeIR,
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientDischargeDraftIR,
    GetAdmissionHistoryIR,
    GetPatientsByDiagnosisIR,
    GetCohortSummaryIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)


def route(ir: IR) -> tuple[str, dict] | None:
    """
    Map an IR object to an (mcp_tool_name, kwargs) pair.

    Returns None for UnknownIR or any unrecognised type so that the caller
    can surface a clarification message without crashing.
    """
    _table = {
        GetPatientAgeIR:            lambda r: ("get_patient_age",             {"patient_id": r.patient_id}),
        GetPatientStatusIR:         lambda r: ("get_patient_status",          {"patient_id": r.patient_id}),
        GetPatientSummaryIR:        lambda r: ("get_patient_summary",         {"patient_id": r.patient_id}),
        GetPatientDischargeDraftIR: lambda r: ("get_patient_discharge_draft", {"patient_id": r.patient_id}),
        GetAdmissionHistoryIR:      lambda r: ("get_admission_history",       {"patient_id": r.patient_id}),
        GetPatientsByDiagnosisIR:   lambda r: ("get_patients_by_diagnosis",   {"icd10_prefix": r.diagnosis_prefix}),
        GetCohortSummaryIR:         lambda r: ("get_cohort_summary",          {"icd10_prefix": r.diagnosis_prefix}),
        GetRecentlyAdmittedIR:      lambda r: ("get_recently_admitted",       {"days": r.days}),
    }

    handler = _table.get(type(ir))
    if handler is None:
        return None
    return handler(ir)
