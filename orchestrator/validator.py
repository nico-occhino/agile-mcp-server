"""Validation rules for orchestrator IR objects."""

from __future__ import annotations

from orchestrator.ir_schema import (
    IR,
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)


def validate_ir(ir: IR) -> list[str]:
    """Return validation issues. An empty list means the IR is executable."""
    if isinstance(ir, UnknownIR):
        return [ir.reason]

    if isinstance(ir, (GetPatientStatusIR, GetPatientSummaryIR)):
        if not ir.patient_id:
            return ["patient_id mancante."]
        if not ir.patient_id.isdigit():
            return ["patient_id deve essere numerico."]
        return []

    if isinstance(ir, GetPatientsByDiagnosisIR):
        if not ir.diagnosis_code_prefix:
            return ["diagnosis_code_prefix mancante."]
        if not ir.diagnosis_code_prefix.isdigit():
            return ["diagnosis_code_prefix deve essere numerico."]
        return []

    if isinstance(ir, GetRecentlyAdmittedIR):
        if ir.days <= 0:
            return ["days deve essere positivo."]
        if ir.days > 365:
            return ["days deve essere minore o uguale a 365."]
        return []

    return ["Tipo IR non riconosciuto."]
