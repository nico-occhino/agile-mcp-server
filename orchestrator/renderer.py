"""Render orchestrator results into simple Italian demo text."""

from __future__ import annotations

from orchestrator.ir_schema import (
    IR,
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)


def render_response(ir: IR, result: dict) -> str:
    """Render a routed tool result for a clinician-facing Phase 1 demo."""
    if isinstance(ir, UnknownIR):
        return f"Richiesta non eseguibile: {ir.reason}"

    if result.get("found") is False:
        return f"Errore: {result.get('error', 'Risultato non trovato.')}"

    if isinstance(ir, GetPatientStatusIR):
        return _render_patient_status(result)

    if isinstance(ir, GetPatientSummaryIR):
        return _render_patient_summary(result)

    if isinstance(ir, GetPatientsByDiagnosisIR):
        return _render_diagnosis_cohort(result)

    if isinstance(ir, GetRecentlyAdmittedIR):
        return _render_recent_admissions(result)

    return "Risposta disponibile, ma il formato non e supportato dal renderer."


def _render_patient_status(result: dict) -> str:
    lines = [
        f"Paziente: {result.get('full_name', result.get('patient_id', 'sconosciuto'))}",
        f"Stato: {result.get('stato', 'non disponibile')}",
    ]
    ward = result.get("reparto") or result.get("ultimo_reparto")
    diagnosis = result.get("diagnosi_principale") or result.get("ultima_diagnosi")
    if ward:
        lines.append(f"Reparto: {ward}")
    if diagnosis:
        lines.append(f"Diagnosi principale: {diagnosis}")
    if result.get("data_ingresso"):
        lines.append(f"Data ingresso: {result['data_ingresso']}")
    if result.get("ultima_dimissione"):
        lines.append(f"Ultima dimissione: {result['ultima_dimissione']}")
    if result.get("message"):
        lines.append(result["message"])
    return "\n".join(lines)


def _render_patient_summary(result: dict) -> str:
    lines = [f"Riassunto paziente {result.get('patient_id', '')}:".strip()]
    summary = result.get("result") or result.get("summary")
    if summary:
        lines.append(str(summary))
    confidence = result.get("confidence")
    confidence_level = result.get("confidence_level")
    if confidence is not None or confidence_level:
        metadata = "Confidenza"
        if confidence_level:
            metadata += f": {confidence_level}"
        if confidence is not None:
            metadata += f" ({confidence:.0%})"
        lines.append(metadata)
    if result.get("rationale"):
        lines.append(f"Nota: {result['rationale']}")
    return "\n".join(lines)


def _render_diagnosis_cohort(result: dict) -> str:
    lines = [
        f"Pazienti trovati per diagnosi {result.get('diagnosis_code_prefix', '')}: {result.get('total_found', 0)}"
    ]
    patients = result.get("patients", [])
    if not patients and result.get("message"):
        lines.append(result["message"])
    for patient in patients:
        parts = [
            patient.get("patient_id", "ID sconosciuto"),
            patient.get("full_name", "nome non disponibile"),
        ]
        if patient.get("stato"):
            parts.append(f"stato {patient['stato']}")
        if patient.get("reparto") or patient.get("ultimo_reparto"):
            parts.append(f"reparto {patient.get('reparto') or patient.get('ultimo_reparto')}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def _render_recent_admissions(result: dict) -> str:
    lines = [
        f"Ricoveri negli ultimi {result.get('look_back_days', '?')} giorni: {result.get('total_found', 0)}"
    ]
    for admission in result.get("admissions", []):
        lines.append(
            "- "
            + " | ".join(
                [
                    admission.get("patient_id", "ID sconosciuto"),
                    admission.get("full_name", "nome non disponibile"),
                    admission.get("data_ingresso", "data non disponibile"),
                    admission.get("reparto", "reparto non disponibile"),
                ]
            )
        )
    return "\n".join(lines)
