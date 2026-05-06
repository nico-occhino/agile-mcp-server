"""Deterministic rule-based parser for Phase 1 NL2API tests."""

from __future__ import annotations

import re
import unicodedata

from orchestrator.ir_schema import (
    IR,
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)


def parse_query_to_ir(query: str) -> IR:
    """Parse a natural-language query into the minimal orchestrator IR.

    This parser intentionally uses only regex and string matching. It is the
    Phase 1 replacement point for a future LLM-based parser.
    """
    normalized = _normalize(query)
    if not normalized:
        return UnknownIR(reason="La richiesta e vuota.")

    patient_id = _extract_patient_id(normalized)

    if _looks_like_patient_summary(normalized):
        if patient_id is None:
            return UnknownIR(reason="Richiesta di riassunto paziente senza ID numerico.")
        return GetPatientSummaryIR(patient_id=patient_id)

    if _looks_like_patient_status(normalized):
        if patient_id is None:
            return UnknownIR(reason="Richiesta di stato paziente senza ID numerico.")
        return GetPatientStatusIR(patient_id=patient_id)

    diagnosis_code_prefix = _extract_diagnosis_code_prefix(normalized)
    if diagnosis_code_prefix is not None:
        return GetPatientsByDiagnosisIR(diagnosis_code_prefix=diagnosis_code_prefix)

    days = _extract_recent_admission_days(normalized)
    if days is not None:
        return GetRecentlyAdmittedIR(days=days)

    return UnknownIR(reason="Intento non supportato dal parser deterministico Phase 1.")


def _normalize(query: str) -> str:
    text = unicodedata.normalize("NFKD", query.strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


def _extract_patient_id(text: str) -> str | None:
    match = re.search(r"\b(?:paziente|patient)\s+(\d+)\b", text)
    return match.group(1) if match else None


def _looks_like_patient_status(text: str) -> bool:
    if re.search(r"\b(?:summary|riassunto|sintesi)\b", text):
        return False
    return bool(
        re.search(r"\b(?:come sta|stato|status)\b", text)
        and re.search(r"\b(?:paziente|patient)\b", text)
    )


def _looks_like_patient_summary(text: str) -> bool:
    return bool(
        re.search(r"\b(?:summary|riassunto|sintesi)\b", text)
        and re.search(r"\b(?:paziente|patient)\b", text)
    )


def _extract_diagnosis_code_prefix(text: str) -> str | None:
    match = re.search(
        r"\b(?:pazienti|patients)\s+(?:con|with)\s+diagnos(?:i|is)\s+(\d+)\b",
        text,
    )
    return match.group(1) if match else None


def _extract_recent_admission_days(text: str) -> int | None:
    patterns = [
        r"\bricoverati\s+(?:negli|nelle|nei)\s+ultim[ie]\s+(\d+)\s+giorni\b",
        r"\brecently\s+admitted\s+(?:in\s+the\s+)?last\s+(\d+)\s+days\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None
