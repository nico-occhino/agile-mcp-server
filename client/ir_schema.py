"""
client/ir_schema.py
-------------------
Pydantic discriminated-union models for every intent the client can express.

Each class maps 1-to-1 to one MCP tool on the server. The `intent` field
is a Literal discriminator so that TypeAdapter can parse raw JSON unambiguously.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


# ---------------------------------------------------------------------------
# Individual IR classes — one per server tool
# ---------------------------------------------------------------------------

class GetPatientAgeIR(BaseModel):
    intent: Literal["get_patient_age"] = "get_patient_age"
    patient_id: str


class GetPatientStatusIR(BaseModel):
    intent: Literal["get_patient_status"] = "get_patient_status"
    patient_id: str


class GetPatientSummaryIR(BaseModel):
    intent: Literal["get_patient_summary"] = "get_patient_summary"
    patient_id: str


class GetPatientDischargeDraftIR(BaseModel):
    intent: Literal["get_patient_discharge_draft"] = "get_patient_discharge_draft"
    patient_id: str


class GetAdmissionHistoryIR(BaseModel):
    intent: Literal["get_admission_history"] = "get_admission_history"
    patient_id: str


class GetPatientsByDiagnosisIR(BaseModel):
    intent: Literal["get_patients_by_diagnosis"] = "get_patients_by_diagnosis"
    diagnosis_code_prefix: str


class GetCohortSummaryIR(BaseModel):
    intent: Literal["get_cohort_summary"] = "get_cohort_summary"
    diagnosis_code_prefix: str


class GetRecentlyAdmittedIR(BaseModel):
    intent: Literal["get_recently_admitted"] = "get_recently_admitted"
    days: int = 7


class UnknownIR(BaseModel):
    intent: Literal["unknown"] = "unknown"
    raw_query: str
    reason: str


# ---------------------------------------------------------------------------
# Discriminated union — the `intent` field is the discriminator tag
# ---------------------------------------------------------------------------

IR = Annotated[
    Union[
        GetPatientAgeIR,
        GetPatientStatusIR,
        GetPatientSummaryIR,
        GetPatientDischargeDraftIR,
        GetAdmissionHistoryIR,
        GetPatientsByDiagnosisIR,
        GetCohortSummaryIR,
        GetRecentlyAdmittedIR,
        UnknownIR,
    ],
    Field(discriminator="intent"),
]

# Pre-built TypeAdapter for JSON parsing — import and reuse this in parser.py
_adapter: TypeAdapter[IR] = TypeAdapter(IR)
