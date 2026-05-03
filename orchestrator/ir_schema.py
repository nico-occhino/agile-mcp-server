"""
Pydantic Intermediate Representation (IR) for the NL2API orchestrator.

Each IR model represents one validated capability that can be routed to an
existing curated clinical tool. The union is discriminated by ``intent`` so the
IR remains explicit and auditable.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class GetPatientStatusIR(BaseModel):
    intent: Literal["get_patient_status"] = "get_patient_status"
    patient_id: str


class GetPatientSummaryIR(BaseModel):
    intent: Literal["get_patient_summary"] = "get_patient_summary"
    patient_id: str


class GetPatientsByDiagnosisIR(BaseModel):
    intent: Literal["get_patients_by_diagnosis"] = "get_patients_by_diagnosis"
    diagnosis_code_prefix: str


class GetRecentlyAdmittedIR(BaseModel):
    intent: Literal["get_recently_admitted"] = "get_recently_admitted"
    days: int = 7


class UnknownIR(BaseModel):
    intent: Literal["unknown"] = "unknown"
    reason: str


IR = Annotated[
    Union[
        GetPatientStatusIR,
        GetPatientSummaryIR,
        GetPatientsByDiagnosisIR,
        GetRecentlyAdmittedIR,
        UnknownIR,
    ],
    Field(discriminator="intent"),
]

IRAdapter: TypeAdapter[IR] = TypeAdapter(IR)
