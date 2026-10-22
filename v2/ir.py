"""Typed, bounded intermediate representation for supported read-only intents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QueryIR(BaseModel):
    intent: Literal["patient_status", "patient_summary", "diagnosis_cohort"]
    patient_id: str | None = Field(default=None, pattern=r"^[0-9]{1,8}$")
    diagnosis_code: str | None = Field(default=None, pattern=r"^[0-9]{3}(?:\.[0-9]{1,2})?$")
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def required_argument(self) -> QueryIR:
        if self.intent in {"patient_status", "patient_summary"} and not self.patient_id:
            raise ValueError("patient_id required")
        if self.intent == "diagnosis_cohort" and not self.diagnosis_code:
            raise ValueError("diagnosis_code required")
        return self

