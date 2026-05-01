"""
data/repository.py
------------------
Protocol and injection pattern for the patient data layer.
"""

from __future__ import annotations
import typing

class PatientRepository(typing.Protocol):
    def get_patient(self, patient_id: str) -> dict | None: ...
    def get_patient_demographics(self, patient_id: str) -> dict | None: ...
    def get_patient_event(self, patient_id: str) -> dict | None: ...
    def is_currently_admitted(self, patient_id: str) -> bool: ...
    def list_patients(self) -> list[dict]: ...
    def list_patients_by_diagnosis(self, diagnosis_prefix: str) -> list[dict]: ...
    def list_currently_admitted(self) -> list[dict]: ...
    def get_patient_allergy(self, patient_id: str) -> str | None: ...

_ACTIVE_REPO: PatientRepository | None = None

def get_repository() -> PatientRepository:
    """Return the active repository. Lazy-loads mock_store by default."""
    global _ACTIVE_REPO
    if _ACTIVE_REPO is None:
        from data import mock_store
        _ACTIVE_REPO = mock_store
    return _ACTIVE_REPO

def set_repository(repo: PatientRepository) -> None:
    """Override the active repository (used by tests and Phase 2 swap)."""
    global _ACTIVE_REPO
    _ACTIVE_REPO = repo
