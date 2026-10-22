"""Injectable async repository and Bearer-forwarding hospital HTTP adapter."""

from __future__ import annotations

from typing import Protocol

import httpx


class RepositoryFailure(Exception):
    """Upstream failure safe to expose without request data."""


class PatientRepository(Protocol):
    async def get_patient(self, patient_id: str, token: str) -> dict | None: ...
    async def list_by_diagnosis(self, diagnosis_code: str, limit: int, token: str) -> dict: ...


class HttpPatientRepository:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self.client = client
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str, token: str, params: dict | None = None) -> dict | None:
        try:
            response = await self.client.get(
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RepositoryFailure("Hospital API unavailable or rejected request") from exc

    async def get_patient(self, patient_id: str, token: str) -> dict | None:
        return await self._get(f"/patients/{patient_id}", token)

    async def list_by_diagnosis(self, diagnosis_code: str, limit: int, token: str) -> dict:
        result = await self._get("/patients", token, {"diagnosis_code": diagnosis_code, "limit": limit})
        if result is None:
            raise RepositoryFailure("Hospital API returned no cohort")
        return result

