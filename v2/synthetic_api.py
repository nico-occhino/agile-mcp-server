"""Independent synthetic Italian hospital REST service. No real patient data."""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from v2.security import AccessDenied, authorize
from v2.settings import Settings


PATIENTS = [
    {"id": "101", "name": "Paziente Sintetico A", "status": "ricoverato", "diagnosis_code": "250.00", "ward": "Medicina"},
    {"id": "102", "name": "Paziente Sintetico B", "status": "dimesso", "diagnosis_code": "401.9", "ward": "Cardiologia"},
    {"id": "103", "name": "Paziente Sintetico C", "status": "ricoverato", "diagnosis_code": "250.00", "ward": "Medicina"},
    {"id": "104", "name": "Paziente Sintetico D", "status": "dimesso", "diagnosis_code": "486", "ward": "Pneumologia"},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    app = FastAPI(title="Synthetic hospital API", version="2.0")

    def require_read(authorization: str | None = Header(default=None)) -> None:
        try:
            authorize(authorization, config, "patient:read")
        except AccessDenied as exc:
            raise HTTPException(status_code=401 if "token" in str(exc).lower() else 403, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "data": "synthetic"}

    @app.get("/patients", dependencies=[Depends(require_read)])
    def list_patients(
        diagnosis_code: str | None = Query(default=None, pattern=r"^[0-9]{3}(?:\.[0-9]{1,2})?$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        matched = [p for p in PATIENTS if diagnosis_code is None or p["diagnosis_code"] == diagnosis_code]
        return {"items": matched[offset : offset + limit], "total": len(matched), "offset": offset, "limit": limit}

    @app.get("/patients/{patient_id}", dependencies=[Depends(require_read)])
    def get_patient(patient_id: str) -> dict:
        patient = next((p for p in PATIENTS if p["id"] == patient_id), None)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8081)

