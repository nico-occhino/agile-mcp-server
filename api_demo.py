"""Optional FastAPI/Swagger wrapper for demos and debugging.

MCP remains the primary integration interface for Agile Aria. This module only
exposes a few HTTP endpoints so trainees and stakeholders can inspect behavior
through Swagger during presentations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI

from features.patient_lookup import get_patient_status
from guardrails.decision import evaluate_guardrail
from orchestrator.main import handle_query


app = FastAPI(
    title="agile-mcp-server demo",
    description="Swagger demo wrapper. MCP remains the target Aria integration.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., examples=["Come sta il paziente 45?"])


class GuardrailRequest(BaseModel):
    task_type: str = Field(..., examples=["patient_summary"])
    confidence: float | None = Field(default=None, examples=[0.72])
    validation_issues: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "agile-mcp-server-demo"}


@app.post("/nl2api/query")
def nl2api_query(request: QueryRequest) -> dict:
    return handle_query(request.query)


@app.post("/guardrails/evaluate")
def guardrails_evaluate(request: GuardrailRequest) -> dict:
    return evaluate_guardrail(
        task_type=request.task_type,
        confidence=request.confidence,
        validation_issues=request.validation_issues,
    ).model_dump()


@app.get("/patients/{patient_id}/status")
def patient_status(patient_id: str) -> dict:
    return get_patient_status(patient_id)
