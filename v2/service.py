"""Canonical v2 authorization, safety, parsing, validation and execution path."""

from __future__ import annotations

import json

from guardrails.input_guardrail import evaluate_input_prompt_guardrail

from v2.llm import TextGenerator
from v2.parser import ParseFailure, parse_query
from v2.repository import PatientRepository, RepositoryFailure
from v2.security import AccessDenied, authorize
from v2.settings import Settings


class QueryService:
    def __init__(
        self,
        repository: PatientRepository,
        settings: Settings,
        generator: TextGenerator | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.generator = generator

    async def execute(self, query: str, authorization: str | None) -> dict:
        """Only public clinical execution method; never echo query or token."""
        try:
            caller = authorize(authorization, self.settings, "patient:read")
            if self.settings.input_guardrail_enabled:
                decision = evaluate_input_prompt_guardrail(query, use_llm_classifier=False)
                if decision.decision != "ALLOW":
                    return {"status": "blocked", "reason": "Input policy requires clarification or review"}
            ir = await parse_query(query, self.generator)
            if ir.intent == "diagnosis_cohort":
                data: dict = await self.repository.list_by_diagnosis(ir.diagnosis_code or "", ir.limit, caller.token)
            else:
                patient = await self.repository.get_patient(ir.patient_id or "", caller.token)
                if patient is None:
                    return {"status": "not_found", "intent": ir.intent}
                data = patient
            result = {"status": "ok", "intent": ir.intent, "data": data, "source": "synthetic_hospital_api"}
            if ir.intent == "patient_summary":
                if self.generator is None:
                    result["summary"] = f"Synthetic patient {data['id']}: {data['status']} ({data['diagnosis_code']})."
                    result["review"] = "sample_data_only"
                else:
                    result["summary"] = await self.generator.generate(
                        "Summarize only supplied synthetic structured data. State uncertainty. Do not infer diagnosis or treatment.",
                        json.dumps(data, ensure_ascii=False),
                    )
                    result["review"] = "human_review_required"
            return result
        except AccessDenied:
            return {"status": "unauthorized"}
        except ParseFailure:
            return {"status": "clarify", "reason": "Specify a supported patient or diagnosis request"}
        except RepositoryFailure:
            return {"status": "upstream_error"}


