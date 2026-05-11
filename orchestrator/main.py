"""Top-level NL2API orchestrator pipeline."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from guardrails.input_guardrail import (
    InputGuardrailDecision,
    evaluate_input_prompt_guardrail,
)
from orchestrator.parser import parse_query_to_ir
from orchestrator.renderer import render_response
from orchestrator.router import route_and_execute
from orchestrator.validator import validate_ir


def handle_query(query: str) -> dict[str, Any]:
    """Parse, validate, route, execute, and render a natural-language query."""
    if _input_guardrail_enabled():
        input_guardrail = evaluate_input_prompt_guardrail(
            query,
            use_llm_classifier=False,
        )
        if input_guardrail.decision != InputGuardrailDecision.ALLOW.value:
            return {
                "query": query,
                "ir": None,
                "validation_issues": input_guardrail.reasons,
                "executed": False,
                "result": None,
                "rendered_response": "Richiesta bloccata dal guardrail input: "
                + "; ".join(input_guardrail.reasons),
                "input_guardrail": input_guardrail.model_dump(),
            }

    ir = parse_query_to_ir(query)
    validation_issues = validate_ir(ir)

    executed = False
    result: dict | None = None

    if validation_issues:
        rendered_response = "Richiesta bloccata: " + "; ".join(validation_issues)
    else:
        result = route_and_execute(ir)
        executed = True
        rendered_response = render_response(ir, result)

    trace = {
        "query": query,
        "ir": _model_to_dict(ir),
        "validation_issues": validation_issues,
        "executed": executed,
        "result": result,
        "rendered_response": rendered_response,
    }
    if _input_guardrail_enabled():
        trace["input_guardrail"] = input_guardrail.model_dump()
    return trace


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()


def _input_guardrail_enabled() -> bool:
    return os.getenv("INPUT_GUARDRAIL_ENABLED", "false").lower() == "true"
