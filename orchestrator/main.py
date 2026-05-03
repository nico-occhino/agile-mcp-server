"""Top-level NL2API orchestrator pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from orchestrator.parser import parse_query_to_ir
from orchestrator.renderer import render_response
from orchestrator.router import route_and_execute
from orchestrator.validator import validate_ir


def handle_query(query: str) -> dict[str, Any]:
    """Parse, validate, route, execute, and render a natural-language query."""
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

    return {
        "query": query,
        "ir": _model_to_dict(ir),
        "validation_issues": validation_issues,
        "executed": executed,
        "result": result,
        "rendered_response": rendered_response,
    }


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()
