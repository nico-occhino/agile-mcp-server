"""Deterministic parser with an optional bounded LLM fallback."""

from __future__ import annotations

import json
import re

import httpx
from pydantic import ValidationError

from v2.ir import QueryIR
from v2.llm import TextGenerator


class ParseFailure(Exception):
    """Request needs clarification rather than execution."""


_PATIENT = re.compile(r"(?:paziente|patient)\s*(?:n\.?\s*)?([0-9]{1,8})", re.I)
_CODE = re.compile(r"(?:icd[ -]?9(?:-cm)?|diagnos[ie]|codice)\s*([0-9]{3}(?:\.[0-9]{1,2})?)", re.I)


async def parse_query(query: str, generator: TextGenerator | None = None) -> QueryIR:
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ParseFailure("Provide a shorter, specific request")
    patient = _PATIENT.search(query)
    code = _CODE.search(query)
    lower = query.lower()
    if patient:
        intent = "patient_summary" if any(word in lower for word in ("riassunto", "summary", "sintesi")) else "patient_status"
        return QueryIR(intent=intent, patient_id=patient.group(1))
    if code:
        return QueryIR(intent="diagnosis_cohort", diagnosis_code=code.group(1))
    if generator is None:
        raise ParseFailure("Specify a patient number or ICD-9-CM diagnosis code")
    try:
        raw = await generator.generate(
            "Return only JSON with intent, patient_id or diagnosis_code. Supported intents: patient_status, patient_summary, diagnosis_cohort. No other fields.",
            query,
        )
        payload = json.loads(raw)
        return QueryIR.model_validate(payload)
    except (ValueError, KeyError, TypeError, ValidationError, httpx.HTTPError) as exc:
        raise ParseFailure("Could not safely interpret request") from exc

