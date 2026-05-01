"""
tests/test_client.py
--------------------
Unit tests for the client layer.
No LLM calls. No MCP server required.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from client.ir_schema import (
    IR,
    GetPatientAgeIR,
    GetPatientStatusIR,
    GetCohortSummaryIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)
from client.router import route
from client.main import render


# ---------------------------------------------------------------------------
# TestIRRouter
# ---------------------------------------------------------------------------

class TestIRRouter:
    def test_route_patient_age(self):
        r = route(GetPatientAgeIR(intent="get_patient_age", patient_id="45"))
        assert r == ("get_patient_age", {"patient_id": "45"})

    def test_route_cohort_summary(self):
        r = route(GetCohortSummaryIR(intent="get_cohort_summary", diagnosis_prefix="428"))
        assert r == ("get_cohort_summary", {"icd10_prefix": "428"})

    def test_route_recently_admitted_default_days(self):
        r = route(GetRecentlyAdmittedIR(intent="get_recently_admitted"))
        assert r is not None
        assert r[1]["days"] == 7

    def test_route_unknown_returns_none(self):
        r = route(UnknownIR(intent="unknown", raw_query="?", reason="test"))
        assert r is None


# ---------------------------------------------------------------------------
# TestIRSchema
# ---------------------------------------------------------------------------

class TestIRSchema:
    @pytest.fixture(autouse=True)
    def _adapter(self):
        self.adapter = TypeAdapter(IR)

    def test_ir_adapter_parses_patient_status(self):
        ir = self.adapter.validate_json('{"intent":"get_patient_status","patient_id":"46"}')
        assert isinstance(ir, GetPatientStatusIR)
        assert ir.patient_id == "46"

    def test_ir_adapter_parses_recently_admitted_default(self):
        ir = self.adapter.validate_json('{"intent":"get_recently_admitted"}')
        assert isinstance(ir, GetRecentlyAdmittedIR)
        assert ir.days == 7

    def test_ir_adapter_parses_unknown(self):
        ir = self.adapter.validate_json('{"intent":"unknown","raw_query":"x","reason":"y"}')
        assert isinstance(ir, UnknownIR)


# ---------------------------------------------------------------------------
# TestRenderer
# ---------------------------------------------------------------------------

class TestRenderer:
    def test_render_low_confidence_returns_refusal(self):
        result = {
            "confidence": 0.3,
            "confidence_level": "LOW",
            "rationale": "samples disagreed",
            "result": "something",
        }
        output = render("get_patient_summary", result)
        assert "⚠️" in output
        assert "CONFIDENZA BASSA" in output
        # The LLM result must NOT be shown when confidence is LOW
        assert "something" not in output

    def test_render_surfaces_allergy_first(self):
        result = {
            "found": True,
            "full_name": "Mario Rossi",
            "stato": "dimesso",
            "allergy": "Arachidi",
            "confidence": 0.9,
            "confidence_level": "HIGH",
        }
        output = render("get_patient_status", result)
        lines = output.strip().split("\n")
        assert "🚨" in lines[0]
        assert "Arachidi" in lines[0]

    def test_render_high_confidence_llm_tool(self):
        result = {
            "result": "Paziente stabile.",
            "confidence": 0.85,
            "confidence_level": "HIGH",
            "rationale": "consistent",
        }
        output = render("get_patient_summary", result)
        assert "Paziente stabile." in output
        assert "✅" in output
