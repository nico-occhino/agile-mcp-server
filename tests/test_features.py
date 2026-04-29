"""
tests/test_features.py
-----------------------
Three tests, one per layer. These are the tripwire tests that Nocita and
Morana can run to confirm the POC is working. They also protect you during
refactoring — if a boundary breaks, a test fails immediately.

WHAT WE TEST AND WHY ONLY THIS
--------------------------------
  test_deterministic_lookup   — the data layer and pure-Python logic
  test_uncertainty_math       — the uncertainty module in isolation (no LLM)
  test_cohort_structure       — the multi-step workflow structure (no LLM)

We do NOT test the LLM calls directly here (test_patient_summary, etc.)
because:
  1. They require a real API key and cost money.
  2. They are non-deterministic — the same prompt can return different text.
  3. Testing prompts is a different discipline (evals) — see docs/eval_plan.md.

For LLM tests, write an eval script (scripts/eval.py) that runs the full
pipeline against a suite of known queries and checks output *properties*
("does the response mention the correct patient name?") not exact strings.
That is your thesis evaluation methodology.

HOW TO RUN
----------
    pip install pytest
    pytest tests/ -v
"""

import pytest
from datetime import date

# ---------------------------------------------------------------------------
# Layer 1: Deterministic data retrieval
# ---------------------------------------------------------------------------

class TestDeterministicLookup:

    def test_get_patient_age_known_patient(self):
        """Mario Rossi was born 1958-04-12. His age should be calculable."""
        from features.patient_lookup import get_patient_age
        result = get_patient_age("P001")

        assert result["found"] is True
        assert result["patient_id"] == "P001"
        assert result["full_name"] == "Mario Rossi"
        # Age should be a plausible integer (born 1958, so 67 or 68 in 2026)
        assert 60 <= result["age_years"] <= 80
        assert result["data_nascita"] == "1958-04-12"

    def test_get_patient_age_unknown_patient(self):
        """Requesting a non-existent patient should return found=False, not raise."""
        from features.patient_lookup import get_patient_age
        result = get_patient_age("NONEXISTENT")

        assert result["found"] is False
        assert "error" in result
        assert "NONEXISTENT" in result["error"]

    def test_get_patient_status_currently_admitted(self):
        """P002 has stato='ricoverato' — should surface active admission details."""
        from features.patient_lookup import get_patient_status
        result = get_patient_status("P002")

        assert result["found"] is True
        assert result["stato"] == "ricoverato"
        assert "reparto" in result
        assert "farmaci" in result
        assert isinstance(result["farmaci"], list)

    def test_get_patient_status_discharged(self):
        """P001 was discharged — should surface last discharge date."""
        from features.patient_lookup import get_patient_status
        result = get_patient_status("P001")

        assert result["found"] is True
        assert result["stato"] == "dimesso"
        assert "ultima_dimissione" in result

    def test_get_patient_no_admissions(self):
        """P004 has no ricoveri — should handle gracefully."""
        from features.patient_lookup import get_patient_status
        result = get_patient_status("P004")

        assert result["found"] is True
        assert result["stato"] == "mai_ricoverato"

    def test_get_admission_history_multiple_admissions(self):
        """P003 has two admissions — history should list both, ordered by date."""
        from features.patient_lookup import get_admission_history
        result = get_admission_history("P003")

        assert result["found"] is True
        assert result["total_admissions"] == 2
        admissions = result["admissions"]
        # Should be sorted by date ascending
        dates = [a["data_ingresso"] for a in admissions]
        assert dates == sorted(dates)

    def test_diagnosis_filter(self):
        """P002 has J44.1 (BPCO). Filtering by 'J44' should return P002."""
        from data.mock_store import list_patients_by_diagnosis
        results = list_patients_by_diagnosis("J44")

        patient_ids = [p["id"] for p in results]
        assert "P002" in patient_ids

    def test_diagnosis_filter_no_match(self):
        """No patient has Z99 codes. Should return empty list, not raise."""
        from data.mock_store import list_patients_by_diagnosis
        results = list_patients_by_diagnosis("Z99")
        assert results == []


# ---------------------------------------------------------------------------
# Layer 2: Uncertainty math (no LLM needed)
# ---------------------------------------------------------------------------

class TestUncertaintyMath:

    def test_perfect_agreement_categorical(self):
        """All N samples identical → confidence = 1.0."""
        from workflow.uncertainty import estimate_confidence_categorical
        samples = ["P001", "P001", "P001"]
        confidence, majority = estimate_confidence_categorical(samples)

        assert confidence == pytest.approx(1.0)
        assert majority == "p001"  # normalised to lowercase

    def test_total_disagreement_categorical(self):
        """All N samples different → confidence approaches 0.0."""
        from workflow.uncertainty import estimate_confidence_categorical
        samples = ["P001", "P002", "P003"]
        confidence, _ = estimate_confidence_categorical(samples)

        # With 3 equally likely outcomes: entropy = log(3), confidence = 0
        assert confidence == pytest.approx(0.0, abs=0.01)

    def test_partial_agreement_categorical(self):
        """2/3 samples agree → intermediate confidence."""
        from workflow.uncertainty import estimate_confidence_categorical
        samples = ["P001", "P001", "P002"]
        confidence, majority = estimate_confidence_categorical(samples)

        assert 0.0 < confidence < 1.0
        assert majority == "p001"

    def test_perfect_agreement_freetext(self):
        """Identical texts → Jaccard = 1.0 for all pairs → confidence = 1.0."""
        from workflow.uncertainty import estimate_confidence_freetext
        text = "Il paziente Mario Rossi è stabile, con valori pressori nella norma."
        samples = [text, text, text]
        confidence, _ = estimate_confidence_freetext(samples)

        assert confidence == pytest.approx(1.0)

    def test_completely_different_freetext(self):
        """Completely different texts → very low confidence."""
        from workflow.uncertainty import estimate_confidence_freetext
        samples = [
            "aaa bbb ccc ddd eee fff",
            "ggg hhh iii jjj kkk lll",
            "mmm nnn ooo ppp qqq rrr",
        ]
        confidence, _ = estimate_confidence_freetext(samples)

        assert confidence < 0.2

    def test_build_uncertain_result_high(self):
        """Perfect agreement → HIGH confidence_level."""
        from workflow.uncertainty import build_uncertain_result, ConfidenceLevel
        samples = ["P001", "P001", "P001"]
        result = build_uncertain_result(samples, mode="categorical")

        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.confidence == pytest.approx(1.0)

    def test_build_uncertain_result_low(self):
        """Total disagreement → LOW confidence_level."""
        from workflow.uncertainty import build_uncertain_result, ConfidenceLevel
        samples = ["P001", "P002", "P003"]
        result = build_uncertain_result(samples, mode="categorical")

        assert result.confidence_level == ConfidenceLevel.LOW

    def test_uncertain_result_serialises(self):
        """to_dict() should produce a plain dict without UncertainResult in it."""
        from workflow.uncertainty import build_uncertain_result
        samples = ["P001", "P001", "P002"]
        result = build_uncertain_result(samples, mode="categorical")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "result" in d
        assert "confidence" in d
        assert "confidence_level" in d
        assert "rationale" in d
        # samples should NOT be in the serialised form (too verbose for MCP response)
        assert "samples" not in d


# ---------------------------------------------------------------------------
# Layer 3: Multi-step workflow structure (no LLM)
# ---------------------------------------------------------------------------

class TestCohortWorkflowStructure:
    """
    These tests call the cohort tools against the mock store but do NOT
    make LLM calls. We verify:
      - the workflow runs without raising
      - the response shape is correct
      - edge cases (no matching patients) are handled

    The LLM-dependent cohort functions (get_cohort_summary) are NOT tested
    here — they belong in the eval suite.
    """

    def test_get_patients_by_diagnosis_found(self):
        """'J44' matches P002 — should return a non-empty list."""
        from features.cohort import get_patients_by_diagnosis
        result = get_patients_by_diagnosis("J44")

        assert result["total_found"] >= 1
        assert len(result["patients"]) == result["total_found"]
        # Each patient entry should have at minimum these keys
        first = result["patients"][0]
        assert "patient_id" in first
        assert "full_name" in first

    def test_get_patients_by_diagnosis_not_found(self):
        """Unknown prefix → total_found = 0, patients = []."""
        from features.cohort import get_patients_by_diagnosis
        result = get_patients_by_diagnosis("Z99")

        assert result["total_found"] == 0
        assert result["patients"] == []
        assert "message" in result

    def test_get_recently_admitted_returns_valid_structure(self):
        """Should return a dict with look_back_days, cutoff_date, admissions."""
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=365)  # wide window to catch our test data

        assert "look_back_days" in result
        assert result["look_back_days"] == 365
        assert "cutoff_date" in result
        assert "total_found" in result
        assert "admissions" in result
        assert isinstance(result["admissions"], list)

    def test_get_recently_admitted_zero_days(self):
        """A look-back of 0 days should return nothing (no future admissions)."""
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=0)

        assert result["total_found"] == 0

    def test_cohort_admissions_sorted_descending(self):
        """Admissions in the response should be sorted most-recent first."""
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=365)
        admissions = result["admissions"]

        if len(admissions) >= 2:
            dates = [a["data_ingresso"] for a in admissions]
            assert dates == sorted(dates, reverse=True)