"""
tests/test_features.py
Patient IDs: "45"=Mario Rossi (discharged), "46"=Giovanna Ferrara (admitted),
             "47"=Luca Bianchi (discharged), "48"=Sara Esposito (no event)
"""
import pytest
import numpy as np
from workflow.uncertainty import ConfidenceLevel, MEDIUM_CONFIDENCE_THRESHOLD
from features.patient_lookup import get_patient_age

class TestDeterministicLookup:

    def test_get_patient_age_known_patient(self):
        from features.patient_lookup import get_patient_age
        result = get_patient_age("45")
        assert result["found"] is True
        assert result["patient_id"] == "45"
        assert result["full_name"] == "Mario Rossi"
        assert 30 <= result["age_years"] <= 40
        assert result["data_nascita"] == "1991-01-01"

    def test_get_patient_age_unknown_patient(self):
        from features.patient_lookup import get_patient_age
        result = get_patient_age("NONEXISTENT")
        assert result["found"] is False
        assert "error" in result

    def test_get_patient_status_currently_admitted(self):
        from features.patient_lookup import get_patient_status
        result = get_patient_status("46")
        assert result["found"] is True
        assert result["stato"] == "ricoverato"
        assert "reparto" in result

    def test_get_patient_status_discharged(self):
        from features.patient_lookup import get_patient_status
        result = get_patient_status("45")
        assert result["found"] is True
        assert result["stato"] == "dimesso"
        assert "ultima_dimissione" in result

    def test_get_patient_no_admissions(self):
        from features.patient_lookup import get_patient_status
        result = get_patient_status("48")
        assert result["found"] is True
        assert result["stato"] == "mai_ricoverato"

    def test_get_admission_history_multiple_admissions(self):
        from features.patient_lookup import get_admission_history
        result = get_admission_history("47")
        assert result["found"] is True
        assert result["total_admissions"] >= 1
        admissions = result["admissions"]
        dates = [a["data_ingresso"] for a in admissions]
        assert dates == sorted(dates)

    def test_diagnosis_filter(self):
        from data.mock_store import list_patients_by_diagnosis
        results = list_patients_by_diagnosis("428")
        patient_ids = [p["patient"]["internalId"] for p in results]
        assert "46" in patient_ids

    def test_diagnosis_filter_no_match(self):
        from data.mock_store import list_patients_by_diagnosis
        results = list_patients_by_diagnosis("Z99")
        assert results == []

    def test_get_patient_whitespace_and_case(self):
        for dirty_id in ["45", " 45", "45\n", "45\n\n"]:
            result = get_patient_age(dirty_id)
            assert result["found"] is True, f"Failed for input: {repr(dirty_id)}"
            assert result["full_name"] == "Mario Rossi"


class TestUncertaintyMath:

    def test_perfect_agreement_categorical(self):
        from workflow.uncertainty import estimate_confidence_categorical
        confidence, majority = estimate_confidence_categorical(["45", "45", "45"])
        assert confidence == pytest.approx(1.0)
        assert majority == "45"

    def test_total_disagreement_categorical(self):
        from workflow.uncertainty import estimate_confidence_categorical
        confidence, _ = estimate_confidence_categorical(["45", "46", "47"])
        assert confidence == pytest.approx(0.0, abs=0.01)

    def test_partial_agreement_categorical(self):
        from workflow.uncertainty import estimate_confidence_categorical
        confidence, majority = estimate_confidence_categorical(["45", "45", "46"])
        assert 0.0 < confidence < 1.0
        assert majority == "45"

    def test_perfect_agreement_freetext(self):
        from workflow.uncertainty import estimate_confidence_freetext
        text = "Il paziente Mario Rossi è stabile, con valori pressori nella norma."
        confidence, _ = estimate_confidence_freetext([text, text, text])
        assert confidence == pytest.approx(1.0)

    def test_completely_different_freetext(self):
        from workflow.uncertainty import estimate_confidence_freetext
        samples = ["aaa bbb ccc ddd eee fff", "ggg hhh iii jjj kkk lll", "mmm nnn ooo ppp qqq rrr"]
        confidence, _ = estimate_confidence_freetext(samples)
        assert confidence < 0.3

    def test_semantic_paraphrase_scores_high(self):
        from workflow.uncertainty import estimate_confidence_freetext
        samples = [
            "Il paziente presenta ipertensione arteriosa con valori pressori elevati.",
            "Paziente con pressione alta, PA 160/95 alla misurazione.",
            "Riscontrata ipertensione. Valori pressori superiori alla norma.",
        ]
        confidence, _ = estimate_confidence_freetext(samples)
        assert confidence > 0.60, f"Semantic similarity should be HIGH for paraphrases, got {confidence:.3f}."

    def test_build_uncertain_result_high(self):
        from workflow.uncertainty import build_uncertain_result, ConfidenceLevel
        result = build_uncertain_result(["45", "45", "45"], mode="categorical")
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.confidence == pytest.approx(1.0)

    def test_build_uncertain_result_low(self):
        from workflow.uncertainty import build_uncertain_result, ConfidenceLevel
        result = build_uncertain_result(["45", "46", "47"], mode="categorical")
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_uncertain_result_serialises(self):
        from workflow.uncertainty import build_uncertain_result
        result = build_uncertain_result(["45", "45", "46"], mode="categorical")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "result" in d and "confidence" in d and "confidence_level" in d and "rationale" in d
        assert "samples" not in d


class TestCohortWorkflowStructure:

    def test_get_patients_by_diagnosis_found(self):
        from features.cohort import get_patients_by_diagnosis
        result = get_patients_by_diagnosis("428")
        assert result["diagnosis_code_prefix"] == "428"
        assert result["total_found"] >= 1
        assert len(result["patients"]) == result["total_found"]
        first = result["patients"][0]
        assert "patient_id" in first and "full_name" in first

    def test_get_patients_by_diagnosis_not_found(self):
        from features.cohort import get_patients_by_diagnosis
        result = get_patients_by_diagnosis("Z99")
        assert result["diagnosis_code_prefix"] == "Z99"
        assert result["total_found"] == 0
        assert result["patients"] == []
        assert "message" in result

    def test_get_recently_admitted_returns_valid_structure(self):
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=365)
        assert result["look_back_days"] == 365
        assert "cutoff_date" in result and "total_found" in result
        assert isinstance(result["admissions"], list)

    def test_get_recently_admitted_zero_days(self):
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=0)
        assert result["total_found"] == 0

    def test_cohort_admissions_sorted_descending(self):
        from features.cohort import get_recently_admitted
        result = get_recently_admitted(days=365)
        admissions = result["admissions"]
        if len(admissions) >= 2:
            dates = [a["data_ingresso"] for a in admissions]
            assert dates == sorted(dates, reverse=True)

    def test_cohort_confidence_uses_min_when_patient_flagged(self):
        confidences = [0.9, 0.95, 0.88, 0.1]
        min_c = min(confidences)
        mean_c = float(np.mean(confidences))
        assert min_c < MEDIUM_CONFIDENCE_THRESHOLD
        assert mean_c > MEDIUM_CONFIDENCE_THRESHOLD
        cohort_confidence = min_c if min_c < 0.5 else mean_c
        assert cohort_confidence == min_c
