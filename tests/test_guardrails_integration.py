from dataclasses import dataclass


@dataclass
class FakeUncertainResult:
    confidence: float

    def to_dict(self):
        return {
            "result": "Generated clinical text.",
            "confidence": self.confidence,
            "confidence_level": "HIGH",
            "rationale": "test double",
        }


def test_patient_summary_result_includes_guardrail(monkeypatch):
    from features import patient_summary

    monkeypatch.setattr(
        patient_summary,
        "call_llm_n_times",
        lambda **kwargs: ["Generated clinical text."],
    )
    monkeypatch.setattr(
        patient_summary,
        "build_uncertain_result",
        lambda samples, mode: FakeUncertainResult(confidence=0.90),
    )

    result = patient_summary.get_patient_summary("45")

    assert result["found"] is True
    _assert_guardrail_shape(result["guardrail"], task_type="patient_summary")
    assert result["guardrail"]["decision"] == "ACCEPT"


def test_discharge_draft_result_includes_guardrail(monkeypatch):
    from features import patient_summary

    monkeypatch.setattr(
        patient_summary,
        "call_llm_structured",
        lambda **kwargs: patient_summary.ClinicalFlags(
            primary_diagnosis_description="diagnosi test",
            active_conditions=[],
            key_medications=[],
            known_allergies=["Arachidi"],
            follow_up_required=False,
        ),
    )
    monkeypatch.setattr(
        patient_summary,
        "call_llm_n_times",
        lambda **kwargs: ["Generated discharge draft."],
    )
    monkeypatch.setattr(
        patient_summary,
        "build_uncertain_result",
        lambda samples, mode: FakeUncertainResult(confidence=0.90),
    )

    result = patient_summary.get_patient_discharge_draft("45")

    assert result["found"] is True
    _assert_guardrail_shape(result["guardrail"], task_type="discharge_draft")
    assert result["guardrail"]["decision"] == "ACCEPT"


def _assert_guardrail_shape(guardrail: dict, task_type: str) -> None:
    assert guardrail["task_type"] == task_type
    assert "risk_level" in guardrail
    assert "threshold" in guardrail
    assert "decision" in guardrail
    assert guardrail["reasons"]
