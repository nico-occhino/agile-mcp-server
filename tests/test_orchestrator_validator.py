from orchestrator.ir_schema import (
    GetPatientStatusIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)
from orchestrator.validator import validate_ir


def test_validate_patient_id_must_be_numeric():
    issues = validate_ir(GetPatientStatusIR(patient_id="abc"))
    assert issues == ["patient_id deve essere numerico."]


def test_validate_diagnosis_code_prefix_must_be_numeric():
    issues = validate_ir(GetPatientsByDiagnosisIR(diagnosis_code_prefix="Z99"))
    assert issues == ["diagnosis_code_prefix deve essere numerico."]


def test_validate_days_must_be_positive():
    issues = validate_ir(GetRecentlyAdmittedIR(days=0))
    assert issues == ["days deve essere positivo."]


def test_validate_days_must_not_exceed_one_year():
    issues = validate_ir(GetRecentlyAdmittedIR(days=366))
    assert issues == ["days deve essere minore o uguale a 365."]


def test_validate_unknown_returns_reason():
    issues = validate_ir(UnknownIR(reason="Intento non supportato."))
    assert issues == ["Intento non supportato."]
