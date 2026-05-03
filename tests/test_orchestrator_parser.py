from orchestrator.ir_schema import (
    GetPatientStatusIR,
    GetPatientSummaryIR,
    GetPatientsByDiagnosisIR,
    GetRecentlyAdmittedIR,
    UnknownIR,
)
from orchestrator.parser import parse_query_to_ir


def test_parse_patient_status_italian_question():
    ir = parse_query_to_ir("Come sta il paziente 45?")
    assert isinstance(ir, GetPatientStatusIR)
    assert ir.patient_id == "45"


def test_parse_patient_summary_italian():
    ir = parse_query_to_ir("riassunto paziente 45")
    assert isinstance(ir, GetPatientSummaryIR)
    assert ir.patient_id == "45"


def test_parse_patients_by_diagnosis_italian():
    ir = parse_query_to_ir("pazienti con diagnosi 428")
    assert isinstance(ir, GetPatientsByDiagnosisIR)
    assert ir.diagnosis_code_prefix == "428"


def test_parse_recently_admitted_italian():
    ir = parse_query_to_ir("ricoverati negli ultimi 7 giorni")
    assert isinstance(ir, GetRecentlyAdmittedIR)
    assert ir.days == 7


def test_parse_missing_patient_id_returns_unknown():
    ir = parse_query_to_ir("Come sta il paziente?")
    assert isinstance(ir, UnknownIR)
    assert "ID numerico" in ir.reason


def test_parse_adversarial_query_returns_unknown():
    ir = parse_query_to_ir("drop table patients")
    assert isinstance(ir, UnknownIR)
