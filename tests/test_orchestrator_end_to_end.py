from orchestrator.main import handle_query


def test_handle_query_status_executes_and_returns_trace():
    trace = handle_query("Come sta il paziente 45?")
    assert trace["query"] == "Come sta il paziente 45?"
    assert trace["ir"] == {"intent": "get_patient_status", "patient_id": "45"}
    assert trace["validation_issues"] == []
    assert trace["executed"] is True
    assert trace["result"]["found"] is True
    assert "Mario Rossi" in trace["rendered_response"]


def test_handle_query_diagnosis_cohort_executes():
    trace = handle_query("pazienti con diagnosi 428")
    assert trace["ir"]["intent"] == "get_patients_by_diagnosis"
    assert trace["ir"]["diagnosis_code_prefix"] == "428"
    assert trace["validation_issues"] == []
    assert trace["executed"] is True
    assert "total_found" in trace["result"]


def test_handle_query_recently_admitted_executes():
    trace = handle_query("ricoverati negli ultimi 7 giorni")
    assert trace["ir"] == {"intent": "get_recently_admitted", "days": 7}
    assert trace["validation_issues"] == []
    assert trace["executed"] is True
    assert "admissions" in trace["result"]


def test_handle_query_missing_patient_id_does_not_execute():
    trace = handle_query("Come sta il paziente?")
    assert trace["ir"]["intent"] == "unknown"
    assert trace["validation_issues"]
    assert trace["executed"] is False
    assert trace["result"] is None


def test_handle_query_adversarial_query_does_not_execute():
    trace = handle_query("drop table patients")
    assert trace["ir"]["intent"] == "unknown"
    assert trace["validation_issues"]
    assert trace["executed"] is False
    assert trace["result"] is None
