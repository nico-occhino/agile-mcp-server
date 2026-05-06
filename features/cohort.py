"""
features/cohort.py
------------------
Multi-step workflow composition for population-level queries.
Schema: Nocita's real API shape. Diagnosis codes are Italian ministerial
numeric strings ("428", "4349"), not ICD-10.

COHORT CONFIDENCE LOGIC
-----------------------
If ANY patient confidence < 0.5 (LOW), the cohort confidence is set to
the minimum — not the mean. A single failed summary cannot be hidden by
high scores on other patients. Both mean and min are always returned.
"""
from __future__ import annotations
from datetime import date, timedelta
from data.repository import get_repository

def _get_repo():
    return get_repository()
from features.patient_lookup import get_patient_status
from features.patient_summary import get_patient_summary
from workflow.llm_client import call_llm
import numpy as np

SAFETY_THRESHOLD = 0.50


def get_patients_by_diagnosis(diagnosis_code_prefix: str) -> dict:
    """
    Return all patients whose primary diagnosis code starts with the given prefix.

    Args:
        diagnosis_code_prefix: numeric diagnosis-code prefix (e.g. "428" for heart failure, "434" for stroke)
    """
    matching = _get_repo().list_patients_by_diagnosis(diagnosis_code_prefix)
    if not matching:
        return {"diagnosis_code_prefix": diagnosis_code_prefix, "total_found": 0, "patients": [], "message": f"No patients found with diagnosis code starting with '{diagnosis_code_prefix}'."}
    patients_detail = []
    for p in matching:
        patient = p.get("patient")
        if not patient:
            continue
        status = get_patient_status(patient["internalId"])
        patients_detail.append({"patient_id": patient["internalId"], "full_name": f"{patient['name']} {patient['surname']}", "data_nascita": patient["birthDate"][:10], **{k: v for k, v in status.items() if k not in ("found", "patient_id")}})
    return {"diagnosis_code_prefix": diagnosis_code_prefix, "total_found": len(patients_detail), "patients": patients_detail}


def get_cohort_summary(diagnosis_code_prefix: str) -> dict:
    """
    Generate a cohort-level summary with per-patient LLM summaries
    and a gated aggregate confidence score.

    Args:
        diagnosis_code_prefix: numeric diagnosis-code prefix (e.g. "428", "434")
    """
    matching = _get_repo().list_patients_by_diagnosis(diagnosis_code_prefix)
    if not matching:
        return {"diagnosis_code_prefix": diagnosis_code_prefix, "total_found": 0, "cohort_narrative": f"No patients found with prefix '{diagnosis_code_prefix}'.", "patients": [], "cohort_confidence": None, "flagged_patients": []}

    per_patient = []
    flagged = []
    for p in matching:
        patient = p.get("patient")
        if not patient:
            continue
        summary_result = get_patient_summary(patient["internalId"])
        confidence = summary_result.get("confidence", 0.0)
        level = summary_result.get("confidence_level", "LOW")
        if level == "LOW":
            flagged.append(patient["internalId"])
        per_patient.append({"patient_id": patient["internalId"], "full_name": f"{patient['name']} {patient['surname']}", "summary": summary_result.get("result", "Summary unavailable."), "confidence": confidence, "confidence_level": level})

    confidences = [pp["confidence"] for pp in per_patient]
    cohort_confidence_mean = float(np.mean(confidences)) if confidences else 0.0
    cohort_confidence_min  = float(np.min(confidences))  if confidences else 0.0

    # summaries_block must be defined BEFORE the if/else — it's needed regardless
    # of whether confidence is high or low (the narrative step always runs)
    summaries_block = "\n\n".join(
        f"[{pp['patient_id']}] {pp['full_name']}:\n{pp['summary']}"
        for pp in per_patient
    )

    if cohort_confidence_min < SAFETY_THRESHOLD:
        cohort_confidence = cohort_confidence_min
        cohort_confidence_note = (
            f"Cohort confidence degraded to minimum ({cohort_confidence_min:.0%}) "
            f"because {len(flagged)} patient(s) had LOW individual confidence. "
            f"Mean would have been {cohort_confidence_mean:.0%} — do not use mean in clinical context."
        )
    else:
        cohort_confidence = cohort_confidence_mean
        cohort_confidence_note = (
            f"All patients above safety threshold. "
            f"Mean: {cohort_confidence_mean:.0%}, min: {cohort_confidence_min:.0%}."
        )

    cohort_narrative = call_llm(
        system=("You are a clinical informatics assistant. Write a single cohesive paragraph "
                "(4–6 sentences) characterizing this patient cohort: common patterns, notable "
                "variations, and concerns a department head should be aware of. "
                "Do not list patients by name. Focus on patterns."),
        user=(f"Diagnosis code prefix: {diagnosis_code_prefix}\nPatients: {len(per_patient)}\n\n"
              f"Per-patient summaries:\n{summaries_block}\n\nWrite the cohort narrative."),
        temperature=0.4,
    )

    return {"diagnosis_code_prefix": diagnosis_code_prefix, "total_found": len(per_patient), "cohort_confidence": round(cohort_confidence, 3), "cohort_confidence_mean": round(cohort_confidence_mean, 3), "cohort_confidence_min": round(cohort_confidence_min, 3), "cohort_confidence_note": cohort_confidence_note, "flagged_patients": flagged, "cohort_narrative": cohort_narrative, "patients": per_patient}


def get_recently_admitted(days: int = 7) -> dict:
    """
    Return all patients admitted within the last N days.

    Args:
        days: look-back window in days (default: 7)
    """
    cutoff = date.today() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()
    all_patients = _get_repo().list_patients()
    recent = []
    for p_summary in all_patients:
        record = _get_repo().get_patient(p_summary["internalId"])
        if not record:
            continue
        patient = record.get("patient")
        event = record.get("event")
        if patient and event and event["dateStart"][:10] >= cutoff_str:
            recent.append({"patient_id": patient["internalId"], "full_name": f"{patient['name']} {patient['surname']}", "reparto": event["uoDescription"], "data_ingresso": event["dateStart"][:10], "diagnosi_principale": event.get("diagnosis", {}).get("primary"), "stato": "ricoverato" if event.get("dateEnd") is None else "dimesso"})
    return {"look_back_days": days, "cutoff_date": cutoff_str, "total_found": len(recent), "admissions": sorted(recent, key=lambda x: x["data_ingresso"], reverse=True)}
