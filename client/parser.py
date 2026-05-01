"""
client/parser.py
----------------
parse(query: str) -> IR

Converts a free-text clinical query into a structured IR object using an LLM
at temperature=0 (deterministic extraction). Falls back to UnknownIR on any
parsing or validation failure.
"""

from __future__ import annotations

import json

from client.ir_schema import IR, UnknownIR, _adapter
from workflow.llm_client import call_llm

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Sei un classificatore di intenti clinici. Analizza la query dell'utente e \
restituisci SOLO un oggetto JSON valido (nessun markdown, nessuna spiegazione).

Gli intenti validi sono:

1. get_patient_age
   Esempio: {"intent": "get_patient_age", "patient_id": "45"}

2. get_patient_status
   Esempio: {"intent": "get_patient_status", "patient_id": "46"}

3. get_patient_summary
   Esempio: {"intent": "get_patient_summary", "patient_id": "47"}

4. get_patient_discharge_draft
   Esempio: {"intent": "get_patient_discharge_draft", "patient_id": "48"}

5. get_admission_history
   Esempio: {"intent": "get_admission_history", "patient_id": "45"}

6. get_patients_by_diagnosis
   Esempio: {"intent": "get_patients_by_diagnosis", "diagnosis_prefix": "428"}

7. get_cohort_summary
   Esempio: {"intent": "get_cohort_summary", "diagnosis_prefix": "434"}

8. get_recently_admitted
   Esempio: {"intent": "get_recently_admitted", "days": 7}

REGOLE IMPORTANTI:
- I patient_id sono stringhe numeriche: "45", "46", "47", "48", ecc.
- I diagnosis_prefix sono codici ICD-9-CM numerici ministeriali italiani:
    428 = scompenso cardiaco (heart failure)
    434 = ictus (stroke)
    012 = trauma ortopedico
- Se la query è ambigua, mancano informazioni obbligatorie, o non corrisponde \
a nessun intento, restituisci:
  {"intent": "unknown", "raw_query": "<query originale>", \
"reason": "<motivo breve in inglese>"}
- Restituisci SOLO JSON grezzo. Nessun blocco markdown, nessun testo \
aggiuntivo prima o dopo.
"""


def parse(query: str) -> IR:
    """
    Convert a free-text clinical query into a structured IR object.

    Uses the LLM at temperature=0 for deterministic extraction.
    On any failure, returns UnknownIR with the error reason.
    """
    raw = call_llm(system=_SYSTEM_PROMPT, user=query, temperature=0.0)

    # Strip accidental markdown fences (same pattern as llm_client.py)
    raw = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        return _adapter.validate_json(raw)
    except Exception as exc:
        return UnknownIR(
            intent="unknown",
            raw_query=query,
            reason=str(exc),
        )
