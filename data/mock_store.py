"""
data/mock_store.py
------------------
Synthetic patient records shaped after the SDO (Scheda di Dimissione Ospedaliera),
the standard Italian hospital discharge record.

WHY THIS EXISTS
---------------
Nocita will send us real SDO-shaped fixtures shortly. Until then we need *something*
that has the same structure so the feature code we write now won't need to change when
real data arrives. The domain vocabulary here (DRG, ICD-10 codes, reparto names) is
taken from the actual Italian ministerial SDO specification — it's fake *content*,
real *shape*.

HOW TO READ THE SCHEMA
-----------------------
Each patient has:
  - Demographic data (as in a real anagrafica)
  - A list of ricoveri (hospital admissions), each with:
      * reparto: the ward that admitted the patient
      * data_ingresso / data_dimissione: admission / discharge dates
      * diagnosi_principale: ICD-10 code for the primary diagnosis
      * diagnosi_secondarie: list of comorbidities (also ICD-10)
      * drg_code: Diagnosis Related Group — how the stay is billed to SSN
      * farmaci: list of medications administered during the stay
      * stato: "ricoverato" (currently admitted) | "dimesso" (discharged)

EXTENSION POINT
---------------
When Nocita's JSON arrives, replace this dict with a loader:
    import json, pathlib
    PATIENTS = json.loads(pathlib.Path("data/sdo_fixtures.json").read_text())
The feature code won't need to change because it accesses data through
get_patient() and list_patients(), not through PATIENTS directly.
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Raw data — mirrors the SDO structure Nocita described
# ---------------------------------------------------------------------------

PATIENTS: dict[str, dict] = {
    "P001": {
        "id": "P001",
        "nome": "Mario",
        "cognome": "Rossi",
        "data_nascita": "1958-04-12",
        "sesso": "M",
        "codice_fiscale": "RSSMRA58D12C351Z",
        "comune_residenza": "Catania",
        "ricoveri": [
            {
                "id_ricovero": "R001",
                "reparto": "Cardiologia",
                "data_ingresso": "2026-03-01",
                "data_dimissione": "2026-03-10",
                "diagnosi_principale": "I21.9",   # Infarto miocardico acuto, non specificato
                "diagnosi_secondarie": ["E11.9"],  # Diabete mellito tipo 2
                "drg_code": "121",                 # Malattie cardiovascolari con IMA
                "farmaci": ["Aspirina 100mg", "Metoprololo 50mg", "Atorvastatina 40mg"],
                "note_cliniche": (
                    "Paziente giunto in PS per dolore toracico irradiato al braccio sinistro. "
                    "ECG: sopraslivellamento ST in V1-V4. Eseguita PTCA primaria su IVA. "
                    "Decorso post-operatorio regolare. Dimesso in buone condizioni generali."
                ),
                "stato": "dimesso",
            }
        ],
    },
    "P002": {
        "id": "P002",
        "nome": "Giovanna",
        "cognome": "Ferrara",
        "data_nascita": "1972-09-30",
        "sesso": "F",
        "codice_fiscale": "FRRGNN72P70C351K",
        "comune_residenza": "Palermo",
        "ricoveri": [
            {
                "id_ricovero": "R002",
                "reparto": "Pneumologia",
                "data_ingresso": "2026-04-10",
                "data_dimissione": None,           # currently admitted
                "diagnosi_principale": "J44.1",    # BPCO con riacutizzazione acuta
                "diagnosi_secondarie": ["J45.9", "I10"],  # Asma, Ipertensione
                "drg_code": "088",
                "farmaci": ["Salbutamolo spray", "Fluticasone/Salmeterolo", "Prednisone 25mg"],
                "note_cliniche": (
                    "Paziente con BPCO nota giunta per dispnea ingravescente da 3 giorni. "
                    "SpO2 88% in aria ambiente. Iniziata ossigenoterapia e broncodilatatori. "
                    "Attualmente in reparto, condizioni stabili."
                ),
                "stato": "ricoverato",
            }
        ],
    },
    "P003": {
        "id": "P003",
        "nome": "Luca",
        "cognome": "Bianchi",
        "data_nascita": "1945-11-05",
        "sesso": "M",
        "codice_fiscale": "BNCLCU45S05C351T",
        "comune_residenza": "Messina",
        "ricoveri": [
            {
                "id_ricovero": "R003",
                "reparto": "Neurologia",
                "data_ingresso": "2026-01-15",
                "data_dimissione": "2026-01-28",
                "diagnosi_principale": "I63.9",    # Ictus ischemico, non specificato
                "diagnosi_secondarie": ["I10", "E78.5"],  # Ipertensione, Iperlipidemia mista
                "drg_code": "014",
                "farmaci": ["Warfarin 5mg", "Ramipril 5mg", "Atorvastatina 20mg"],
                "note_cliniche": (
                    "Ricovero per emiplegia destra a insorgenza improvvisa. "
                    "TAC: lesione ischemica nel territorio dell'arteria cerebrale media sinistra. "
                    "Iniziata terapia anticoagulante. Avviata fisioterapia. "
                    "Dimesso con lieve deficit motorio residuo all'arto superiore destro."
                ),
                "stato": "dimesso",
            },
            {
                "id_ricovero": "R004",
                "reparto": "Medicina Interna",
                "data_ingresso": "2026-04-05",
                "data_dimissione": "2026-04-12",
                "diagnosi_principale": "I10",      # Ipertensione essenziale
                "diagnosi_secondarie": ["N18.3"],  # Insufficienza renale cronica stadio 3
                "drg_code": "134",
                "farmaci": ["Amlodipina 10mg", "Furosemide 25mg"],
                "note_cliniche": (
                    "Paziente noto per ipertensione, in scompenso ipertensivo. "
                    "PA 200/110 alla misurazione iniziale. Rivalutazione farmacologica. "
                    "Dimesso con PA 140/85 e terapia aggiustata."
                ),
                "stato": "dimesso",
            },
        ],
    },
    "P004": {
        "id": "P004",
        "nome": "Sara",
        "cognome": "Esposito",
        "data_nascita": "1990-07-22",
        "sesso": "F",
        "codice_fiscale": "SPSSRA90L62C351P",
        "comune_residenza": "Napoli",
        "ricoveri": [],  # no previous admissions — tests the empty-ricovero edge case
    },
}


# ---------------------------------------------------------------------------
# Access functions — feature code should ALWAYS go through these
# ---------------------------------------------------------------------------

def get_patient(patient_id: str) -> Optional[dict]:
    """Return the patient record, or None if not found."""
    # Sanitize: strip whitespace, normalize to uppercase.
    # This makes the lookup robust against LLM hallucinations like
    # "p001", "P001 ", "P001\n" — all map to the canonical "P001".
    clean_id = patient_id.strip().upper()
    return PATIENTS.get(clean_id)


def list_patients() -> list[dict]:
    """Return all patients (without ricoveri detail, for list views)."""
    return [
        {k: v for k, v in p.items() if k != "ricoveri"}
        for p in PATIENTS.values()
    ]


def get_active_ricovero(patient_id: str) -> Optional[dict]:
    """Return the current open admission for a patient, or None."""
    p = get_patient(patient_id)
    if p is None:
        return None
    for r in p.get("ricoveri", []):
        if r["stato"] == "ricoverato":
            return r
    return None


def list_patients_by_diagnosis(icd10_prefix: str) -> list[dict]:
    clean_prefix = icd10_prefix.strip().upper()
    results = []
    for p in PATIENTS.values():
        ricoveri = p.get("ricoveri", [])
        if not ricoveri:
            continue
        latest = max(ricoveri, key=lambda r: r["data_ingresso"])
        if latest["diagnosi_principale"].upper().startswith(clean_prefix):
            results.append(p)
    return results