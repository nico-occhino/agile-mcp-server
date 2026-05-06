"""
client/prompt_builder.py
------------------------
Auto-generates the parser system prompt from the Pydantic IR schema.
"""

from __future__ import annotations

import json
from typing import get_args

from client.ir_schema import IR, UnknownIR

def build_parser_system_prompt() -> str:
    """
    Generate the system prompt for parser.py from the IR schema.

    Walks every concrete IR class in client/ir_schema.py, extracts
    the intent literal, the field names, and produces a JSON example
    for each. Builds a complete Italian-language system prompt.
    """
    ir_union = get_args(IR)[0]
    classes = [cls for cls in get_args(ir_union) if cls is not UnknownIR]

    intents_text = []
    
    patient_ids = ["45", "46", "47", "48"]
    diagnosis_codes = ["428", "434", "012"]
    
    p_idx = 0
    d_idx = 0

    for idx, cls in enumerate(classes, start=1):
        intent_literal = get_args(cls.model_fields["intent"].annotation)[0]
        example = {"intent": intent_literal}
        
        for field_name, field_info in cls.model_fields.items():
            if field_name == "intent":
                continue
            if field_name == "patient_id":
                example[field_name] = patient_ids[p_idx % len(patient_ids)]
                p_idx += 1
            elif field_name == "diagnosis_code_prefix":
                example[field_name] = diagnosis_codes[d_idx % len(diagnosis_codes)]
                d_idx += 1
            elif field_name == "days":
                example[field_name] = 7
            else:
                example[field_name] = f"<{field_name}>"

        example_json = json.dumps(example)
        intents_text.append(f"{idx}. {intent_literal}\n   Esempio: {example_json}")

    intents_joined = "\n\n".join(intents_text)

    prompt = f"""Sei un classificatore di intenti clinici. Analizza la query dell'utente e restituisci SOLO un oggetto JSON valido (nessun markdown, nessuna spiegazione).

Gli intenti validi sono:

{intents_joined}

REGOLE IMPORTANTI:
- I patient_id sono stringhe numeriche: "45", "46", "47", "48", ecc.
- I diagnosis_code_prefix sono codici diagnosi numerici ministeriali italiani:
    428 = scompenso cardiaco (heart failure)
    434 = ictus (stroke)
    012 = trauma ortopedico
- Se la query è ambigua, mancano informazioni obbligatorie, o non corrisponde a nessun intento, restituisci:
  {{"intent": "unknown", "raw_query": "<query originale>", "reason": "<motivo breve in inglese>"}}
- Restituisci SOLO JSON grezzo. Nessun blocco markdown, nessun testo aggiuntivo prima o dopo.

ESEMPI DI INTENTI SCONOSCIUTI (restituisci sempre unknown per questi):

"Chi è?" → {{"intent": "unknown", "raw_query": "Chi è?", "reason": "missing patient_id"}}
"Come si cura il diabete?" → {{"intent": "unknown", "raw_query": "Come si cura il diabete?", "reason": "medical knowledge question, not a patient lookup"}}
"Quali farmaci prende?" → {{"intent": "unknown", "raw_query": "Quali farmaci prende?", "reason": "missing patient_id"}}
"Che ore sono?" → {{"intent": "unknown", "raw_query": "Che ore sono?", "reason": "unrelated to clinical tools"}}
Se la query non specifica un patient_id numerico quando richiesto, o non corrisponde
a nessun tool, restituisci SEMPRE unknown.
"""
    return prompt
