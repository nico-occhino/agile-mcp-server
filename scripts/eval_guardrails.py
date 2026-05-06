"""Run a small guardrail policy evaluation table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardrails.decision import evaluate_guardrail


SCENARIOS = [
    ("patient_summary", 0.90),
    ("patient_summary", 0.70),
    ("discharge_draft", 0.40),
    ("medication_or_therapy", 0.90),
    ("administrative_summary", 0.70),
    ("unknown", 0.80),
]


def main() -> None:
    rows = []
    for task_type, confidence in SCENARIOS:
        result = evaluate_guardrail(task_type=task_type, confidence=confidence)
        rows.append(result.model_dump())

    print("Task type                 Confidence  Risk      Threshold  Decision")
    print("-" * 74)
    for row in rows:
        print(
            f"{row['task_type']:<25} "
            f"{row['confidence']:<10.2f} "
            f"{row['risk_level']:<9} "
            f"{row['threshold']:<10.2f} "
            f"{row['decision']}"
        )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "guardrail_eval.json"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
