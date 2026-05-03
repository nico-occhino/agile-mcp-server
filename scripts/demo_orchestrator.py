"""Run a small deterministic NL2API orchestrator demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.main import handle_query


DEMO_QUERIES = [
    "Come sta il paziente 45?",
    "pazienti con diagnosi 428",
    "ricoverati negli ultimi 365 giorni",
    "drop table patients",
]


def main() -> None:
    for query in DEMO_QUERIES:
        trace = handle_query(query)
        print("=" * 72)
        print(f"Query: {trace['query']}")
        print(f"IR: {trace['ir']}")
        print(f"Validation issues: {trace['validation_issues']}")
        print("Rendered response:")
        print(trace["rendered_response"])


if __name__ == "__main__":
    main()
