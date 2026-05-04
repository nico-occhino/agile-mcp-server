"""Demo the lightweight Phase 1 RAG context builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.context_builder import build_rag_context


def main() -> None:
    context = build_rag_context("discharge summary allergy follow-up", top_k=3)

    print("Retrieved chunks:")
    for chunk in context["chunks"]:
        print(
            f"- {chunk['chunk_id']} | score={chunk['score']:.3f} | "
            f"source={chunk['source']}"
        )

    print("\nContext text:")
    print(context["context_text"])


if __name__ == "__main__":
    main()
