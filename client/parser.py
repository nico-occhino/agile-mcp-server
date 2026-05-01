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

from client.prompt_builder import build_parser_system_prompt
_SYSTEM_PROMPT = build_parser_system_prompt()


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
