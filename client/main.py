"""
client/main.py
--------------
Async entrypoint for the reference MCP client.

Two modes:
  - Single query:    python -m client.main --query "Come sta il paziente 46?"
  - Interactive REPL: python -m client.main   (Ctrl+C to exit)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 can't handle emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Load .env from the project root before any module that reads env vars
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from fastmcp import Client  # noqa: E402  (must come after load_dotenv)
from fastmcp.client.transports import StdioTransport

from client.ir_schema import IR
from client.parser import parse
from client.router import route

# ---------------------------------------------------------------------------
# StdioTransport for the server subprocess
_PROJECT_ROOT = str(Path(__file__).parent.parent)

def _make_transport() -> StdioTransport:
    """Create a fresh StdioTransport pointing to server.py.

    Uses sys.executable (the currently-running venv Python) so that the server
    subprocess has access to the same installed packages as the client.
    """
    return StdioTransport(
        command=sys.executable,
        args=["server.py"],
        cwd=_PROJECT_ROOT,
        log_file=Path(_PROJECT_ROOT) / "server_stderr.log",
    )

# ---------------------------------------------------------------------------
# Excluded fields in deterministic-tool rendering
# ---------------------------------------------------------------------------

_EXCLUDED_FIELDS = {"found", "patient_id", "full_name", "confidence", "confidence_level", "rationale", "allergy"}

# ---------------------------------------------------------------------------
# Emoji helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_EMOJI = {
    "HIGH":   "✅",
    "MEDIUM": "⚠️",
    "LOW":    "❌",
}


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def render(tool_name: str, result: dict) -> str:
    """
    Convert a raw MCP tool result dict into human-readable Italian text.

    Rules (applied in order):
    1. LOW confidence → refusal message; never show the result content.
    2. Allergy present → print 🚨 line FIRST before anything else.
    3. LLM-based tools (have "result" key) → print result text + confidence.
    4. Deterministic tools → print full_name then remaining fields.
    5. Append confidence line at end when the key is present.
    """
    lines: list[str] = []
    confidence_level = result.get("confidence_level", "").upper()
    confidence = result.get("confidence")
    rationale = result.get("rationale", "")

    # --- Rule 1: LOW confidence → hard refusal ---
    if confidence_level == "LOW":
        lines.append(
            f"⚠️  CONFIDENZA BASSA — risultato non visualizzato.\n"
            f"   Motivazione: {rationale}\n"
            f"   Si prega di verificare manualmente con il personale clinico."
        )
        if confidence is not None:
            lines.append(f"   Confidenza: {confidence:.0%}")
        return "\n".join(lines)

    # --- Rule 2: Allergy — always FIRST ---
    allergy = result.get("allergy")
    if allergy is not None:
        lines.append(f"🚨 ALLERGIA: {allergy}")

    if "error" in result:
        name = result.get("patient_name", result.get("full_name", ""))
        name_line = f"Paziente: {name}\n" if name else ""
        return f"{name_line}⚠️  {result['error']}"

    # --- Rules 3 & 4: LLM-based vs deterministic ---
    if "result" in result:
        # LLM-based tool
        lines.append(result["result"])
    else:
        # Deterministic tool — print full_name first, then remaining fields
        full_name = result.get("full_name")
        if full_name:
            lines.append(f"Paziente: {full_name}")
        for key, value in result.items():
            if key not in _EXCLUDED_FIELDS:
                lines.append(f"  {key}: {value}")

    # --- Rule 5: Confidence footer ---
    if confidence is not None:
        emoji = _CONFIDENCE_EMOJI.get(confidence_level, "ℹ️")
        lines.append(f"{emoji} Confidenza: {confidence:.0%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# handle — full pipeline
# ---------------------------------------------------------------------------

async def handle(query: str) -> None:
    """Run one query through the full parse → route → MCP call → render pipeline."""
    print(f"📝 Query: {query}")

    ir: IR = parse(query)
    print(f"🧠 IR:    {ir.model_dump_json()}")

    routed = route(ir)
    if routed is None:
        print(
            "❓ Non riesco a capire la richiesta o mancano informazioni necessarie.\n"
            "   Prova a riformulare la domanda specificando l'ID del paziente o\n"
            "   il codice ICD-9-CM della diagnosi."
        )
        return

    tool_name, kwargs = routed
    print(f"🔧 Tool:  {tool_name}({json.dumps(kwargs, ensure_ascii=False)})")

    async with Client(_make_transport()) as mcp:
        result_raw = await mcp.call_tool(tool_name, kwargs)

    result: dict = json.loads(result_raw.content[0].text)

    print("📋 Risultato:")
    print(render(tool_name, result))


# ---------------------------------------------------------------------------
# main — argument parsing + REPL
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference MCP client for agile-mcp-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            '  python -m client.main --query "Come sta il paziente 46?"\n'
            "  python -m client.main   # avvia REPL interattivo"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query in linguaggio naturale da eseguire una sola volta.",
    )
    args = parser.parse_args()

    if args.query:
        asyncio.run(handle(args.query))
    else:
        # Interactive REPL
        print("Agile MCP Client — digita una query clinica. Ctrl+C per uscire.\n")
        try:
            while True:
                try:
                    query = input("Query: ").strip()
                except EOFError:
                    break
                if not query:
                    continue
                asyncio.run(handle(query))
                print()
        except KeyboardInterrupt:
            print("\nArrivederci.")
            sys.exit(0)


if __name__ == "__main__":
    main()
