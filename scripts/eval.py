import sys
import yaml
import json
import asyncio
from pathlib import Path
from datetime import datetime

from client.parser import parse
from client.router import route
from fastmcp.client import Client
from fastmcp.client.transports.stdio import StdioTransport

_PROJECT_ROOT = Path(__file__).parent.parent
_SERVER_PATH = _PROJECT_ROOT / "server.py"

def _make_transport() -> StdioTransport:
    return StdioTransport(
        command=sys.executable,
        args=[str(_SERVER_PATH)],
        log_file=_PROJECT_ROOT / "server_stderr.log"
    )

async def run_eval():
    fixtures_path = _PROJECT_ROOT / "eval" / "fixtures" / "basic.yaml"
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = yaml.safe_load(f)

    report_lines = ["# Evaluation Report", f"**Date:** {datetime.now().isoformat()}", ""]
    
    passed = 0
    total = len(fixtures)

    try:
        async with Client(_make_transport()) as mcp:
            for i, fixture in enumerate(fixtures, 1):
                query = fixture["query"]
                exp_intent = fixture["expected_intent"]
                exp_kwargs = fixture.get("expected_kwargs", {})
                exp_subs = fixture.get("expected_substrings", [])
                exp_fail = fixture.get("expected_to_fail", False)

                report_lines.append(f"## Test {i}: {query}")
                report_lines.append(f"- **Expected Intent:** `{exp_intent}`")
                
                ir = parse(query)
                ir_dump = ir.model_dump()
                actual_intent = ir_dump.get("intent")
                
                if actual_intent != exp_intent:
                    if exp_fail and actual_intent == "unknown":
                        report_lines.append("- **Status:** ✅ PASSED (Failed as expected)")
                        passed += 1
                    else:
                        report_lines.append(f"- **Status:** ❌ FAILED (Intent mismatch: got `{actual_intent}`)")
                    report_lines.append("")
                    continue

                routed = route(ir)
                if routed is None:
                    if exp_fail:
                        report_lines.append("- **Status:** âœ… PASSED (Failed as expected)")
                        passed += 1
                    else:
                        report_lines.append("- **Status:** âŒ FAILED (IR is not routable)")
                    report_lines.append("")
                    continue

                if exp_fail:
                    report_lines.append("- **Status:** ❌ FAILED (Expected failure but got valid intent)")
                    report_lines.append("")
                    continue

                kwargs_match = True
                for k, v in exp_kwargs.items():
                    if str(ir_dump.get(k)) != str(v):
                        kwargs_match = False
                        report_lines.append(f"- **Status:** ❌ FAILED (Kwarg mismatch: `{k}` exp `{v}` got `{ir_dump.get(k)}`)")
                        break
                
                if not kwargs_match:
                    report_lines.append("")
                    continue

                tool_name, tool_args = routed
                try:
                    result = await mcp.call_tool(tool_name, arguments=tool_args)
                    result_text = result.content[0].text
                    
                    subs_match = True
                    for sub in exp_subs:
                        if sub not in result_text:
                            subs_match = False
                            report_lines.append(f"- **Status:** ❌ FAILED (Missing expected substring: `{sub}`)")
                            break
                    
                    if subs_match:
                        report_lines.append("- **Status:** ✅ PASSED")
                        passed += 1
                except Exception as e:
                    report_lines.append(f"- **Status:** ❌ FAILED (Tool call error: {e})")
                
                report_lines.append("")
                
    except Exception as e:
        report_lines.append(f"## CRITICAL ERROR\nServer connection failed: {e}")

    report_lines.insert(3, f"**Summary:** {passed}/{total} Passed")
    report_lines.insert(4, "")

    report_out = _PROJECT_ROOT / "eval" / "reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Eval completed: {passed}/{total} passed. Report saved to {report_out}")

if __name__ == "__main__":
    asyncio.run(run_eval())
