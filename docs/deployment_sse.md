# MCP SSE Deployment Notes

## Local SSE Run

```bash
fastmcp run server.py --transport sse --port 8001
```

Expected MCP SSE endpoint:

```text
http://127.0.0.1:8001/sse
```

## Why `/` Returns 404

The MCP server exposes the SSE transport endpoint, not a general web homepage.
Opening `http://127.0.0.1:8001/` in a browser can return 404 and still be
normal. MCP clients should connect to `/sse`.

## MCP Inspector

Use MCP Inspector to verify tool discovery and calls:

```bash
npx @modelcontextprotocol/inspector
```

Configure it to connect to the SSE endpoint above, then confirm that tools such
as `get_patient_status` and `evaluate_clinical_output_guardrail` appear.

## Aria Configuration

When Agile provides the target test server, Aria should store this MCP server's
host and port in its MCP server configuration table. The exact public endpoint
will depend on Agile's network, firewall, and reverse-proxy setup.

## Environment Variables

LLM-based tools need the usual OpenAI-compatible configuration:

```env
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
UNCERTAINTY_SAMPLES=3
```

Deterministic lookup tools and the standalone guardrail tool do not call the
LLM.
