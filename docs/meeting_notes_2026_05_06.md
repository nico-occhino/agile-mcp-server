# Meeting Notes - 2026-05-06

These notes summarize technical direction only and do not include sensitive
patient data.

## Aria Integration

- Aria is Agile's orchestrator.
- Aria can connect to multiple MCP servers configured by address and port.
- This repository should become a stable, remotely deployable MCP server.
- The MCP/SSE deployment path is more important than replacing MCP with REST.

## Project Focus Split

- Sisca's focus: RAG, documents, LM Wiki, and discharge-letter style/context.
- Nicolo's focus: validation, uncertainty-aware guardrails, risk-based
  thresholds, latency/token/cost evaluation, and remote MCP deploy readiness.

## Next Steps

- Agile will provide API/account/test-server details later.
- Keep patient facts behind curated tools and future Agile APIs.
- Keep guardrail decisions explicit: ACCEPT, REQUIRES_REVIEW, REJECT.
- Add evaluation outputs for latency, optional token usage, and cost when
  available.
