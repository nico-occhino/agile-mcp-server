# ADR 0001: One public execution path

Status: accepted for the v2 portfolio demonstration.

The deployable MCP server exposes only `execute_query`. It authenticates the
caller, applies optional static input policy, parses into a typed `QueryIR`,
validates intent-specific arguments, and invokes `QueryService` through an
injected repository. Direct patient tools in the original `server.py` are
development-only and are not packaged as the v2 service.

This avoids multiple public routes with different authorization and validation
rules. The tradeoff is a smaller v2 intent catalog than the legacy prototype.
New intents must be added to the IR, parser, service, repository and tests
together before they become public.

