# Debugging Playbook

## MCP Server Starts But GET / Returns 404

- Symptom: Browser opens `http://127.0.0.1:8001/` and shows 404.
- Probable cause: FastMCP SSE does not serve a homepage.
- Command/check: `fastmcp run server.py --transport sse --port 8001`
- Fix: Connect the MCP client or Inspector to `/sse`, not `/`.

## Inspector Cannot Connect To /sse

- Symptom: MCP Inspector cannot list tools.
- Probable cause: wrong port, server not running, blocked local process, or stale
  Inspector target.
- Command/check: start with `fastmcp run server.py --transport sse --port 8001`
  and use `http://127.0.0.1:8001/sse`.
- Fix: restart the server, verify the port, and keep SSE transport for Aria.

## Swagger Works But MCP Does Not

- Symptom: `uvicorn api_demo:app --reload --port 8000` works, but Aria or
  Inspector cannot call tools.
- Probable cause: Swagger is the optional FastAPI demo, not the MCP server.
- Command/check: run `fastmcp run server.py --transport sse --port 8001`.
- Fix: Use Swagger only for demo/debug. Use FastMCP/SSE for MCP.

## AUTH_ENABLED=true And Requests Fail

- Symptom: auth demo returns missing or invalid token errors.
- Probable cause: auth enforcement is enabled but the caller is not sending an
  Aria JWT.
- Command/check: inspect `.env` for `AUTH_ENABLED=true`.
- Fix: Set `AUTH_ENABLED=false` for Phase 1 demos, or send
  `Authorization: Bearer <token>`.

## JWT Missing

- Symptom: `Missing authentication token.`
- Probable cause: no Authorization header or malformed Bearer value.
- Command/check: verify the request has `Authorization: Bearer <token>`.
- Fix: Send the token in the HTTP header for SSE requests.

## JWT Invalid Signature

- Symptom: `Invalid token` from auth helpers.
- Probable cause: wrong `JWT_SECRET`, wrong algorithm, or edited token payload.
- Command/check: compare `JWT_ALGORITHM` and `JWT_SECRET` with Aria config.
- Fix: Use the shared HS256 secret agreed with Agile; never log or commit it.

## Wrong JWT_AUDIENCE / JWT_ISSUER

- Symptom: otherwise valid token fails verification.
- Probable cause: `aud` or `iss` claims do not match environment config.
- Command/check: decode header/payload locally without trusting it and compare
  `aud` and `iss` to `JWT_AUDIENCE` and `JWT_ISSUER`.
- Fix: align environment variables with Aria token claims.

## LLM API Key Missing

- Symptom: patient summary or discharge draft fails when calling the LLM.
- Probable cause: `LLM_API_KEY` / provider settings are missing or invalid.
- Command/check: inspect `.env` values for `LLM_API_KEY`, `LLM_BASE_URL`, and
  `LLM_MODEL`.
- Fix: configure an OpenAI-compatible provider, or use deterministic tools and
  unit tests that monkeypatch LLM calls.

## Agile API 401 / 403

- Symptom: future Agile adapter receives unauthorized or forbidden responses.
- Probable cause: missing downstream Bearer token, wrong permissions, or expired
  token.
- Command/check: confirm whether Aria JWT should be forwarded downstream.
- Fix: forward the confirmed token format and verify claims/permissions.

## Agile API 404

- Symptom: patient or admission endpoint returns not found.
- Probable cause: wrong patient ID, endpoint mismatch, or domain-level missing
  record.
- Command/check: compare request path and example payloads from Agile docs.
- Fix: map 404 to `None` or empty list where the repository protocol expects it.

## Agile API 500

- Symptom: downstream API returns server error.
- Probable cause: Agile service failure or malformed request reaching backend.
- Command/check: capture status, endpoint name, correlation ID, and sanitized
  error category.
- Fix: return controlled tool error; do not expose raw payloads or stack traces.

## Timeout

- Symptom: MCP tool hangs or returns timeout.
- Probable cause: network/VPN issue, slow Agile API, or no HTTP timeout.
- Command/check: test connectivity from the deployment host and inspect adapter
  timeout settings.
- Fix: set explicit httpx timeouts and return controlled timeout errors.

## Embedding Model First-Run Delay

- Symptom: first freetext uncertainty test or call is slow.
- Probable cause: `sentence-transformers/all-MiniLM-L6-v2` loads or downloads on
  first use.
- Command/check: run
  `python -c "from workflow.uncertainty import _get_embedding_model; _get_embedding_model(); print('ready')"`.
- Fix: warm the model at startup in environments that use freetext uncertainty.
