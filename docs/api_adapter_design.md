# API Adapter Design

Phase 2 should connect Agile APIs through the repository abstraction, not by
placing HTTP calls inside feature functions.

```text
features
  -> data.repository.get_repository()
  -> data.agile_api_repository.AgileApiRepository
  -> workflow.api_client
  -> Agile API
```

## Why Not httpx Inside Features

Clinical feature functions should express domain behavior: patient status,
admission history, cohort lookup, and summary generation. If they call `httpx`
directly, every feature becomes responsible for auth headers, timeouts, status
codes, retries, schema mapping, and logging. That spreads integration details
across the safety-critical layer.

Keeping HTTP in `workflow/api_client.py` and domain mapping in
`data/agile_api_repository.py` gives one central place to adapt Agile responses
without changing MCP tool behavior.

## Proposed Modules

`workflow/api_client.py`

- low-level async HTTP only
- base URL and timeout handling
- request headers
- Bearer token forwarding when confirmed
- status-code translation
- JSON parsing

`data/agile_api_repository.py`

- implements `PatientRepository`
- maps Agile response JSON into the internal shape expected by features
- handles missing fields and nulls consistently
- converts API errors into controlled repository-level failures

`data/repository.py`

- remains the central selection point
- defaults to `data.mock_store` in Phase 1
- can select `AgileApiRepository` later by environment/startup config

## Internal Schema Mapping

Features currently expect patient records shaped like the mock data: a patient
object, an event object, allergy, dates, department, diagnosis, DRG, and
admission status. `AgileApiRepository` should normalize Agile API responses into
that internal shape.

This mapping should explicitly document:

- patient ID source field
- demographic fields
- current event vs historical events
- null `dateEnd` meaning currently admitted
- allergy field behavior
- diagnosis code and description fields
- date/time parsing and timezone assumptions

## Error Handling

The adapter should avoid leaking raw HTTP exceptions to feature functions.
Preferred behavior:

- patient not found -> `None` or empty list, matching current repository methods
- unauthorized/forbidden -> controlled repository error for audit/debug
- timeout -> controlled timeout error
- malformed response -> controlled schema/mapping error

MCP tools can then render stable, predictable responses instead of surfacing
transport details.

## Timeout Handling

HTTP calls should have explicit connect/read timeouts. Defaults should be short
enough for an interactive MCP workflow and configurable for Agile environments.
The adapter should not hang a tool call indefinitely.

## JWT Forwarding

Nocita indicated that the Aria JWT may be forwarded downstream to clinical
record APIs as `Authorization: Bearer <token>`. The current auth layer can
extract the token from the MCP/SSE header. The exact forwarding behavior should
wait for Agile API confirmation.

Clinical feature functions should receive repository results, not raw tokens.
If forwarding is needed, pass auth context/token through central request context
or repository construction rather than adding token parameters to every feature.

## Test Strategy

- unit-test `AgileApiRepository` with mocked HTTP responses
- test status-code mapping for 401, 403, 404, 429, 500, and timeout
- test schema mapping with representative Agile payloads
- test JWT forwarding by asserting the outgoing Authorization header
- keep feature tests independent from real Agile APIs
- keep unit tests free of real LLM and real network calls
