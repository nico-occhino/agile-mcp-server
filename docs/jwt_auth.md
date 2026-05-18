# JWT Auth Skeleton

## What JWT Is

A JSON Web Token (JWT) is a signed token with three parts:

```text
header.payload.signature
```

- Header: algorithm and token type.
- Payload: claims such as subject, role, permissions, issuer, and expiration.
- Signature: cryptographic proof that the payload was issued by the trusted
  signer and has not been modified.

Signature verification matters because JWT payloads are only base64url encoded,
not encrypted. Anyone can read or edit a payload locally, but an edited token
will fail verification unless the attacker can also produce a valid signature.

## HS256 vs RS256

`HS256` uses one shared secret for signing and verification. It is simple for
local demos, but every verifier that knows the secret could also sign tokens.

`RS256` uses a private key to sign and a public key to verify. Aria can keep the
private key, while this MCP server receives only the public key. That is usually
the cleaner deployment shape when multiple services verify tokens.

## Claims Used

The Phase 2 skeleton maps these JWT claims into `AuthContext`:

| Claim | AuthContext field |
|---|---|
| `sub` | `subject` |
| `username` or `preferred_username` | `username` |
| `role` | `role` |
| `department` | `department` |
| `permissions` | `permissions` |
| `scope` | `permissions`, split by spaces |
| `iss` | `issuer` |
| `aud` | `audience` |
| `iat` | `issued_at` |
| `exp` | `expires_at` |

If both `permissions` and `scope` exist, they are combined and deduplicated in
order.

## AuthContext

`AuthContext` is the typed representation of the caller:

```text
subject, username, role, department, permissions,
issuer, audience, issued_at, expires_at, raw_claims
```

Feature functions should not pass raw JWT claims around. Future integration
should extract `AuthContext` once per request and use it for authorization and
audit metadata.

## Tool Permission Policy

The initial policy maps MCP tool names to required permissions:

| Tool | Permissions |
|---|---|
| `get_patient_age` | `patient:read` |
| `get_patient_status` | `patient:read` |
| `get_admission_history` | `patient:read` |
| `get_patient_summary` | `patient:read`, `summary:generate` |
| `get_patient_discharge_draft` | `patient:read`, `discharge:generate` |
| `get_patients_by_diagnosis` | `cohort:read` |
| `get_cohort_summary` | `cohort:read`, `summary:generate` |
| `get_recently_admitted` | `cohort:read` |
| `evaluate_clinical_output_guardrail` | none |
| `evaluate_input_prompt_guardrail` | none |
| `decode_jwt_auth_context` | none |
| `authorize_tool_access` | none |

Empty permissions mean allowed. `role=admin` can access all tools. Unknown
tools require `admin`, so the default is conservative.

## Current Phase 2 Demo Surface

Authentication is not enforced by default:

```env
AUTH_ENABLED=false
```

Available demo surfaces:

- MCP tool: `decode_jwt_auth_context`
- MCP tool: `authorize_tool_access`
- MCP tool: `current_jwt_auth_context`
- Swagger endpoint: `POST /auth/decode`
- Swagger endpoint: `POST /auth/authorize-tool`
- Local script: `python scripts/demo_jwt.py`

These surfaces do not access patient data and do not execute clinical tools.

## Confirmed Aria Direction

Ing. Nocita clarified the expected Phase 2 integration:

- Aria is the MCP client.
- This repository is the MCP server.
- Aria will call this MCP server over SSE.
- Aria will pass JWT in the HTTP header as `Authorization: Bearer <token>`.
- JWT should become mandatory in Phase 2.
- If `AUTH_ENABLED=true` and no JWT is received, the request should be rejected
  because it is not coming through Aria.
- Agile likely uses `HS256` with a pre-shared secret as the first integration
  path.
- `RS256` support remains available, but it is not the first expected path.
- JWT header and payload are readable; the signature provides integrity.
- If encrypted payloads are required later, that would be JWE, not plain JWT.
- The received JWT may also be forwarded to downstream clinical-record APIs as
  a Bearer token.

## FastMCP Header Access

The installed FastMCP version exposes HTTP request headers for SSE requests via:

```python
from fastmcp.server.dependencies import get_http_headers

headers = get_http_headers(include={"authorization"})
```

FastMCP excludes `authorization` by default from forwarded header helpers, so
the helper must explicitly include it. This repo wraps that behavior in
`auth.request_auth.get_current_auth_context()`.

The MCP demo tool `current_jwt_auth_context` exercises this real header-based
path without protecting clinical tools yet. The next step is to test it from
Aria over SSE and then decide where to enforce authorization centrally.

## Integration Questions for Nocita

1. What exact claims will Aria include?
2. Does Aria enforce authorization before calling our MCP server, or must this
   server enforce it too?
3. Should this server forward the same JWT to Agile APIs as a Bearer token?
4. Will the pre-shared `HS256` secret be environment-specific and rotated?

## Future Integration Notes

When Aria token passing is confirmed, auth context should be extracted once per
MCP request. Clinical tools should receive an `AuthContext` or request context,
not a raw token, unless the transport makes that impossible. Prefer central
middleware/request context over adding token arguments to every tool. If FastMCP
cannot expose request metadata for the chosen transport, wrapper tools or a
controlled token parameter can be used as a fallback.

## Limitations

- No production auth enforcement by default.
- No refresh-token flow.
- No real Agile API calls.
- No real secrets committed.
- The current policy is an initial auditable map, not a final hospital RBAC
  model.
