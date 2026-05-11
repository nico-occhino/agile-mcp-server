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
- Swagger endpoint: `POST /auth/decode`
- Swagger endpoint: `POST /auth/authorize-tool`
- Local script: `python scripts/demo_jwt.py`

These surfaces do not access patient data and do not execute clinical tools.

## Integration Questions for Nocita

1. Will Aria pass JWT in the HTTP `Authorization` header, MCP metadata, or a
   tool argument?
2. Will Aria use `HS256` with a shared secret or `RS256` with a public/private
   key pair?
3. What exact claims will Aria include?
4. Does Aria enforce authorization before calling our MCP server, or must this
   server enforce it too?
5. Should this server forward the same JWT to Agile APIs as a Bearer token?

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
