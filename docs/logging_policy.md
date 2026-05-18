# Logging Policy

This project handles synthetic data in Phase 1, but Phase 2 should assume real
clinical data. Logs must be useful for debugging without becoming a data leak.

## Safe To Log

These fields are generally safe when kept concise:

- MCP tool name
- latency and timeout category
- success/failure status
- guardrail decision
- output confidence score
- task risk level
- input risk scores when they do not include raw patient text
- `token_present` boolean
- auth subject and role only if Agile permits it
- sanitized error class or status code
- correlation/request ID

## Do Not Log

Never log these by default:

- raw JWT
- `Authorization` header
- JWT signature
- raw clinical payload
- full patient identity in real data
- full prompt containing patient data
- full generated clinical text unless explicitly permitted
- API keys, shared secrets, private keys, or public-key material copied from
  secret storage
- full downstream HTTP request/response bodies

## Phase 2 Guidance

- Prefer `token_present=True/False` over logging token values.
- Prefer `subject` or role only after Agile confirms they are allowed in logs.
- Prefer patient internal IDs only when strictly necessary and approved.
- Keep debug logs off in production unless Agile explicitly enables them.
- Use correlation IDs to connect MCP logs with Aria/downstream API logs.
- Treat prompts and generated clinical text as clinical data.
