# Agile API Integration Questions

Use this checklist before implementing `data/agile_api_repository.py`.

## Connectivity

- What is the base URL for development, staging, and production?
- Is access restricted by VPN, IP allowlist, mutual TLS, or internal DNS?
- Are Swagger/OpenAPI specs available?
- Are there separate endpoints for current admission, history, demographics,
  allergies, diagnoses, cohorts, and recent admissions?

## Endpoint Mapping

- Which Agile endpoint maps to `get_patient_status`?
- Which endpoint maps to `get_admission_history`?
- Which endpoint maps to `get_patients_by_diagnosis`?
- Which endpoint maps to `get_recently_admitted`?
- Are patient summaries and discharge drafts expected to use only retrieved
  facts from these endpoints, or will Agile provide dedicated summary APIs?

## Examples

- Can Agile provide request/response examples for each endpoint?
- Are examples representative of null values, missing fields, multiple events,
  and discharged/currently admitted patients?
- What is the canonical patient identifier field?

## Authentication

- Is the downstream auth header exactly `Authorization: Bearer <token>`?
- Should this MCP server forward the Aria JWT downstream unchanged?
- If not, should it exchange the JWT for an API token?
- Which claims and permissions are guaranteed in the Aria JWT?
- Is `HS256` with a pre-shared secret confirmed for the MCP boundary?

## Errors

- What status codes should be expected for 400, 401, 403, 404, 409, 429, and 500?
- What is the error response JSON shape?
- Are clinical "not found" cases represented as 404, 200 with empty body, or
  domain-specific error objects?

## Data Semantics

- Is pagination required for cohorts or history?
- What are page size limits?
- What date/time format is used?
- Which timezone should be assumed when none is present?
- Are diagnosis filters exact-code searches or prefix searches?
- Are diagnosis codes Italian ICD-9-CM numeric codes, internal Agile codes, or
  another vocabulary?

## Operations

- What are the expected timeout and retry policies?
- Are there rate limits?
- Which request IDs or correlation IDs should be sent?
- Which fields are forbidden in logs?
- Are raw clinical payloads allowed in development logs, or always prohibited?
