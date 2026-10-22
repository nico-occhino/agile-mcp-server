# ADR 0002: Independent synthetic REST boundary

Status: accepted for the v2 portfolio demonstration.

The hospital API runs as a separate FastAPI service. It validates the forwarded
Bearer JWT and offers health, patient lookup, and filtered paginated list
endpoints. Patient names and codes are explicitly synthetic. The MCP server
depends on `PatientRepository`, and `HttpPatientRepository` owns HTTP mechanics.

This proves the transport and trust boundary without asserting that the
organization's real API has the same shape. Future integration requires a new
adapter and agreed JWT claims, paths, data schemas, failure codes, and privacy
controls. No production endpoint or secret is embedded in the repository.

