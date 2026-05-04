# Architecture

This proof of concept separates natural-language understanding from clinical
capability execution. The orchestrator is intentionally small: its job is to
turn a request into an auditable Intermediate Representation (IR), validate that
IR, and only then invoke an allowed tool.

## NL2API Flow

```text
Natural language query
  -> rule-based parser / future LLM parser
  -> Pydantic IR
  -> validator
  -> router
  -> curated clinical tool
  -> rendered response
```

## RAG-Augmented Generation Flow

```text
Natural language query
  -> IR / validator / router
  -> clinical tool retrieves patient facts
  -> RAG retriever retrieves domain context
  -> LLM generation uses patient facts + retrieved context
  -> uncertainty + response
```

## Safety Boundaries

- The LLM should not directly access the database, SQL, or hospital APIs.
- The IR is the audit checkpoint: it records the selected intent and extracted
  parameters before execution.
- MCP tools are the allowed capability layer. Adding a new capability means
  adding a curated tool and an explicit IR intent.
- The current parser is deterministic only for Phase 1. A future LLM parser can
  replace it, but it should still emit the same Pydantic IR and pass through the
  same validator and router.
- APIs and curated tools are the source of clinical facts.
- RAG is not used to retrieve live patient records.
- RAG is not a safety guarantee; it only supplies domain context such as
  templates, glossary entries, and safety reminders.
- Uncertainty confidence measures semantic self-consistency, not clinical
  correctness.
- Learned indexing belongs behind the retriever interface, not inside patient
  lookup or clinical execution logic.
