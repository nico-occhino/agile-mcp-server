# agile-mcp-server

**MCP server POC — uncertainty-aware NL2API for hospital management**

Curricular internship at Agile S.r.l., Catania.
Supervisors: Prof. Morana (UniCT), Ing. Nocita (CTO Agile).
Authors: Nicolò Carmelo Occhino, Francesca Calcagno.

---

## What this is

A hospital clinician types a natural-language request. This server translates it into a safe, pre-approved API call against Agile's hospital management system and returns a structured answer — with a calibrated confidence score.

The key design choice, explained by Prof. Morana in the kick-off meeting: **NL2API, not Text-to-SQL**. Instead of letting an LLM generate raw SQL (which opens a SQL-injection surface in a healthcare context), the LLM selects from a finite catalog of pre-hardened API tools. New capability requires a deliberate act by a human engineer, not a cleverly-worded prompt.

---

## Architecture

```
Clinician input
      ↓
Agile Gateway (MCP client) ← built by Agile, not in this repo
      ↓
agile-mcp-server (this repo)
      ├── features/patient_lookup.py   — deterministic data retrieval
      ├── features/patient_summary.py  — LLM summary + uncertainty estimation
      └── features/cohort.py           — multi-step workflow for population queries
      ↓
data/mock_store.py (Phase 1) → workflow/api_client.py (Phase 2, stub)
```

The server exposes 8 MCP tools. Each tool that uses an LLM internally returns an `UncertainResult` — the answer plus a confidence score and level (HIGH/MEDIUM/LOW). LOW-confidence responses should be surfaced as a clarification request, not displayed directly.

---

## The uncertainty layer

Standard LLM pipelines return one answer with no indication of confidence. In a clinical setting, a confidently-wrong discharge summary is more dangerous than a system that says "I'm not sure, please verify."

This server implements MC-sampling uncertainty estimation:

1. Call the LLM **N times independently** at `temperature > 0`.
2. Measure semantic disagreement across the N responses.
3. Return the most representative response + a confidence score in [0, 1].

**For categorical responses** (e.g. extracting a patient ID): normalized Shannon entropy over the vote distribution. If all N samples agree, entropy = 0, confidence = 1.0.

**For free-text responses** (e.g. clinical summaries): mean pairwise cosine similarity of sentence embeddings (`all-MiniLM-L6-v2`, 22M parameters, runs locally). This replaces the naive Jaccard n-gram overlap, which scores semantically equivalent Italian paraphrases as 0.0 — a critical failure for clinical text.

**Cohort confidence gating**: when summarizing a group of patients, the cohort confidence is set to the **minimum** individual confidence, not the mean. One LOW-confidence patient in a cohort of ten is not averaged away — it surfaces as a flag. Both min and mean are returned in the response for thesis analysis.

**Connection to prior work**: this is the inference-time analogue of MC Dropout. Instead of T stochastic forward passes through a network with dropped neurons (BioVid Bayesian CNN), we run N stochastic LLM samples. Same family of estimators, same interpretation of variance as epistemic uncertainty.

---

## Data schema

Records mirror Agile's real API response (provided by Ing. Nocita 2026-04-29):

```json
{
  "event": {
    "patientId": "45",
    "eventType": "ORDINARIO",
    "dateStart": "2026-04-28 15:07:00",
    "dateEnd": "2026-04-28 15:08:00",
    "uoDescription": "Reparto Ortopedia",
    "diagnosis": { "primary": "0123", "secondary": {} },
    "drg": { "code": "210", "mdc": "08", "weight": "2.345" },
    "eventReason": "Paziente ricoverato per..."
  },
  "patient": {
    "internalId": "45",
    "name": "Mario", "surname": "Rossi",
    "birthDate": "1991-01-01 00:00:00",
    "allergy": "Arachidi"
  }
}
```

Key differences from standard SDO assumptions:
- Patient IDs are numeric strings (`"45"`), not alphabetic codes.
- Diagnoses use Italian ministerial ICD-9-CM codes (`"4280"`), not ICD-10.
- `dateEnd: null` means currently admitted.
- `allergy` is safety-critical — always surfaced in summaries and prompts.

---

## Setup

```bash
# Clone
git clone https://github.com/nico-occhino/agile-mcp-server.git
cd agile-mcp-server

# Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -e .

# First run downloads the embedding model (~90MB, cached after that)
python -c "from workflow.uncertainty import _get_embedding_model; _get_embedding_model(); print('ready')"
```

**Environment variables** — copy `.env.example` to `.env` and fill in:

```env
# Any OpenAI-compatible provider works
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://api.openai.com/v1   # or Groq, Mistral, Agile internal
LLM_MODEL=gpt-4o-mini
UNCERTAINTY_SAMPLES=3
```

Supported providers and their `LLM_BASE_URL`:

| Provider | URL | Notes |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | default |
| Groq | `https://api.groq.com/openai/v1` | fast, free tier |
| Mistral | `https://api.mistral.ai/v1` | good free tier |
| Ollama | `http://localhost:11434/v1` | local, no key needed |
| Agile internal | set `AGILE_API_BASE_URL` | mid-internship |

---

## Running

```bash
# Start the MCP server (stdio transport, for development)
python server.py

# Start with HTTP transport (for Agile's gateway to connect)
fastmcp run server.py --transport sse --port 8000

# Open the browser-based inspector (run in same terminal as your venv)
npx @modelcontextprotocol/inspector python server.py
```

**Connect to Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "agile-hospital": {
      "command": "python",
      "args": ["C:/absolute/path/to/server.py"]
    }
  }
}
```

---

## Testing

```bash
pytest tests/ -v
```

24 tests, organized by layer:

- **TestDeterministicLookup** — data layer, pure Python, no LLM. Always fast.
- **TestUncertaintyMath** — entropy and cosine similarity math in isolation. 3 tests require the embedding model and a network connection on first run (model is ~90MB, cached after).
- **TestCohortWorkflowStructure** — multi-step workflow shape and edge cases, no LLM.

LLM integration tests (full pipeline calls with real API keys) belong in `scripts/eval.py`, not in the unit test suite. They should check output *properties* ("does the response mention the correct patient name?") not exact strings. That is the thesis evaluation methodology.

---

## Registered MCP tools

| Tool | Type | Description |
|---|---|---|
| `get_patient_age` | Deterministic | Age in years from patient ID |
| `get_patient_status` | Deterministic | Current admission status + allergy |
| `get_admission_history` | Deterministic | Event history (Phase 2: full history endpoint) |
| `get_patient_summary` | LLM + uncertainty | Concise clinical summary with confidence score |
| `get_patient_discharge_draft` | LLM + uncertainty | Draft discharge letter (multi-step: extract → generate) |
| `get_patients_by_diagnosis` | Deterministic | Cohort filter by diagnosis prefix |
| `get_cohort_summary` | LLM + uncertainty | Cohort narrative with gated aggregate confidence |
| `get_recently_admitted` | Deterministic | Time-window admission filter |

---

## Repository structure

```
agile-mcp-server/
├── server.py                  # FastMCP entrypoint, registers all 8 tools
├── pyproject.toml             # dependencies
├── .env.example               # environment variable template
│
├── data/
│   ├── mock_store.py          # Phase 1: synthetic patients (Nocita's real schema)
│   └── mock_patient.json      # placeholder for Nocita's JSON fixtures (Phase 2)
│
├── features/
│   ├── patient_lookup.py      # deterministic tools (no LLM)
│   ├── patient_summary.py     # LLM + uncertainty tools
│   └── cohort.py              # multi-step population workflow
│
├── workflow/
│   ├── uncertainty.py         # MC-sampling, Shannon entropy, cosine similarity
│   ├── llm_client.py          # OpenAI-compatible LLM wrapper
│   └── api_client.py          # Phase 2 stub (Agile's real APIs)
│
└── tests/
    └── test_features.py       # 24 unit tests, no LLM calls
```

---

## Development phases

| Phase | Data | Status |
|---|---|---|
| 1 | Synthetic mock store (`data/mock_store.py`) | ✅ Active |
| 2 | Nocita's real SDO-shaped JSON fixtures | 🔜 Pending NDA + API contracts |
| 3 | Live Agile hospital APIs via `workflow/api_client.py` | 🔜 Pending VPN/on-site access |

To move from Phase 1 to Phase 2: replace the import in each feature file from `from data.mock_store import get_patient` to `from workflow.api_client import get_patient`. Feature code doesn't change; only the data layer does.

---

## Known limitations (Phase 1 POC)

- **Single event per patient**: Agile's API returns one current event per request. Multi-event admission history requires a dedicated history endpoint (Phase 2).
- **No authentication**: `workflow/api_client.py` stubs auth. Production requires OAuth2 or API key auth at the FastAPI gateway layer.
- **Embedding model warm-up**: first call to any freetext uncertainty function takes ~1s to load `all-MiniLM-L6-v2`. Subsequent calls use the in-memory cache. Production would warm up at server startup.
- **Diagnosis codes**: mock data uses Italian ministerial ICD-9-CM numeric codes. The `list_patients_by_diagnosis` function uses prefix matching (`"428"` matches `"4280"`, `"4281"`, etc.), which works for the mock but may need a proper code vocabulary in Phase 2.