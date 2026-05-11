# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sales call agent with two composable pipelines:
- **Pipeline A** — raw transcript → structured `CallAnalysis` (transcript extraction via one-shot LLM structured output)
- **Pipeline B** — `CallAnalysis` → enriched, scored, routed `EnrichedCallResult` (agent loop with tool calls)

n8n triggers A→B in sequence (from a raw transcript) or B alone (from an Attention webhook payload).

## Commands

```bash
# Install dependencies
uv sync

# Run the API server
uv run uvicorn api.main:app --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_extractor.py

# Run a single test
uv run pytest tests/test_extractor.py::test_name

# Start full stack (API + n8n)
docker-compose up

# Build and start
docker-compose up --build
```

## Architecture

### Data Flow

```
Raw transcript
     │
     ▼
extractor.py  ──(OpenAI structured output)──►  CallAnalysis
     │
     ▼
enricher.py   ──(agent loop + tool calls)──►   EnrichedCallResult
     │                                          ├── CompanyProfile
     ▼                                          ├── RoutingDecision
follow_up.py  ──(plain LLM calls)──►           ├── follow_up_email
                                                └── slack_summary
```

### Key Design Decisions

**`src/agent/schemas.py` is the single source of truth.** All Pydantic models live here — `CallAnalysis`, `CompanyProfile`, `RoutingDecision`, `EnrichedCallResult`. API contracts, LLM prompts, and n8n field mappings all derive from these shapes.

**Pipeline A uses `client.beta.chat.completions.parse`** with `response_format=CallAnalysis` — this returns a validated Pydantic object directly, no manual JSON parsing needed.

**Pipeline B is a proper agent loop** in `enricher.py`: the LLM calls `lookup_company` (via `tools.py`) before reasoning, then a second structured output call extracts `CompanyProfile` and `RoutingDecision` from the agent's final message.

**Two FastAPI endpoints** in `api/main.py`:
- `POST /analyze/transcript` — Pipeline A → B (transcript string in)
- `POST /analyze/webhook` — Pipeline B only (pre-structured `CallAnalysis` in, e.g. from Attention)

Both return `EnrichedCallResult`.

### Package Layout

```
src/agent/        # core logic
api/main.py       # FastAPI app
n8n/              # exported workflow JSONs (commit these)
retool/           # exported dashboard JSON
tests/fixtures/   # sample transcripts and webhook payloads
```

### Dependencies

Managed via `uv`. Key packages: `fastapi`, `uvicorn`, `openai`, `pydantic`, `structlog`, `httpx`. Dev: `pytest`, `pytest-asyncio`, `respx` (for mocking HTTP in tests).

### Logging

`src/agent/logger.py` uses `structlog` to emit JSON log lines. Every request should log `call_id`, `pipeline`, `priority`, `tokens_used`, and `duration_ms`.

### Testing Strategy

- `test_extractor.py` — mock OpenAI HTTP call with `respx`, validate schema output
- `test_enricher.py` — mock both OpenAI and `lookup_company`, test routing edge cases
- `test_api.py` — integration tests hitting FastAPI endpoints with mocked dependencies
