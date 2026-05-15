# Sales Call Agent

Sales call analysis pipeline that turns raw transcripts into enriched, scored, and routed call results.

## Quickstart

```bash
cp .env.example .env  # add your OPENAI_API_KEY
docker-compose up
```

- API + Swagger UI: http://localhost:8000/docs
- n8n workflows: http://localhost:5678

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze/transcript` | Pipeline A→B: raw transcript → enriched result |
| POST | `/analyze/webhook` | Pipeline B only: structured `CallAnalysis` → enriched result |

## Development

```bash
uv sync
uv run uvicorn api.main:app --reload
uv run pytest
```
