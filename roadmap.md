This is the initial roadmap we have conceived to outline the different aspects of our project's design.
The design has two pipelines that compose:

**Pipeline A** — raw transcript → structured `CallAnalysis` (you build what Attention's AI does)  
**Pipeline B** — `CallAnalysis` → enriched, scored, routed `EnrichedCallResult` (you build what an FDE deploys on top)

n8n can trigger either A→B in sequence (from a raw transcript) or B alone (from a real Attention webhook). This makes the project composable and shows both skillsets cleanly.

---

### Repo structure

```
sales-call-agent/
├── pyproject.toml              # uv project file
├── uv.lock                     # lockfile — always commit
├── .python-version             # e.g. "3.12"
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   └── agent/
│       ├── schemas.py          # all Pydantic models (single source of truth)
│       ├── extractor.py        # Pipeline A: transcript → CallAnalysis
│       ├── enricher.py         # Pipeline B: CallAnalysis → EnrichedCallResult
│       ├── tools.py            # tool functions available to the agent loop
│       ├── follow_up.py        # email + summary generation
│       └── logger.py           # structured logging config
│
├── api/
│   └── main.py                 # FastAPI — two endpoints, one for each pipeline entry point
│
├── n8n/
│   ├── workflow_transcript.json   # Workflow 1: transcript webhook → A → B → outputs
│   └── workflow_attention.json    # Workflow 2: Attention-style webhook → B → outputs
│
├── retool/
│   └── dashboard.json          # exported Retool app
│
├── tests/
│   ├── test_extractor.py
│   ├── test_enricher.py
│   └── fixtures/
│       ├── sample_transcript.txt
│       └── sample_attention_webhook.json
│
└── README.md
```

---

## Roadmap

### Phase 0 — Project Setup (2 hours)

**Step 1: Initialise the repo with uv**

```bash
uv init sales-call-agent
cd sales-call-agent
uv python pin 3.12
uv add fastapi uvicorn openai pydantic structlog httpx
uv add --dev pytest pytest-asyncio respx
git init && git add . && git commit -m "chore: project scaffold"
```

`pyproject.toml` becomes your single dependency file. Push to GitHub immediately — commit history matters.

---

### Phase 1 — Schema Design (2 hours)

**Step 2: Define all Pydantic models before writing any logic**

This is the most important design decision in the project. Everything else — the LLM prompts, the API contracts, the n8n field mappings — derives from these shapes. Get them right first.

```python
# src/agent/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# ── Pipeline A output / Pipeline B input ──────────────────────────────
class CallAnalysis(BaseModel):
    call_id: str
    timestamp: datetime

    # Contact
    contact_name: str
    contact_title: Optional[str]
    company_name: str

    # Deal signals
    deal_stage: Literal["Discovery", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    budget_signal: Optional[str]        # what was said about budget
    authority_signal: Optional[str]     # who makes the decision
    need: str                           # core pain point stated
    timeline: Optional[str]

    objections: list[str]
    next_steps: list[str]
    summary: str                        # 2–3 sentence call summary
    confidence: float = Field(ge=0, le=1)  # how complete the transcript was

# ── Pipeline B output ─────────────────────────────────────────────────
class CompanyProfile(BaseModel):
    name: str
    industry: Optional[str]
    employee_count: Optional[int]
    description: Optional[str]
    icp_score: float = Field(ge=0, le=1)   # 0 = bad fit, 1 = perfect fit
    icp_reasoning: str

class RoutingDecision(BaseModel):
    priority: Literal["Low", "Medium", "High", "Urgent"]
    escalate_to_manager: bool
    reasoning: str
    suggested_action: str               # e.g. "Book executive call within 48h"

class EnrichedCallResult(BaseModel):
    call_analysis: CallAnalysis
    company_profile: CompanyProfile
    routing: RoutingDecision
    follow_up_email: str
    slack_summary: str                  # short card text for Slack
```

One schema file, imported everywhere. If you need to change a field, you change it once.

---

### Phase 2 — Pipeline A: Transcript Extractor (1 day)

**Step 3: Build the extractor using structured outputs**

One-shot LLM call — no agent loop needed here. The transcript is the full context; the LLM's job is purely to parse and classify, not to reason over multiple steps.

```python
# src/agent/extractor.py
from openai import OpenAI
from .schemas import CallAnalysis
import uuid
from datetime import datetime

client = OpenAI()

SYSTEM_PROMPT = """You are a sales intelligence analyst.
Extract structured data from the sales call transcript below.
Be precise and conservative — only report what is explicitly stated or strongly implied.
If something is unclear or not mentioned, use null rather than guessing."""

def extract_from_transcript(transcript: str, call_id: str | None = None) -> CallAnalysis:
    result = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        response_format=CallAnalysis,
    )
    analysis = result.choices[0].message.parsed
    # fill in generated fields not present in the transcript
    analysis.call_id = call_id or str(uuid.uuid4())
    analysis.timestamp = datetime.utcnow()
    return analysis
```

`client.beta.chat.completions.parse` with `response_format=CallAnalysis` guarantees a valid Pydantic object back — no JSON parsing, no `try/except` around `.get()` calls. This is the production pattern for structured extraction.

**Step 4: Write sample transcripts and test manually**

Write 2–3 realistic fake transcripts in `tests/fixtures/sample_transcript.txt` covering different scenarios: a positive discovery call, a stalled negotiation, a near-lost deal. Run the extractor against each and inspect the output. This is your smoke test before you write formal tests.

---

### Phase 3 — Pipeline B: Enricher & Router (1 day)

**Step 5: Build the company enrichment tool**

This is where the agent loop lives. The enricher needs to *look something up* before it can reason — so it uses tool calls rather than a one-shot prompt.

```python
# src/agent/tools.py
import httpx

def lookup_company(company_name: str) -> dict:
    """
    Fetch basic company info. Use Clearbit's free Autocomplete API in dev,
    swap for a paid enrichment API in production.
    """
    resp = httpx.get(
        "https://autocomplete.clearbit.com/v1/companies/suggest",
        params={"query": company_name},
        timeout=5.0
    )
    results = resp.json()
    if not results:
        return {"name": company_name, "domain": None, "industry": None, "description": None}
    top = results[0]
    return {
        "name": top.get("name"),
        "domain": top.get("domain"),
        "description": top.get("description"),
    }

def score_icp(company_info: dict, deal_analysis: dict) -> dict:
    """ICP scoring is done via LLM reasoning — this tool just structures the call."""
    return {**company_info, **deal_analysis}
```

**Step 6: Build the enricher as a proper agent loop**

```python
# src/agent/enricher.py
import json
from openai import OpenAI
from .schemas import CallAnalysis, CompanyProfile, RoutingDecision, EnrichedCallResult
from .tools import lookup_company
from .follow_up import generate_follow_up, generate_slack_summary

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_company",
            "description": "Look up company info (industry, size, description) by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"}
                },
                "required": ["company_name"]
            }
        }
    }
]

TOOL_MAP = {"lookup_company": lookup_company}

SYSTEM_PROMPT = """You are a senior sales operations analyst.
Given a structured call analysis, you will:
1. Look up the company to enrich your understanding
2. Score their ICP fit (0–1) with reasoning
3. Decide on routing priority and whether to escalate
4. Suggest a concrete next action

Be decisive — a clear Low/Medium/High/Urgent priority with a specific reason."""

def enrich_and_route(analysis: CallAnalysis) -> EnrichedCallResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyse this call:\n{analysis.model_dump_json(indent=2)}"}
    ]

    # Agent loop
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # LLM is done reasoning — parse its final structured response
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[tc.function.name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

    # Now extract structured outputs from the final reasoning
    company_profile = _parse_company_profile(msg.content, analysis)
    routing = _parse_routing(msg.content, analysis)

    return EnrichedCallResult(
        call_analysis=analysis,
        company_profile=company_profile,
        routing=routing,
        follow_up_email=generate_follow_up(analysis, company_profile),
        slack_summary=generate_slack_summary(analysis, routing),
    )
```

For `_parse_company_profile` and `_parse_routing`: do a second targeted structured output call on the agent's final message to extract those fields cleanly. Keeps parsing separate from reasoning.

**Step 7: Follow-up and Slack summary generators**

These are plain LLM calls, not agent loops — they take structured input and produce prose:

```python
# src/agent/follow_up.py

def generate_follow_up(analysis: CallAnalysis, company: CompanyProfile) -> str:
    prompt = f"""Write a follow-up email from the sales rep to {analysis.contact_name} at {analysis.company_name}.
    
Key context:
- Their main need: {analysis.need}
- Objections raised: {', '.join(analysis.objections) or 'none'}
- Agreed next steps: {', '.join(analysis.next_steps)}
- Their industry: {company.industry or 'unknown'}

Requirements: professional but warm, under 120 words, reference the specific next steps.
Do not write a subject line — just the body."""
    ...

def generate_slack_summary(analysis: CallAnalysis, routing: RoutingDecision) -> str:
    # Returns a short markdown string suitable for a Slack block
    ...
```

---

### Phase 4 — API Layer (half a day)

**Step 8: Two FastAPI endpoints**

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.extractor import extract_from_transcript
from agent.enricher import enrich_and_route
from agent.schemas import CallAnalysis, EnrichedCallResult
from agent.logger import get_logger

app = FastAPI(title="Sales Call Agent")
log = get_logger()

class TranscriptRequest(BaseModel):
    transcript: str
    call_id: str | None = None
    rep_name: str = "Sales Rep"

@app.post("/analyze/transcript", response_model=EnrichedCallResult)
async def analyze_transcript(req: TranscriptRequest):
    """Pipeline A → B: raw transcript in, fully enriched result out."""
    log.info("transcript_received", call_id=req.call_id)
    analysis = extract_from_transcript(req.transcript, req.call_id)
    result = enrich_and_route(analysis)
    log.info("transcript_processed", call_id=analysis.call_id, priority=result.routing.priority)
    return result

@app.post("/analyze/webhook", response_model=EnrichedCallResult)
async def analyze_webhook(analysis: CallAnalysis):
    """Pipeline B only: Attention-style structured payload in, enriched result out."""
    log.info("webhook_received", call_id=analysis.call_id)
    result = enrich_and_route(analysis)
    log.info("webhook_processed", call_id=analysis.call_id, priority=result.routing.priority)
    return result
```

Two endpoints, same output shape. n8n can call either depending on where the trigger comes from.

---

### Phase 5 — n8n Workflows (1 day)

**Step 9: Docker Compose with both services**

```yaml
# docker-compose.yml
services:
  agent:
    build: .
    ports: ["8000:8000"]
    env_file: .env

  n8n:
    image: n8nio/n8n:latest
    ports: ["5678:5678"]
    volumes: ["n8n_data:/home/node/.n8n"]
    environment:
      WEBHOOK_URL: "http://localhost:5678"

volumes:
  n8n_data:
```

**Step 10: Workflow 1 — Transcript workflow**

Nodes in order:
- **Webhook** — listens at `/call-ended-raw`, receives `{transcript, call_id, rep_name}`
- **HTTP Request** — POST to `http://agent:8000/analyze/transcript`
- **Switch** — branch on `routing.priority`: `Urgent/High` → one path, `Medium/Low` → another
- **Urgent/High path**: Slack message to `#deals-urgent` + HubSpot deal update + email to rep AND manager
- **Medium/Low path**: HubSpot deal update + email to rep only

**Step 11: Workflow 2 — Attention webhook workflow**

Same structure but the entry point is different:
- **Webhook** — listens at `/attention-call-ended`, receives a `CallAnalysis`-shaped payload
- **HTTP Request** — POST to `http://agent:8000/analyze/webhook`
- Same Switch + output branches as Workflow 1

Export both as JSON (`⋯` → Export) and commit them to `n8n/`. This is a key deliverable — someone can import your workflow JSON and run it.

---

### Phase 6 — Retool Dashboard (half a day)

**Step 12: Simple call review table**

Your FastAPI server should persist results to SQLite (add one `calls` table, write on every processed call). Retool connects to it directly via a REST query to your API, or you add a `GET /calls` endpoint.

The dashboard needs just two components to look professional:
- A filterable table (columns: date, company, deal stage, priority, ICP score, sentiment)
- A side panel that shows the follow-up email and full analysis when you click a row

This takes 2–3 hours in Retool's drag-and-drop UI. Export as JSON and commit to `retool/`.

---

### Phase 7 — Observability & Tests (half a day each)

**Step 13: Structured logging**

```python
# src/agent/logger.py
import structlog

def get_logger():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )
    return structlog.get_logger()
```

Every request emits a JSON log line with `call_id`, `pipeline`, `priority`, `tokens_used`, `duration_ms`. This is what "observable" means in the JD.

**Step 14: Tests**

Three test files:
- `test_extractor.py` — mocks the OpenAI call (use `respx` to mock the HTTP request), checks schema validation on the output
- `test_enricher.py` — mocks both OpenAI and `lookup_company`, verifies the routing logic for edge cases (e.g. Urgent only when `deal_stage == Negotiation` AND `sentiment == Positive`)
- `test_api.py` — integration test that hits the FastAPI endpoints end-to-end with mocked dependencies

---

### Phase 8 — README & Final Polish (2 hours)

**Step 15: README that demos in 5 commands**

```bash
git clone https://github.com/yourname/sales-call-agent
cd sales-call-agent
cp .env.example .env  # add your OPENAI_API_KEY
docker-compose up

# In another terminal:
curl -X POST http://localhost:8000/analyze/transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Alice: Hi James, thanks for joining..."}'
```

Open `localhost:5678` to see n8n. Open `localhost:8000/docs` to see the Swagger UI. That's the demo.

---

### Summary

| Phase | What you build | Days |
|---|---|---|
| 0 | Repo + uv + Docker scaffold | 0.25 |
| 1 | Schema design | 0.25 |
| 2 | Pipeline A: transcript extractor | 1 |
| 3 | Pipeline B: enricher + router + follow-up | 1 |
| 4 | FastAPI with two endpoints | 0.5 |
| 5 | Two n8n workflows | 1 |
| 6 | Retool dashboard | 0.5 |
| 7 | Observability + tests | 1 |
| 8 | README + polish | 0.25 |
| **Total** | | **~5.75 days** |