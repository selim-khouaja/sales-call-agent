import json
import re
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from .schemas import CallAnalysis, CompanyProfile, RoutingDecision, EnrichedCallResult
from .tools import TOOLS, TOOL_MAP
from .follow_up import generate_follow_up, generate_slack_summary


# Wrapper model for a single structured extraction call
class _EnrichmentResult(BaseModel):
    company_profile: CompanyProfile
    routing: RoutingDecision


def _build_extraction_format() -> dict:
    schema = _EnrichmentResult.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "EnrichmentResult",
            "strict": True,
            "schema": schema,
        },
    }


SYSTEM_PROMPT = """You are a senior sales operations analyst.
Given a structured call analysis, you will:
1. Call lookup_company to fetch external info about the prospect's company
2. Score their ICP fit (0–1) with clear reasoning based on company size, industry, and deal signals
3. Decide on routing priority (Low/Medium/High/Urgent) and whether to escalate to a manager
4. Suggest a concrete next action with a specific timeframe

Be decisive. Urgent = active negotiation with strong buy signals. High = clear need, good fit, near-term timeline."""


def enrich_and_route(
    analysis: CallAnalysis,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> EnrichedCallResult:
    if client is None:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if model is None:
        import os
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Analyse this call and enrich it:\n{analysis.model_dump_json(indent=2)}",
        },
    ]

    # Agent loop
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        # Append assistant message with tool calls
        messages.append(msg)

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[tc.function.name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # Structured extraction: parse CompanyProfile + RoutingDecision from agent's reasoning
    extraction_messages = [
        {
            "role": "user",
            "content": (
                f"Based on this reasoning, extract structured CompanyProfile and RoutingDecision.\n\n"
                f"Original call analysis:\n{analysis.model_dump_json(indent=2)}\n\n"
                f"Agent reasoning:\n{msg.content or '(no final message)'}"
            ),
        }
    ]
    extraction_response = client.chat.completions.create(
        model=model,
        messages=extraction_messages,
        response_format=_build_extraction_format(),
        max_tokens=4096,
    )
    raw = extraction_response.choices[0].message.content
    # Strip markdown code fences if the model wrapped the JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        raw = match.group(1).strip()
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
    enrichment = _EnrichmentResult.model_validate_json(raw)

    return EnrichedCallResult(
        call_analysis=analysis,
        company_profile=enrichment.company_profile,
        routing=enrichment.routing,
        follow_up_email=generate_follow_up(analysis, enrichment.company_profile, client=client, model=model),
        slack_summary=generate_slack_summary(analysis, enrichment.routing, client=client, model=model),
    )
