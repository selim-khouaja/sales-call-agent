from openai import OpenAI
from pydantic import BaseModel

from .schemas import CallAnalysis, CompanyProfile, RoutingDecision


class _EmailResult(BaseModel):
    email_body: str


class _SlackResult(BaseModel):
    slack_message: str


def generate_follow_up(
    analysis: CallAnalysis,
    company: CompanyProfile,
    client: OpenAI | None = None,
    model: str = "openai/gpt-4o",
) -> str:
    if client is None:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    objections = ", ".join(analysis.objections) if analysis.objections else "none"
    next_steps = ", ".join(analysis.next_steps) if analysis.next_steps else "to be confirmed"

    prompt = f"""Write a follow-up email body from the sales rep to {analysis.contact_name} at {analysis.company_name}.

Key context:
- Their main need: {analysis.need}
- Objections raised: {objections}
- Agreed next steps: {next_steps}
- Their industry: {company.industry or "unknown"}
- Deal stage: {analysis.deal_stage}

Requirements: professional but warm, under 120 words, reference the specific next steps agreed on. Do not write a subject line — just the body."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "EmailResult",
                "strict": True,
                "schema": _EmailResult.model_json_schema(),
            },
        },
    )
    return _EmailResult.model_validate_json(response.choices[0].message.content).email_body


def generate_slack_summary(
    analysis: CallAnalysis,
    routing: RoutingDecision,
    client: OpenAI | None = None,
    model: str = "openai/gpt-4o",
) -> str:
    if client is None:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    prompt = f"""Write a short Slack notification for a sales call summary. Use plain markdown (bold with *, bullet points with -).

Call details:
- Company: {analysis.company_name}
- Contact: {analysis.contact_name} ({analysis.contact_title or "unknown title"})
- Deal stage: {analysis.deal_stage}
- Sentiment: {analysis.sentiment}
- Priority: {routing.priority}
- Escalate to manager: {routing.escalate_to_manager}
- Suggested action: {routing.suggested_action}
- Call summary: {analysis.summary}

Keep it under 80 words. Include the priority and suggested action prominently."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SlackResult",
                "strict": True,
                "schema": _SlackResult.model_json_schema(),
            },
        },
    )
    return _SlackResult.model_validate_json(response.choices[0].message.content).slack_message
