import uuid
from datetime import datetime, timezone

from openai import OpenAI

from .schemas import CallAnalysis

SYSTEM_PROMPT = """You are a sales intelligence analyst.
Extract structured data from the sales call transcript below.
Be precise and conservative — only report what is explicitly stated or strongly implied.
If something is unclear or not mentioned, use null rather than guessing."""


def _build_response_format() -> dict:
    schema = CallAnalysis.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "CallAnalysis",
            "strict": True,
            "schema": schema,
        },
    }


def extract_from_transcript(
    transcript: str,
    call_id: str | None = None,
    client: OpenAI | None = None,
    model: str = "openai/gpt-4o",
) -> CallAnalysis:
    if client is None:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        response_format=_build_response_format(),
    )

    raw = response.choices[0].message.content
    analysis = CallAnalysis.model_validate_json(raw)

    # Overwrite fields set by the system, not extracted from the transcript
    analysis.call_id = call_id or str(uuid.uuid4())
    analysis.timestamp = datetime.now(tz=timezone.utc)

    return analysis
