import re
import uuid
from datetime import datetime, timezone

from openai import OpenAI

from .schemas import CallAnalysis
from ._schema_utils import make_strict_schema


def _clean_json(raw: str) -> str:
    """Strip markdown code fences and surrounding text — some models wrap JSON in ```json ... ```."""
    # Try to extract from code fence first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        return match.group(1).strip()
    # Fall back: find the first { and last } and extract that substring
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return raw[start : end + 1]
    return raw.strip()

SYSTEM_PROMPT = """You are a sales intelligence analyst.
Extract structured data from the sales call transcript below.
Be precise and conservative — only report what is explicitly stated or strongly implied.
If something is unclear or not mentioned, use null rather than guessing."""


def _build_response_format() -> dict:
    schema = make_strict_schema(CallAnalysis.model_json_schema())
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
    model: str | None = None,
    max_retries: int = 3,
) -> CallAnalysis:
    if client is None:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if model is None:
        import os
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

    last_exc: Exception | None = None
    for _ in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            response_format=_build_response_format(),
            max_tokens=4096,
        )
        try:
            raw = _clean_json(response.choices[0].message.content)
            analysis = CallAnalysis.model_validate_json(raw)
            break
        except Exception as exc:
            last_exc = exc
            continue
    else:
        raise ValueError(f"Failed to parse valid CallAnalysis after {max_retries} attempts") from last_exc

    # Overwrite fields set by the system, not extracted from the transcript
    analysis.call_id = call_id or str(uuid.uuid4())
    analysis.timestamp = datetime.now(tz=timezone.utc)

    return analysis
