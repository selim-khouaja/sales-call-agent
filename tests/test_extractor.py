import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from agent.extractor import extract_from_transcript
from agent.schemas import CallAnalysis

_LLM_RESPONSE = {
    "call_id": "llm-generated-id",  # overwritten by the function
    "timestamp": "2020-01-01T00:00:00Z",  # overwritten by the function
    "contact_name": "Nina Patel",
    "contact_title": "Head of Sales",
    "company_name": "Acme Corp",
    "deal_stage": "Discovery",
    "sentiment": "Positive",
    "budget_signal": "$50k",
    "authority_signal": None,
    "need": "Reduce manual data entry",
    "timeline": "End of Q1",
    "objections": [],
    "next_steps": ["Demo next Tuesday"],
    "summary": "Discovery call with Acme Corp.",
    "confidence": 0.85,
}

_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _api_response(content: str) -> dict:
    """Minimal OpenAI-compatible chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "openai/gpt-oss-20b:free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake")


@respx.mock
def test_returns_call_analysis():
    respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    result = extract_from_transcript("some transcript")
    assert isinstance(result, CallAnalysis)


@respx.mock
def test_fields_parsed_correctly():
    respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    result = extract_from_transcript("some transcript")
    assert result.contact_name == "Nina Patel"
    assert result.company_name == "Acme Corp"
    assert result.deal_stage == "Discovery"
    assert result.sentiment == "Positive"
    assert result.need == "Reduce manual data entry"
    assert result.next_steps == ["Demo next Tuesday"]
    assert result.confidence == 0.85


@respx.mock
def test_provided_call_id_is_used():
    respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    result = extract_from_transcript("some transcript", call_id="my-id")
    assert result.call_id == "my-id"


@respx.mock
def test_call_id_generated_when_not_provided():
    respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    result = extract_from_transcript("some transcript")
    assert result.call_id
    assert result.call_id != "llm-generated-id"


@respx.mock
def test_timestamp_is_current():
    respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    before = datetime.now(tz=timezone.utc)
    result = extract_from_transcript("some transcript")
    after = datetime.now(tz=timezone.utc)
    assert before <= result.timestamp <= after


@respx.mock
def test_transcript_sent_to_llm():
    route = respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    extract_from_transcript("the actual transcript text")
    request_body = json.loads(route.calls[0].request.content)
    user_msg = next(m for m in request_body["messages"] if m["role"] == "user")
    assert "the actual transcript text" in user_msg["content"]


@respx.mock
def test_response_format_uses_json_schema():
    route = respx.post(_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_api_response(json.dumps(_LLM_RESPONSE)))
    )
    extract_from_transcript("some transcript")
    request_body = json.loads(route.calls[0].request.content)
    fmt = request_body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "CallAnalysis"
