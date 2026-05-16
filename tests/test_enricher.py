import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.enricher import enrich_and_route
from agent.schemas import CallAnalysis, EnrichedCallResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_analysis(**overrides) -> CallAnalysis:
    defaults = {
        "call_id": "test-id",
        "timestamp": datetime.now(tz=timezone.utc),
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
    return CallAnalysis(**{**defaults, **overrides})


def _enrichment_json(priority: str = "High", escalate: bool = False, icp_score: float = 0.8) -> str:
    return json.dumps({
        "company_profile": {
            "name": "Acme Corp",
            "industry": "Technology",
            "employee_count": 200,
            "description": "A tech company",
            "icp_score": icp_score,
            "icp_reasoning": "Good fit based on size and need",
        },
        "routing": {
            "priority": priority,
            "escalate_to_manager": escalate,
            "reasoning": "Test routing reasoning",
            "suggested_action": "Book demo within 48h",
        },
    })


_EMAIL_JSON = json.dumps({"email_body": "Hi Nina, looking forward to the demo on Tuesday."})
_SLACK_JSON = json.dumps({"slack_message": "*Acme Corp* — High priority. Book demo within 48h."})


def _make_response(content: str, tool_calls=None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def _make_tool_call(name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = "tc_abc"
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mock_client(*responses) -> MagicMock:
    """Client that returns each response in order on successive create() calls."""
    client = MagicMock()
    client.chat.completions.create.side_effect = list(responses)
    return client


# ── Tests: basic contract ─────────────────────────────────────────────────────

def test_returns_enriched_call_result():
    analysis = _make_analysis()
    client = _mock_client(
        _make_response("reasoning text"),           # agent loop — no tool calls
        _make_response(_enrichment_json()),         # structured extraction
        _make_response(_EMAIL_JSON),               # follow-up email
        _make_response(_SLACK_JSON),               # slack summary
    )
    result = enrich_and_route(analysis, client=client)
    assert isinstance(result, EnrichedCallResult)


def test_call_analysis_preserved():
    analysis = _make_analysis()
    client = _mock_client(
        _make_response("reasoning"),
        _make_response(_enrichment_json()),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    result = enrich_and_route(analysis, client=client)
    assert result.call_analysis == analysis


def test_follow_up_and_slack_populated():
    analysis = _make_analysis()
    client = _mock_client(
        _make_response("reasoning"),
        _make_response(_enrichment_json()),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    result = enrich_and_route(analysis, client=client)
    assert result.follow_up_email == "Hi Nina, looking forward to the demo on Tuesday."
    assert "Acme Corp" in result.slack_summary


# ── Tests: agent loop tool execution ─────────────────────────────────────────

def test_tool_call_is_executed():
    """LLM requests lookup_company → tool is called → result fed back → loop continues."""
    analysis = _make_analysis()
    tool_call = _make_tool_call("lookup_company", {"company_name": "Acme Corp"})
    fake_lookup = MagicMock(return_value={"name": "Acme Corp", "domain": "acme.com", "description": "Tech"})

    client = _mock_client(
        _make_response("Let me look this up", tool_calls=[tool_call]),  # loop iter 1: tool call
        _make_response("Final reasoning"),                               # loop iter 2: done
        _make_response(_enrichment_json()),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    with patch("agent.enricher.TOOL_MAP", {"lookup_company": fake_lookup}):
        result = enrich_and_route(analysis, client=client)

    fake_lookup.assert_called_once_with(company_name="Acme Corp")
    assert client.chat.completions.create.call_count == 5
    assert isinstance(result, EnrichedCallResult)


def test_tool_result_appended_to_messages():
    """Tool result must appear as a 'tool' role message in the next LLM call."""
    analysis = _make_analysis()
    tool_call = _make_tool_call("lookup_company", {"company_name": "Acme Corp"})
    lookup_result = {"name": "Acme Corp", "domain": "acme.com", "description": "Tech"}

    client = _mock_client(
        _make_response("Looking up...", tool_calls=[tool_call]),
        _make_response("Done reasoning"),
        _make_response(_enrichment_json()),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    with patch("agent.enricher.TOOL_MAP", {"lookup_company": MagicMock(return_value=lookup_result)}):
        enrich_and_route(analysis, client=client)

    # Second agent loop call should include the tool result message
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"]) == lookup_result


# ── Tests: routing edge cases ─────────────────────────────────────────────────

def test_urgent_priority_when_negotiation_positive():
    analysis = _make_analysis(deal_stage="Negotiation", sentiment="Positive")
    client = _mock_client(
        _make_response("reasoning"),
        _make_response(_enrichment_json(priority="Urgent", escalate=True)),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    result = enrich_and_route(analysis, client=client)
    assert result.routing.priority == "Urgent"
    assert result.routing.escalate_to_manager is True


def test_low_priority_negative_sentiment():
    analysis = _make_analysis(deal_stage="Discovery", sentiment="Negative")
    client = _mock_client(
        _make_response("reasoning"),
        _make_response(_enrichment_json(priority="Low", escalate=False, icp_score=0.2)),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    result = enrich_and_route(analysis, client=client)
    assert result.routing.priority == "Low"
    assert result.routing.escalate_to_manager is False
    assert result.company_profile.icp_score == 0.2


def test_icp_score_bounds():
    analysis = _make_analysis()
    client = _mock_client(
        _make_response("reasoning"),
        _make_response(_enrichment_json(icp_score=0.95)),
        _make_response(_EMAIL_JSON),
        _make_response(_SLACK_JSON),
    )
    result = enrich_and_route(analysis, client=client)
    assert 0.0 <= result.company_profile.icp_score <= 1.0
