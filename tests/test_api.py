from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent.schemas import CallAnalysis, CompanyProfile, EnrichedCallResult, RoutingDecision
from api.main import app

_client = TestClient(app)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_analysis(**overrides) -> CallAnalysis:
    defaults = {
        "call_id": "test-call-id",
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


def _make_enriched_result(analysis: CallAnalysis | None = None) -> EnrichedCallResult:
    if analysis is None:
        analysis = _make_analysis()
    return EnrichedCallResult(
        call_analysis=analysis,
        company_profile=CompanyProfile(
            name="Acme Corp",
            industry="Technology",
            employee_count=200,
            description="A tech company",
            icp_score=0.8,
            icp_reasoning="Good size and industry fit",
        ),
        routing=RoutingDecision(
            priority="High",
            escalate_to_manager=False,
            reasoning="Clear need and near-term timeline",
            suggested_action="Book demo within 48h",
        ),
        follow_up_email="Hi Nina, looking forward to the demo on Tuesday.",
        slack_summary="*Acme Corp* — High priority. Book demo within 48h.",
    )


# ── POST /analyze/transcript ──────────────────────────────────────────────────

def test_transcript_endpoint_returns_200():
    analysis = _make_analysis()
    result = _make_enriched_result(analysis)
    with (
        patch("api.main.extract_from_transcript", return_value=analysis),
        patch("api.main.enrich_and_route", return_value=result),
    ):
        resp = _client.post("/analyze/transcript", json={"transcript": "Hello world"})
    assert resp.status_code == 200


def test_transcript_endpoint_returns_enriched_result():
    analysis = _make_analysis()
    result = _make_enriched_result(analysis)
    with (
        patch("api.main.extract_from_transcript", return_value=analysis),
        patch("api.main.enrich_and_route", return_value=result),
    ):
        resp = _client.post("/analyze/transcript", json={"transcript": "Hello world"})
    body = resp.json()
    assert body["routing"]["priority"] == "High"
    assert body["call_analysis"]["company_name"] == "Acme Corp"
    assert body["follow_up_email"] == "Hi Nina, looking forward to the demo on Tuesday."


def test_transcript_endpoint_passes_call_id():
    analysis = _make_analysis(call_id="custom-id")
    result = _make_enriched_result(analysis)
    with (
        patch("api.main.extract_from_transcript", return_value=analysis) as mock_extract,
        patch("api.main.enrich_and_route", return_value=result),
    ):
        _client.post(
            "/analyze/transcript",
            json={"transcript": "Hello world", "call_id": "custom-id"},
        )
    mock_extract.assert_called_once_with("Hello world", "custom-id")


def test_transcript_endpoint_returns_500_on_error():
    with (
        patch("api.main.extract_from_transcript", side_effect=RuntimeError("LLM failure")),
    ):
        resp = _client.post("/analyze/transcript", json={"transcript": "Hello"})
    assert resp.status_code == 500


# ── POST /analyze/webhook ─────────────────────────────────────────────────────

def test_webhook_endpoint_returns_200():
    analysis = _make_analysis()
    result = _make_enriched_result(analysis)
    with patch("api.main.enrich_and_route", return_value=result):
        resp = _client.post("/analyze/webhook", json=analysis.model_dump(mode="json"))
    assert resp.status_code == 200


def test_webhook_endpoint_returns_enriched_result():
    analysis = _make_analysis()
    result = _make_enriched_result(analysis)
    with patch("api.main.enrich_and_route", return_value=result):
        resp = _client.post("/analyze/webhook", json=analysis.model_dump(mode="json"))
    body = resp.json()
    assert body["routing"]["priority"] == "High"
    assert body["company_profile"]["icp_score"] == 0.8


def test_webhook_endpoint_passes_analysis_to_enricher():
    analysis = _make_analysis(deal_stage="Negotiation", sentiment="Positive")
    result = _make_enriched_result(analysis)
    with patch("api.main.enrich_and_route", return_value=result) as mock_enrich:
        _client.post("/analyze/webhook", json=analysis.model_dump(mode="json"))
    called_analysis = mock_enrich.call_args[0][0]
    assert called_analysis.deal_stage == "Negotiation"
    assert called_analysis.sentiment == "Positive"


def test_webhook_endpoint_returns_500_on_error():
    analysis = _make_analysis()
    with patch("api.main.enrich_and_route", side_effect=RuntimeError("enricher failure")):
        resp = _client.post("/analyze/webhook", json=analysis.model_dump(mode="json"))
    assert resp.status_code == 500


def test_webhook_rejects_invalid_payload():
    resp = _client.post("/analyze/webhook", json={"not": "a call analysis"})
    assert resp.status_code == 422
