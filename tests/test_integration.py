"""
Integration tests for the sales call agent pipeline.

Requires: docker-compose up --build (both `agent` on :8000 and `n8n` on :5678)

NOTE on n8n: We test http://localhost:8000/analyze/transcript directly rather than
the n8n webhook because:
  - /webhook-test/ requires a manual UI click ("Execute workflow") — not automatable
  - The production webhook needs active=true + live Slack/Gmail/HubSpot credentials
  - The FastAPI pipeline is the meaningful agent logic; n8n is routing glue on top of it
"""
import re
import httpx
import pytest
from pathlib import Path

AGENT_URL = "http://localhost:8000"
FIXTURES = Path(__file__).parent / "fixtures"


def _load_transcripts() -> list[tuple[str, str]]:
    """Return list of (label, transcript_text) parsed from sample_transcript.txt."""
    raw = (FIXTURES / "sample_transcript.txt").read_text()
    chunks = re.split(r"\*\*Transcript \d+:", raw)
    results = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        first_newline = chunk.find("\n")
        label = chunk[:first_newline].strip(" *") if first_newline != -1 else "transcript"
        results.append((label, chunk))
    return results


@pytest.fixture(scope="session")
def agent_running():
    try:
        httpx.get(f"{AGENT_URL}/docs", timeout=5)
    except Exception:
        pytest.skip("Agent not running — start with: docker-compose up --build")


@pytest.mark.integration
class TestTranscriptPipelineCoverage:
    def test_all_transcripts_return_valid_result(self, agent_running):
        for label, transcript in _load_transcripts():
            resp = httpx.post(
                f"{AGENT_URL}/analyze/transcript",
                json={"transcript": transcript, "call_id": f"fixture-{label[:20]}"},
                timeout=120,
            )
            assert resp.status_code == 200, f"[{label}] failed: {resp.text}"
            body = resp.json()
            assert "routing" in body
            assert body["routing"]["priority"] in ("Low", "Medium", "High", "Urgent"), \
                f"[{label}] unexpected priority: {body['routing']['priority']}"
            assert "call_analysis" in body
            assert "company_profile" in body
            assert "follow_up_email" in body
            assert "slack_summary" in body

    def test_high_urgent_branch_covered(self, agent_running):
        """Transcript 1 (positive discovery, qualified buyer) should produce High or Urgent."""
        label, transcript = _load_transcripts()[0]
        resp = httpx.post(
            f"{AGENT_URL}/analyze/transcript",
            json={"transcript": transcript, "call_id": "fixture-branch-high"},
            timeout=120,
        )
        assert resp.status_code == 200
        priority = resp.json()["routing"]["priority"]
        assert priority in ("High", "Urgent"), \
            f"Expected High/Urgent for positive discovery, got {priority}"

    def test_low_medium_branch_covered(self, agent_running):
        """Transcripts 2 and 3 (stalled/near-lost) should include at least one Low or Medium."""
        low_medium_found = False
        for label, transcript in _load_transcripts()[1:]:
            resp = httpx.post(
                f"{AGENT_URL}/analyze/transcript",
                json={"transcript": transcript, "call_id": f"fixture-branch-low-{label[:10]}"},
                timeout=120,
            )
            assert resp.status_code == 200
            if resp.json()["routing"]["priority"] in ("Low", "Medium"):
                low_medium_found = True
                break
        assert low_medium_found, \
            "No Low/Medium priority found across stalled/near-lost transcripts — both n8n branches must be covered"
