import time

from dotenv import load_dotenv
load_dotenv()  # load .env before any module reads os.environ

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
    start = time.perf_counter()
    try:
        analysis = extract_from_transcript(req.transcript, req.call_id)
        result = enrich_and_route(analysis)
    except Exception as exc:
        log.error("transcript_failed", call_id=req.call_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    duration_ms = round((time.perf_counter() - start) * 1000)
    log.info(
        "transcript_processed",
        call_id=analysis.call_id,
        priority=result.routing.priority,
        duration_ms=duration_ms,
    )
    return result


@app.post("/analyze/webhook", response_model=EnrichedCallResult)
async def analyze_webhook(analysis: CallAnalysis):
    """Pipeline B only: Attention-style structured payload in, enriched result out."""
    log.info("webhook_received", call_id=analysis.call_id)
    start = time.perf_counter()
    try:
        result = enrich_and_route(analysis)
    except Exception as exc:
        log.error("webhook_failed", call_id=analysis.call_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    duration_ms = round((time.perf_counter() - start) * 1000)
    log.info(
        "webhook_processed",
        call_id=analysis.call_id,
        priority=result.routing.priority,
        duration_ms=duration_ms,
    )
    return result
