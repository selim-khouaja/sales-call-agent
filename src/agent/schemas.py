from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ── Pipeline A output / Pipeline B input ──────────────────────────────────────

class CallAnalysis(BaseModel):
    call_id: str
    timestamp: datetime

    # Contact
    contact_name: str
    contact_title: Optional[str] = None
    company_name: str

    # Deal signals
    deal_stage: Literal["Discovery", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    budget_signal: Optional[str] = None
    authority_signal: Optional[str] = None
    need: str
    timeline: Optional[str] = None

    objections: list[str]
    next_steps: list[str]
    summary: str
    confidence: float = Field(ge=0, le=1)


# ── Pipeline B output ──────────────────────────────────────────────────────────

class CompanyProfile(BaseModel):
    name: str
    industry: Optional[str] = None
    employee_count: Optional[int] = None
    description: Optional[str] = None
    icp_score: float = Field(ge=0, le=1)
    icp_reasoning: str


class RoutingDecision(BaseModel):
    priority: Literal["Low", "Medium", "High", "Urgent"]
    escalate_to_manager: bool
    reasoning: str
    suggested_action: str


class EnrichedCallResult(BaseModel):
    call_analysis: CallAnalysis
    company_profile: CompanyProfile
    routing: RoutingDecision
    follow_up_email: str
    slack_summary: str
