from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class AlertLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class NewsSource(BaseModel):
    url: str
    title: str
    domain: str
    is_trusted: bool
    published_date: Optional[str] = None

class VerificationResult(BaseModel):
    is_verified: bool
    confidence_score: float
    sources: List[NewsSource]
    contradicting_sources: List[NewsSource]
    summary: str
    verification_time: float  # in seconds

class ViralPrediction(BaseModel):
    will_go_viral: bool
    probability: float
    estimated_reach: int
    time_to_viral: Optional[float] = None  # hours
    risk_factors: List[str]

class CounterNarrative(BaseModel):
    narrative: str
    citations: List[str]
    target_platforms: List[str]
    urgency: AlertLevel

class AgentAction(BaseModel):
    action_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: str
    status: str  # "INITIATED", "IN_PROGRESS", "COMPLETED", "FAILED"

class NewsAnalysis(BaseModel):
    news_id: str
    headline: str
    content: str
    source_url: Optional[str] = None
    
    # Analysis Results
    falsehood_score: float
    alert_level: AlertLevel
    verification: VerificationResult
    viral_prediction: ViralPrediction
    
    # Agent Actions
    actions_taken: List[AgentAction]
    counter_narrative: Optional[CounterNarrative] = None
    
    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.now)
    processing_time: float  # Total time in seconds
    
    # Government Protocol
    requires_approval: bool = False
    approved_by: Optional[str] = None
    deployed: bool = False

class CrisisSimulationRequest(BaseModel):
    scenario: str  # "cyberattack", "riot", "earthquake", "custom"
    custom_headline: Optional[str] = None
    custom_content: Optional[str] = None

class AnalysisRequest(BaseModel):
    headline: str
    content: str
    source_url: Optional[str] = None
    enable_counter_narrative: bool = True
    simulate_crisis: bool = False

class MultilingualSummary(BaseModel):
    language: str  # Language code (en, hi, mr, etc.)
    language_name: str  # Human-readable name
    headline: str
    summary: str
    content_snippet: str

class SummaryResponse(BaseModel):
    news_id: str
    original_headline: str
    detected_language: str
    summaries: Dict[str, MultilingualSummary]
    generation_time: float