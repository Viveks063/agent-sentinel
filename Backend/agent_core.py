import time
from typing import List, Optional
from datetime import datetime
from config import config
from models import (
    NewsAnalysis, AlertLevel, AgentAction, CounterNarrative,
    VerificationResult, ViralPrediction
)
from semantic_verifier import verifier
from citation_engine import citation_engine
from viral_predictor import viral_predictor
from gdelt_monitor import gdelt_monitor

class AgentSentinelCore:
    """
    THE GENIUS: The autonomous AI agent that orchestrates everything
    """
    
    def __init__(self):
        self.action_log: List[AgentAction] = []
    
    def _log_action(self, action_type: str, details: str, status: str = "COMPLETED"):
        """Log agent actions for transparency"""
        action = AgentAction(
            action_type=action_type,
            details=details,
            status=status
        )
        self.action_log.append(action)
        return action
    
    def _calculate_falsehood_score(self, 
                                   verification: VerificationResult,
                                   viral_prediction: ViralPrediction,
                                   gdelt_coverage: dict) -> float:
        """
        THE GENIUS ALGORITHM: Multi-factor falsehood scoring
        FIXED VERSION: Proper score calculation
        """
        
        # Base score from verification
        if verification.is_verified and len(verification.sources) > 0:
            # Strong verification = LOW falsehood score
            base_score = 0.1 + (0.3 * (1.0 - verification.confidence_score))
        elif len(verification.contradicting_sources) > 0:
            # Contradicted by trusted sources = HIGH falsehood score
            base_score = 0.7 + (0.2 * verification.confidence_score)
        else:
            # No verification either way = MEDIUM-HIGH score
            base_score = 0.75
        
        # GDELT factor - FIXED
        gdelt_factor = 0.0
        if gdelt_coverage.get("has_coverage", False):
            coverage_ratio = gdelt_coverage.get("coverage_ratio", 0)
            if coverage_ratio > 0.7:
                # High trusted coverage = LOWER falsehood (it's real news)
                gdelt_factor = -0.15
            elif coverage_ratio < 0.3:
                # Low trusted coverage = suspicious
                gdelt_factor = 0.1
        else:
            # No GDELT coverage for "breaking news" = very suspicious
            if "breaking" in verification.sources[0].title.lower() if verification.sources else False:
                gdelt_factor = 0.2
        
        # Viral prediction factor
        viral_factor = viral_prediction.probability * 0.15
        
        # Combine factors - FIXED BOUNDS
        total_score = base_score + gdelt_factor + viral_factor
        total_score = max(0.0, min(1.0, total_score))  # Clamp to [0, 1]
        
        return total_score
    
    def _determine_alert_level(self, falsehood_score: float) -> AlertLevel:
        """Determine alert level based on falsehood score"""
        if falsehood_score >= 0.9:
            return AlertLevel.CRITICAL
        elif falsehood_score >= 0.75:
            return AlertLevel.HIGH
        elif falsehood_score >= 0.5:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    async def _generate_counter_narrative(self,
                                         headline: str,
                                         content: str,
                                         verification: VerificationResult,
                                         alert_level: AlertLevel) -> Optional[CounterNarrative]:
        """Generate counter-narratives with citations"""
        
        if alert_level in [AlertLevel.LOW, AlertLevel.MEDIUM]:
            return None
        
        # Generate narrative
        if len(verification.contradicting_sources) > 0:
            narrative = f"OFFICIAL STATEMENT: The claim '{headline}' has been fact-checked and found to be FALSE.\n\n"
            narrative += f"Verification: {verification.summary}\n\n"
        else:
            narrative = f"ADVISORY: The claim '{headline}' cannot be verified through trusted sources.\n\n"
        
        citations = citation_engine.generate_citations(verification)
        platforms = ["Twitter/X", "Facebook", "WhatsApp"]
        
        return CounterNarrative(
            narrative=narrative,
            citations=citations,
            target_platforms=platforms,
            urgency=alert_level
        )
    
    async def analyze_news(self,
                          headline: str,
                          content: str,
                          source_url: Optional[str] = None,
                          enable_counter_narrative: bool = True,
                          news_id: Optional[str] = None) -> NewsAnalysis:
        """
        THE GENIUS PIPELINE - FIXED VERSION
        """
        
        start_time = time.time()
        self.action_log = []
        
        if not news_id:
            news_id = f"news_{int(time.time())}"
        
        self._log_action("ANALYSIS_START", f"Analyzing: {headline[:50]}...", "IN_PROGRESS")
        
        # Step 1: Semantic Verification
        self._log_action("SEMANTIC_VERIFICATION", "Initiating verification...", "IN_PROGRESS")
        try:
            verification = await verifier.verify_claim(headline, content)
            self._log_action("SEMANTIC_VERIFICATION", 
                            f"Complete: {verification.summary}", "COMPLETED")
        except Exception as e:
            print(f"Verification error: {e}")
            # Fallback verification result
            from models import NewsSource
            verification = VerificationResult(
                is_verified=False,
                confidence_score=0.0,
                sources=[],
                contradicting_sources=[],
                summary="Verification failed",
                verification_time=0.0
            )
        
        # Step 2: GDELT Check
        self._log_action("GDELT_CHECK", "Querying GDELT...", "IN_PROGRESS")
        try:
            gdelt_coverage = await gdelt_monitor.check_event_coverage(headline)
            self._log_action("GDELT_CHECK", 
                            f"Found {gdelt_coverage['total_articles']} articles", 
                            "COMPLETED")
        except Exception as e:
            print(f"GDELT error: {e}")
            gdelt_coverage = {
                "has_coverage": False,
                "total_articles": 0,
                "trusted_articles": 0,
                "coverage_ratio": 0
            }
        
        # Step 3: Viral Prediction
        self._log_action("VIRAL_PREDICTION", "Analyzing viral potential...", "IN_PROGRESS")
        text = f"{headline} {content}".lower()
        emotional_words = [word for word in text.split() 
                          if word in ['urgent', 'breaking', 'shocking', 'alert', 'warning',
                                     'crisis', 'attack', 'death', 'riot', 'emergency']]
        
        viral_prediction = viral_predictor.calculate_viral_probability(
            falsehood_score=0.5,
            current_reach=100,
            emotional_trigger_words=emotional_words,
            has_multimedia=False,
            source_credibility=0.7 if verification.is_verified else 0.3
        )
        self._log_action("VIRAL_PREDICTION", 
                        f"Viral probability: {viral_prediction.probability:.2%}", 
                        "COMPLETED")
        
        # Step 4: FIXED Falsehood Score Calculation
        self._log_action("FALSEHOOD_SCORING", "Computing threat score...", "IN_PROGRESS")
        falsehood_score = self._calculate_falsehood_score(
            verification, viral_prediction, gdelt_coverage
        )
        alert_level = self._determine_alert_level(falsehood_score)
        self._log_action("FALSEHOOD_SCORING", 
                        f"Score: {falsehood_score:.3f} | Level: {alert_level.value}", 
                        "COMPLETED")
        
        # Step 5: Counter-Narrative
        counter_narrative = None
        if enable_counter_narrative and alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            self._log_action("COUNTER_NARRATIVE", "Generating response...", "IN_PROGRESS")
            counter_narrative = await self._generate_counter_narrative(
                headline, content, verification, alert_level
            )
            self._log_action("COUNTER_NARRATIVE", "Response prepared", "COMPLETED")
        
        requires_approval = alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]
        processing_time = time.time() - start_time
        
        self._log_action("ANALYSIS_COMPLETE", 
                        f"Processing time: {processing_time:.2f}s", 
                        "COMPLETED")
        
        return NewsAnalysis(
            news_id=news_id,
            headline=headline,
            content=content,
            source_url=source_url,
            falsehood_score=falsehood_score,
            alert_level=alert_level,
            verification=verification,
            viral_prediction=viral_prediction,
            actions_taken=self.action_log.copy(),
            counter_narrative=counter_narrative,
            processing_time=processing_time,
            requires_approval=requires_approval
        )

# Singleton
agent_core = AgentSentinelCore()