import math
from typing import List
from models import ViralPrediction

class ViralPredictionEngine:
    """
    THE GENIUS: Predict if fake news will go viral BEFORE it spreads
    This enables PRE-BUNKING instead of DE-BUNKING
    """
    
    @staticmethod
    def calculate_viral_probability(
        falsehood_score: float,
        current_reach: int = 0,
        emotional_trigger_words: List[str] = None,
        has_multimedia: bool = False,
        source_credibility: float = 0.5
    ) -> ViralPrediction:
        """
        THE GENIUS ALGORITHM: Multi-factor viral prediction
        
        Factors:
        1. Falsehood score (higher = more sensational = more viral)
        2. Emotional triggers (fear, anger, urgency)
        3. Multimedia presence (images/videos spread faster)
        4. Source credibility (low credibility can still go viral)
        5. Current reach trajectory
        """
        
        # Base probability from falsehood score
        # THE GENIUS INSIGHT: False news spreads 6x faster (MIT study)
        base_probability = falsehood_score * 0.7
        
        # Emotional trigger bonus
        emotional_bonus = 0.0
        if emotional_trigger_words:
            high_emotion_words = [
                'urgent', 'breaking', 'shocking', 'alert', 'warning',
                'crisis', 'attack', 'death', 'riot', 'emergency',
                'exclusive', 'leaked', 'revealed', 'exposed'
            ]
            matches = sum(1 for word in emotional_trigger_words 
                         if word.lower() in high_emotion_words)
            emotional_bonus = min(0.2, matches * 0.05)
        
        # Multimedia bonus
        multimedia_bonus = 0.1 if has_multimedia else 0.0
        
        # Source credibility penalty (ironically, low credibility spreads faster)
        credibility_factor = 1.0 - (source_credibility * 0.2)
        
        # Calculate total probability
        total_probability = (base_probability + emotional_bonus + multimedia_bonus) * credibility_factor
        total_probability = max(0.0, min(1.0, total_probability))
        
        # Estimate reach
        if total_probability > 0.7:
            # Exponential spread model
            estimated_reach = int(current_reach * math.exp(total_probability * 5))
            estimated_reach = max(10000, estimated_reach)
            time_to_viral = 2.0  # 2 hours
        elif total_probability > 0.5:
            estimated_reach = int(current_reach * math.exp(total_probability * 3))
            estimated_reach = max(5000, estimated_reach)
            time_to_viral = 6.0  # 6 hours
        else:
            estimated_reach = int(current_reach * 1.5)
            time_to_viral = None
        
        # Risk factors
        risk_factors = []
        if falsehood_score > 0.8:
            risk_factors.append("High misinformation score")
        if emotional_bonus > 0.1:
            risk_factors.append("Strong emotional triggers detected")
        if has_multimedia:
            risk_factors.append("Contains multimedia (faster spread)")
        if source_credibility < 0.3:
            risk_factors.append("Low-credibility source")
        
        will_go_viral = total_probability > 0.7
        
        return ViralPrediction(
            will_go_viral=will_go_viral,
            probability=total_probability,
            estimated_reach=estimated_reach,
            time_to_viral=time_to_viral,
            risk_factors=risk_factors
        )

viral_predictor = ViralPredictionEngine()