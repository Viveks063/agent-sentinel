import random
from typing import Dict
from config import config
from models import NewsAnalysis, AlertLevel

class CrisisSimulator:
    """
    THE GENIUS: Simulate crisis scenarios for demo
    This is what wins hackathons - showing the system under stress
    """
    
    @staticmethod
    def get_scenario(scenario_name: str, custom_headline: str = None, 
                     custom_content: str = None) -> Dict:
        """
        Get or create crisis scenario
        """
        if scenario_name == "custom" and custom_headline and custom_content:
            return {
                "headline": custom_headline,
                "content": custom_content,
                "true_falsehood_score": 0.85,  # Default high score
                "sources": []
            }
        
        return config.CRISIS_SCENARIOS.get(scenario_name, 
                                           config.CRISIS_SCENARIOS["cyberattack"])
    
    @staticmethod
    def simulate_time_comparison() -> Dict:
        """
        THE GENIUS METRIC: Show time saved vs traditional methods
        """
        return {
            "traditional_method": {
                "method": "Manual verification by team",
                "time_seconds": config.TRADITIONAL_RESPONSE_TIME,
                "time_human": "48 hours",
                "steps": [
                    "1. Social media monitoring (2 hours)",
                    "2. Initial verification attempts (8 hours)",
                    "3. Cross-referencing sources (12 hours)",
                    "4. Legal review (6 hours)",
                    "5. Drafting response (8 hours)",
                    "6. Approval chain (12 hours)"
                ]
            },
            "sentinel_method": {
                "method": "Agent Sentinel Autonomous System",
                "time_seconds": config.SENTINEL_RESPONSE_TIME,
                "time_human": "1.5 seconds",
                "steps": [
                    "1. AI Agent detection (0.2s)",
                    "2. Semantic verification (0.8s)",
                    "3. Citation generation (0.3s)",
                    "4. Alert preparation (0.2s)"
                ]
            },
            "time_saved_seconds": config.TRADITIONAL_RESPONSE_TIME - config.SENTINEL_RESPONSE_TIME,
            "time_saved_human": "47 hours 59 minutes",
            "speed_multiplier": f"{config.TRADITIONAL_RESPONSE_TIME / config.SENTINEL_RESPONSE_TIME:.0f}x faster"
        }

crisis_simulator = CrisisSimulator()