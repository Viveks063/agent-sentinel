import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (if needed)
    GDELT_API_KEY = os.getenv("GDELT_API_KEY", "")
    
    # Model Configuration
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Thresholds (THE GENIUS CALIBRATION)
    FALSEHOOD_THRESHOLD = 0.75  # Above this = CRITICAL THREAT
    VERIFICATION_CONFIDENCE_MIN = 0.6  # Minimum confidence for verification
    VIRAL_PREDICTION_THRESHOLD = 0.7  # Predict viral spread
    
    # GDELT Configuration
    GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    GDELT_QUERY_LIMIT = 250
    
    # Crisis Simulation Scenarios
    CRISIS_SCENARIOS: Dict[str, Dict] = {
        "cyberattack": {
            "headline": "BREAKING: Major cyberattack shuts down Mumbai power grid",
            "content": "Unverified reports claim nationwide infrastructure breach by foreign actors",
            "true_falsehood_score": 0.95,
            "sources": []
        },
        "riot": {
            "headline": "URGENT: Violent riots erupt in South Mumbai, 50+ casualties",
            "content": "Social media posts claim communal violence spreading across city",
            "true_falsehood_score": 0.92,
            "sources": []
        },
        "earthquake": {
            "headline": "ALERT: 7.8 magnitude earthquake hits Mumbai, tsunami warning issued",
            "content": "Multiple sources reporting major seismic activity",
            "true_falsehood_score": 0.88,
            "sources": []
        }
    }
    
    # Trusted Sources (THE GENIUS WHITELIST)
    TRUSTED_SOURCES = [
        "reuters.com",
        "bbc.com",
        "bbc.co.uk",
        "apnews.com",
        "theguardian.com",
        "nytimes.com",
        "bloomberg.com",
        "timesofindia.com",
        "hindustantimes.com",
        "thehindu.com",
        "ndtv.com",
        "indianexpress.com",
        "who.int",
        "cdc.gov",
        "pib.gov.in"  # Press Information Bureau India
    ]
    
    # Government Alert Protocols
    ALERT_LEVELS = {
        "LOW": {"score_range": (0.0, 0.2), "action": "monitor"},
        "MEDIUM": {"score_range": (0.2, 0.4), "action": "analyze"},
        "HIGH": {"score_range": (0.4, 0.6), "action": "alert"},  # Now 0.54 will be HIGH
        "CRITICAL": {"score_range": (0.6, 1.0), "action": "intervene"}
    }
    
    # Response Time Targets (for comparison metrics)
    TRADITIONAL_RESPONSE_TIME = 48 * 3600  # 48 hours in seconds
    SENTINEL_RESPONSE_TIME = 1.5  # 1.5 seconds

config = Config()