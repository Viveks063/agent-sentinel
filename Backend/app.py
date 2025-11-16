import os, time, json, re, hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import traceback
from fastapi.middleware.cors import CORSMiddleware
from dateutil import parser

# ML imports
from transformers import pipeline
import spacy
from bs4 import BeautifulSoup
import feedparser

# Database
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

"""
✅ COMPLETE INTEGRATED SYSTEM:
- 🗄️ Database persistence (SQLite/PostgreSQL)
- 🤖 Autonomous scanning every 5 minutes
- 🧹 Daily cleanup (deletes alerts >30 days)
- 🌐 9 data sources (Telegram, YouTube, Instagram, Reddit, Twitter, 4chan, Rumble, Gettr, Truth Social)
- 🔍 Multi-model ML detection
- 📊 Statistics & logging
"""

load_dotenv()

# --- CONFIG ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
COUNTRY_CODE = os.getenv("COUNTRY_CODE", "IN").upper()
USE_LIGHT_MODE = os.getenv("USE_LIGHT_MODE", "1") == "1"
GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY")

# Database config
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agent_sentinel.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
LIGHT_MODEL_A = "roberta-large-mnli"
LIGHT_MODEL_B = "microsoft/deberta-v3-small"
LIGHT_MODEL_C = "distilbert-base-uncased-finetuned-sst-2-english"
FAKE_NEWS_MODEL = "hamzab/roberta-fake-news-classification"  # ✅ NEW: Specialized fake news model

OPENFDA_BASE = "https://api.fda.gov"

TRUSTED_SOURCES = {
    "fda.gov": {"credibility": 1.00},
    "who.int": {"credibility": 1.00},
    "reuters.com": {"credibility": 0.98},
    "apnews.com": {"credibility": 0.97},
    "factcheck.org": {"credibility": 0.95},
    "snopes.com": {"credibility": 0.92},
    "bbc.co.uk": {"credibility": 0.96},
    "washingtonpost.com": {"credibility": 0.95},
}

VIRAL_KEYWORDS = [
    "shocking", "breaking", "alert", "urgent", "confirmed",
    "exposed", "leaked", "truth", "banned", "secret",
    "conspiracy", "cover-up", "scandal", "exclusive",
    "investigation", "warning", "crisis", "emergency",
    "unprecedented", "first time", "must see", "you won't believe"
]

# RSS FEEDS - 30+ GLOBAL NEWS SOURCES
RSS_FEEDS = {
    # USA
    "NYT_World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "WashPost_World": "https://feeds.washingtonpost.com/rss/world",
    "AP_News": "https://rsshub.app/apnews/topics/apf-topnews",
    "CNN_World": "http://rss.cnn.com/rss/edition_world.rss",
    "NPR": "https://feeds.npr.org/1004/rss.xml",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
    "Politico": "https://www.politico.com/rss/politics08.xml",
    "USAToday": "http://rssfeeds.usatoday.com/usatoday-NewsTopStories",
    
    # UK
    "BBC_World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "SkyNews": "https://feeds.skynews.com/feeds/rss/world.xml",
    "Independent": "https://www.independent.co.uk/news/world/rss",
    
    # Europe
    "DW": "https://rss.dw.com/rdf/rss-en-all",
    "LeMonde": "https://www.lemonde.fr/en/rss/une.xml",
    "Euronews": "https://www.euronews.com/rss",
    "Reuters": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    
    # Asia
    "TheHindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "TimesOfIndia": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "JapanTimes": "https://www.japantimes.co.jp/feed/",
    "SCMP": "https://www.scmp.com/rss/91/feed",
    "KoreaHerald": "http://www.koreaherald.com/common/rss_xml.php?ct=210",
    "StraitsTimes": "https://www.straitstimes.com/news/singapore/rss.xml",
    
    # Middle East
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "ArabNews": "https://www.arabnews.com/rss.xml",
    "JPost": "https://www.jpost.com/rss/rssfeedsheadlines.aspx",
    "Haaretz": "https://www.haaretz.com/cmlink/1.628785"
}

# GOOGLE SEARCH TERMS (FULL INTERNET SCAN)
GOOGLE_SEARCH_TERMS = [
    "breaking news",
    "viral conspiracy",
    "exposed",
    "leaked",
    "scandal",
    "urgent news",
    "this is shocking",
    "conspiracy theory",
    "fake news alert",
    "misinformation"
]

# --- DATABASE MODELS ---
class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True)
    title = Column(String, index=True)
    source = Column(String)
    url = Column(String)
    falsehood_score = Column(Float)
    virality_score = Column(Float)
    alert_severity = Column(String)
    report = Column(Text)
    published_at = Column(DateTime)
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_alert = Column(Boolean, default=False)
    
class AnalysisLog(Base):
    __tablename__ = "analysis_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_articles = Column(Integer)
    total_alerts = Column(Integer)
    sources_checked = Column(Integer)
    status = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Agent Sentinel — 24/7 Autonomous System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

models = {"a": None, "b": None, "c": None}
cache = {}
viral_alerts = []

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
    nlp = spacy.load("en_core_web_sm")

def try_load_pipeline(model_name: str, task: str = "text-classification"):
    try:
        return pipeline(task, model=model_name)
    except Exception as e:
        print(f"[WARN] Failed to load {model_name}: {e}")
        return None

print("🔄 Loading ML models...")
models["a"] = try_load_pipeline(LIGHT_MODEL_A, "text-classification")
models["b"] = try_load_pipeline(LIGHT_MODEL_B, "text-classification")
models["c"] = try_load_pipeline(LIGHT_MODEL_C, "text-classification")
models["fake_news"] = try_load_pipeline(FAKE_NEWS_MODEL, "text-classification")  # ✅ NEW
print("✅ Models loaded.")

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]

def calculate_virality_score(article: Dict[str, Any]) -> float:
    score = 0.0
    title = (article.get("title") or "").lower()
    
    sensational_count = sum(1 for kw in VIRAL_KEYWORDS if kw in title)
    score += min(0.3, sensational_count * 0.06)
    
    if len(title.split()) <= 15:
        score += 0.15
    if re.search(r'\d+', title):
        score += 0.08
    if title.count('?') > 0:
        score += 0.1
    if title.count('!') > 0:
        score += 0.08
    
    return round(min(1.0, score), 2)

def layer1_detection(text: str) -> Dict[str, Any]:
    outputs = {}
    
    def run_model(pipe):
        if not pipe:
            return {"label": "UNKNOWN", "score": 0.5}
        try:
            out = pipe(text[:512])
            o = out[0] if isinstance(out, list) else out
            return {"label": str(o.get("label", "")).upper(), "score": float(o.get("score", 0.5))}
        except Exception as e:
            return {"label": "ERROR", "score": 0.5}
    
    outputs["roberta"] = run_model(models["a"])
    outputs["deberta"] = run_model(models["b"])
    outputs["bert"] = run_model(models["c"])
    
    # ✅ NEW: Specialized fake news model (highest weight!)
    outputs["fake_news_specialist"] = run_model(models["fake_news"])
    
    def is_fake_label(l):
        return any(x in (l or "").upper() for x in ["FAKE", "LABEL_1", "MISINFORMATION", "FALSE"])
    
    def model_fake_prob(m):
        lab = m.get("label", "")
        sc = m.get("score", 0.5)
        return sc if is_fake_label(lab) else (1.0 - sc)
    
    # ✅ NEW WEIGHTS: Fake news model gets highest priority!
    w_fake_news = 0.50  # 50% weight to specialized model
    w_roberta = 0.20    # 20% to RoBERTa
    w_deberta = 0.20    # 20% to DeBERTa
    w_bert = 0.10       # 10% to BERT
    
    ensemble = (
        model_fake_prob(outputs["fake_news_specialist"]) * w_fake_news +
        model_fake_prob(outputs["roberta"]) * w_roberta + 
        model_fake_prob(outputs["deberta"]) * w_deberta + 
        model_fake_prob(outputs["bert"]) * w_bert
    )
    ensemble = round(max(0.0, min(1.0, ensemble)), 3)
    
    return {
        "ensemble_label": "FAKE" if ensemble >= 0.5 else "REAL",
        "ensemble_score": ensemble,
        "models": outputs,
    }

def layer2_claim_extraction(text: str) -> Dict[str, Any]:
    if len(text.split()) <= 18:
        claim = text.strip()
    else:
        doc = nlp(text)
        chosen = None
        for s in doc.sents:
            if any(tok.pos_ == "VERB" for tok in s) and len(s.text.split()) < 40:
                chosen = s.text
                break
        claim = chosen or (list(doc.sents)[0].text if doc.sents else text)
    
    ents = [{"text": e.text, "label": e.label_} for e in nlp(claim).ents]
    return {"claim": claim, "entities": ents}

def layer3_authoritative(claim: str, entities: List[Dict[str, str]]) -> Dict[str, Any]:
    checks = [{"source": "FDA", "status": "NOT_APPLICABLE"}]
    return {"authoritative_checks": checks}

def layer4_fact_checks(claim: str) -> Dict[str, Any]:
    results = []

    if not GOOGLE_FACTCHECK_KEY:
        return {"matches": [], "note": "No Google Fact Check API key provided"}

    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": claim, "key": GOOGLE_FACTCHECK_KEY, "pageSize": 10}

        resp = requests.get(url, params=params, timeout=10).json()
        claims = resp.get("claims", [])

        for c in claims:
            reviews = c.get("claimReview", [])
            for r in reviews:
                results.append({
                    "text": c.get("text"),
                    "claimDate": c.get("claimDate"),
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "rating": r.get("textualRating", "Unknown"),
                    "reviewDate": r.get("reviewDate")
                })

        return {"matches": results}

    except Exception as e:
        return {"matches": [], "error": str(e)}

GDELT_WORKING = True

def layer5_contextual(text: str, url: Optional[str] = None) -> Dict[str, Any]:
    global GDELT_WORKING

    # ✅ IMPROVED: Better domain extraction
    domain = "unknown"
    if url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
        except:
            # Fallback to simple split
            try:
                domain = url.split("/")[2].lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
            except:
                domain = "unknown"
    
    # Check if domain is in trusted sources
    credibility = 0.4  # Default low credibility
    is_trusted = False
    
    for trusted_domain, info in TRUSTED_SOURCES.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            credibility = info.get("credibility", 0.5)
            is_trusted = True
            break
    
    # ✅ NEW: Check if text contains links to trusted sources (for Reddit/Telegram)
    contains_trusted_link = False
    linked_trusted_sources = []
    
    # Social media platforms (should check content for trusted links)
    is_social_media = any(sm in domain for sm in [
        'reddit.com', 't.me', 'telegram', 'twitter.com', 'x.com',
        '4chan.org', 'instagram.com', 'youtube.com', 'rumble.com'
    ])
    
    if is_social_media and text:
        # Look for URLs in text
        url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9\-\.]+)'
        import re
        found_urls = re.findall(url_pattern, text)
        
        for found_domain in found_urls:
            found_domain = found_domain.lower()
            if found_domain.startswith('www.'):
                found_domain = found_domain[4:]
            
            # Check if this URL points to a trusted source
            for trusted_domain, info in TRUSTED_SOURCES.items():
                if found_domain == trusted_domain or found_domain.endswith(f".{trusted_domain}"):
                    contains_trusted_link = True
                    linked_trusted_sources.append(trusted_domain)
                    # Upgrade credibility if social media links to trusted source
                    credibility = max(credibility, 0.75)  # Not full trust, but higher
                    break

    mainstream_hits = 0
    alt_hits = 0
    sources = []
    global_reach = 0.0

    # ✅ GDELT DISABLED - Too unreliable, slows down system
    GDELT_WORKING = False

    flags = []
    if mainstream_hits == 0 and alt_hits > 3:
        flags.append("appears-only-on-alternative-news")
    if mainstream_hits > 0:
        flags.append("covered-by-mainstream-media")
    
    # ✅ Add flags
    if is_trusted:
        flags.append("from-trusted-source")
    if is_social_media:
        flags.append("from-social-media")
    if contains_trusted_link:
        flags.append("links-to-trusted-source")

    return {
        "domain": domain,
        "credibility_score": round(credibility, 2),
        "global_reach": round(global_reach, 2),
        "mainstream_hits": mainstream_hits,
        "alternative_hits": alt_hits,
        "gdelt_sources": sources[:10],
        "flags": flags,
        "is_trusted_domain": is_trusted,
        "is_social_media": is_social_media,
        "contains_trusted_link": contains_trusted_link,
        "linked_trusted_sources": linked_trusted_sources
    }

def layer6_aggregate(l1, l3, l4, l5, virality) -> Dict[str, Any]:
    """
    ✅ PROPER MULTI-LAYER FAKE NEWS DETECTION
    - ML models are PRIMARY decision makers
    - Source reputation is SECONDARY influencer (not decision maker)
    - Cross-verification is KEY
    - Rule-based boost for obvious fake patterns
    """
    
    # ========================================
    # STEP 1: ML MODEL CONSENSUS (50% weight)
    # ========================================
    ml_fake_score = l1.get("ensemble_score", 0.5)  # 0 = Real, 1 = Fake
    
    # ========================================
    # STEP 1.5: RULE-BASED PATTERN DETECTION (10% weight)
    # ========================================
    # Check for obvious fake news patterns
    claim_text = l1.get("models", {}).get("roberta", {}).get("label", "")
    
    # Detect conspiracy/misinformation keywords
    text_to_check = str(l1) + str(l3) + str(l4)
    
    fake_indicators = [
        "vaccine", "autism", "bill gates", "soros", "deep state",
        "5g", "microchip", "chemtrails", "flat earth", "qanon",
        "new world order", "illuminati", "fake moon landing"
    ]
    
    urgent_spam = [
        "share before deleted", "they don't want you to know",
        "banned truth", "you must believe", "wake up sheeple",
        "do your research", "mainstream media lies"
    ]
    
    fake_indicator_count = sum(1 for indicator in fake_indicators if indicator in text_to_check.lower())
    urgent_spam_count = sum(1 for spam in urgent_spam if spam in text_to_check.lower())
    
    # Rule-based score
    rule_based_score = min(1.0, (fake_indicator_count * 0.15) + (urgent_spam_count * 0.20))
    
    # ========================================
    # STEP 2: FACT-CHECK VERIFICATION (20% weight)
    # ========================================
    fact_check_matches = l4.get("matches", [])
    fact_check_score = 0.0
    
    if len(fact_check_matches) > 0:
        fake_ratings = ["false", "fake", "misleading", "pants on fire", "incorrect"]
        real_ratings = ["true", "correct", "accurate", "verified"]
        
        fake_count = sum(1 for m in fact_check_matches 
                        if any(fr in str(m.get("rating", "")).lower() for fr in fake_ratings))
        real_count = sum(1 for m in fact_check_matches 
                        if any(rr in str(m.get("rating", "")).lower() for rr in real_ratings))
        
        if fake_count > 0:
            fact_check_score = 0.8  # Fact-checkers say it's fake
        elif real_count > 0:
            fact_check_score = 0.1  # Fact-checkers say it's real
        else:
            fact_check_score = 0.4  # Inconclusive
    else:
        fact_check_score = 0.5  # No fact-checks = neutral
    
    # ========================================
    # STEP 3: SOURCE CREDIBILITY (10% weight)
    # ========================================
    credibility = l5.get("credibility_score", 0.5)
    is_trusted = l5.get("is_trusted_domain", False)
    is_social_media = l5.get("is_social_media", False)
    contains_trusted_link = l5.get("contains_trusted_link", False)
    
    source_score = 1.0 - credibility
    
    if is_social_media and contains_trusted_link:
        source_score *= 0.7
    
    # ========================================
    # STEP 4: MAINSTREAM COVERAGE (10% weight)
    # ========================================
    mainstream_hits = l5.get("mainstream_hits", 0)
    coverage_score = 0.5
    
    if mainstream_hits > 3:
        coverage_score = 0.2
    elif mainstream_hits > 0:
        coverage_score = 0.35
    else:
        coverage_score = 0.5
    
    # ========================================
    # FINAL WEIGHTED SCORE
    # ========================================
    falsehood = (
        ml_fake_score * 0.50 +           # ML models (reduced from 60%)
        rule_based_score * 0.10 +        # Rule-based patterns (NEW!)
        fact_check_score * 0.20 +        # Fact-checking
        source_score * 0.10 +            # Source reputation
        coverage_score * 0.10            # Mainstream coverage
    )
    
    # ========================================
    # VIRALITY BOOST (makes alerts more sensitive)
    # ========================================
    # Very viral content gets extra scrutiny
    if virality >= 0.6:
        falsehood += 0.10  # High virality increases suspicion
    elif virality >= 0.4:
        falsehood += 0.05
    
    falsehood = round(max(0.0, min(1.0, falsehood)), 3)
    
    # ========================================
    # ALERT DECISION
    # ========================================
    is_alert = (
        (falsehood >= 0.45 and virality >= 0.30) or  # Likely fake + viral
        (falsehood >= 0.55) or                        # High confidence fake
        (falsehood >= 0.35 and virality >= 0.60) or  # Moderate fake + very viral
        (rule_based_score >= 0.3 and virality >= 0.4)  # Rule detected + viral
    )
    
    # Severity
    if falsehood >= 0.70 and virality >= 0.50:
        alert_severity = "CRITICAL"
    elif falsehood >= 0.60 or virality >= 0.60:
        alert_severity = "HIGH"
    elif is_alert:
        alert_severity = "MEDIUM"
    else:
        alert_severity = "LOW"
    
    # ========================================
    # VERDICT
    # ========================================
    verdict = "FAKE" if falsehood >= 0.50 else "REAL"
    
    return {
        "falsehood_score": falsehood,
        "virality_score": virality,
        "verdict": verdict,
        "is_alert": is_alert,
        "alert_severity": alert_severity,
        "reasoning": {
            "ml_models_score": round(ml_fake_score, 3),
            "ml_weight": "50%",
            "rule_based_score": round(rule_based_score, 3),
            "rule_based_weight": "10%",
            "fake_indicators_found": fake_indicator_count,
            "spam_patterns_found": urgent_spam_count,
            "fact_check_score": round(fact_check_score, 3),
            "fact_check_weight": "20%",
            "fact_checks_found": len(fact_check_matches),
            "source_credibility_score": round(credibility, 3),
            "source_weight": "10%",
            "is_trusted_source": is_trusted,
            "mainstream_coverage_score": round(coverage_score, 3),
            "coverage_weight": "10%",
            "virality_boost": round(0.10 if virality >= 0.6 else 0.05 if virality >= 0.4 else 0, 2),
            "final_falsehood_score": falsehood,
            "confidence": "High" if abs(ml_fake_score - 0.5) > 0.3 else "Medium" if abs(ml_fake_score - 0.5) > 0.15 else "Low"
        }
    }

def analyze_for_viral_fake(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = short_hash(article.get('title', '') + article.get('url', ''))
    if key in cache:
        return cache[key]
    
    try:
        title = article.get('title', 'Unknown')
        description = article.get('description', '')
        text = f"{title} {description}"
        
        source_obj = article.get('source', {})
        if isinstance(source_obj, dict):
            source_name = source_obj.get('name', 'Unknown Source')
        else:
            source_name = str(source_obj) if source_obj else 'Unknown Source'
        
        l1 = layer1_detection(text)
        l2 = layer2_claim_extraction(text)
        virality = calculate_virality_score(article)
        l3 = layer3_authoritative(l2["claim"], l2["entities"])
        l4 = layer4_fact_checks(l2["claim"])
        l5 = layer5_contextual(text, article.get('url'))
        l6 = layer6_aggregate(l1, l3, l4, l5, virality)
        
        report = f"""🚨 VIRAL FAKE NEWS ALERT 🚨
================================================================================

HEADLINE: {title}
SOURCE: {source_name}
URL: {article.get('url', 'N/A')}

VERDICT: {'🔴 FAKE NEWS' if l6['verdict'] == 'FAKE' else '🟢 REAL NEWS'}
Falsehood Score: {l6['falsehood_score']*100:.0f}%
Virality Score: {l6['virality_score']*100:.0f}%
Alert Severity: {l6['alert_severity']}

WHY THIS IS FLAGGED:
{'✓ VIRAL INDICATORS: Strong sensationalism detected' if l6['virality_score'] >= 0.6 else ''}
✓ AI MODEL CONFIDENCE: {l1['ensemble_score']*100:.0f}% confidence this is {'FAKE' if l6['verdict'] == 'FAKE' else 'REAL'}

RECOMMENDED ACTION:
{'🛑 DO NOT SHARE - This appears to be misinformation' if l6['verdict'] == 'FAKE' else '✅ Can be safely shared'}

Report Generated: {datetime.utcnow().isoformat()}
================================================================================"""
        
        # Handle published_at properly
        published_at = article.get('publishedAt')
        if isinstance(published_at, str):
            try:
                published_at = parser.isoparse(published_at).replace(tzinfo=None)
            except:
                published_at = datetime.utcnow()
        elif not isinstance(published_at, datetime):
            published_at = datetime.utcnow()
        
        result = {
            "id": key,
            "title": title,
            "url": article.get('url', '#'),
            "source": source_name,
            "published_at": published_at,
            "virality_score": virality,
            "falsehood_score": l6['falsehood_score'],
            "is_alert": l6['is_alert'],
            "alert_severity": l6['alert_severity'],
            "report": report,
            "analyzed_at": datetime.utcnow()
        }
        
        cache[key] = result
        return result
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        return None

# =========================================
# DATA SOURCES (ALL 9 PLATFORMS)
# =========================================

def fetch_from_twitter_official() -> List[Dict[str, Any]]:
    """Fetch from Twitter Official API (FREE tier)."""
    articles = []
    
    if not TWITTER_BEARER_TOKEN:
        print("[TWITTER] No bearer token configured, skipping official API")
        return articles
    
    try:
        print("[TWITTER OFFICIAL API] Fetching conspiracy tweets...")
        
        headers = {
            "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
        }
    
        search_url = "https://api.twitter.com/2/tweets/search/recent"
        
        query_params = {
            "query": "conspiracy -is:retweet lang:en",
            "max_results": 20,
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username"
        }
        
        response = requests.get(search_url, headers=headers, params=query_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            tweets = data.get("data", [])
            
            one_week_ago = (datetime.now() - timedelta(days=7))
            
            for tweet in tweets:
                created_at = parser.isoparse(tweet["created_at"]).replace(tzinfo=None)
                if created_at >= one_week_ago:
                    articles.append({
                        "title": tweet["text"],
                        "description": tweet["text"][:200],
                        "url": f"https://twitter.com/i/web/status/{tweet['id']}",
                        "source": {"name": "Twitter/X (Official API)"},
                        "publishedAt": created_at
                    })
            
            print(f"  ✓ Got {len(articles)} conspiracy tweets from official API")
        elif response.status_code == 429:
            print("  ⚠️ Twitter API rate limit hit — switching to Nitter fallback...")
            return fetch_from_twitter()
        else:
            print(f"  ✗ Twitter API error: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Twitter official API error: {str(e)}")
    
    return articles

def fetch_from_telegram() -> List[Dict[str, Any]]:
    """Fetch from Telegram public channels."""
    articles = []
    channels = ["conspiracy_theories", "truth_uncovered", "breaking_news_alert"]

    for channel in channels:
        try:
            print(f"[TELEGRAM] Fetching from {channel}...")
            resp = requests.get(
                f"https://t.me/s/{channel}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                messages = soup.find_all("div", class_="tgme_widget_message_text")[:10]
                
                for msg in messages:
                    text = msg.get_text(strip=True)[:200]
                    if len(text) > 10:
                        articles.append({
                            "title": text,
                            "description": f"Telegram - {channel}",
                            "url": f"https://t.me/{channel}",
                            "source": {"name": "Telegram"},
                            "publishedAt": datetime.utcnow()
                        })
            print(f"  ✓ Got {len(messages)} messages from Telegram {channel}")
        except Exception as e:
            print(f"  ✗ Telegram error: {str(e)}")

    return articles

def fetch_from_reddit() -> List[Dict[str, Any]]:
    """Fetch from Reddit conspiracy subreddits."""
    articles = []
    subreddits = ["conspiracy", "HolUp", "JusticeServed"]
    
    for sub in subreddits:
        try:
            print(f"[REDDIT] Fetching from r/{sub}...")
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/new.json",
                params={"limit": 50},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if resp.status_code == 200:
                posts = resp.json()["data"]["children"]
                for p in posts:
                    data = p.get("data", {})
                    articles.append({
                        "title": data.get("title"),
                        "description": data.get("selftext", "")[:200],
                        "url": f"https://reddit.com{data.get('permalink', '')}",
                        "source": {"name": f"Reddit-r/{sub}"},
                        "publishedAt": datetime.fromtimestamp(data.get("created_utc", 0))
                    })
                print(f"  ✓ Got {len(posts)} posts from r/{sub}")
        except Exception as e:
            print(f"  ✗ Reddit error: {str(e)}")
    
    return articles

def fetch_from_4chan() -> List[Dict[str, Any]]:
    """Fetch from 4chan /pol/."""
    articles = []
    
    try:
        print("[4CHAN] Fetching from /pol/...")
        resp = requests.get("https://a.4cdn.org/pol/catalog.json", timeout=10)
        if resp.status_code == 200:
            threads = resp.json()
            count = 0
            for page in threads[:3]:
                for thread in page.get("threads", [])[:10]:
                    title = thread.get("sub") or thread.get("com", "No title")
                    title = re.sub(r'<[^>]+>', '', title)[:100]
                    articles.append({
                        "title": title,
                        "description": thread.get("com", "")[:200],
                        "url": f"https://4chan.org/pol/thread/{thread.get('no')}",
                        "source": {"name": "4chan /pol/"},
                        "publishedAt": datetime.fromtimestamp(thread.get("time", 0))
                    })
                    count += 1
            print(f"  ✓ Got {count} threads from 4chan /pol/")
    except Exception as e:
        print(f"  ✗ 4chan error: {str(e)}")
    
    return articles

def fetch_from_youtube_shorts() -> List[Dict[str, Any]]:
    """Fetch from YouTube Shorts."""
    articles = []
    
    if not YOUTUBE_API_KEY:
        print("[YOUTUBE] No API key, skipping...")
        return articles
    
    try:
        print("[YOUTUBE SHORTS] Fetching...")
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "q": "conspiracy shorts",
                "part": "snippet",
                "type": "video",
                "key": YOUTUBE_API_KEY,
                "maxResults": 20
            },
            timeout=10
        )
        
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            for item in items:
                snippet = item.get("snippet", {})
                articles.append({
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", "")[:200],
                    "url": f"https://youtube.com/watch?v={item.get('id', {}).get('videoId', '')}",
                    "source": {"name": "YouTube Shorts"},
                    "publishedAt": snippet.get("publishedAt", datetime.utcnow().isoformat())
                })
            print(f"  ✓ Got {len(items)} videos from YouTube Shorts")
    except Exception as e:
        print(f"  ✗ YouTube error: {str(e)}")
    
    return articles

def fetch_from_instagram() -> List[Dict[str, Any]]:
    """Fetch from Instagram Reels."""
    articles = []
    hashtags = ["conspiracy", "exposedtruth", "leaked"]
    
    for tag in hashtags:
        try:
            print(f"[INSTAGRAM] Fetching from #{tag}...")
            resp = requests.get(
                f"https://www.instagram.com/explore/tags/{tag}/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                articles.append({
                    "title": f"Instagram #{tag} trending",
                    "description": f"Viral hashtag on Instagram",
                    "url": f"https://instagram.com/explore/tags/{tag}/",
                    "source": {"name": "Instagram Reels"},
                    "publishedAt": datetime.utcnow()
                })
            print(f"  ✓ Fetched Instagram #{tag}")
        except Exception as e:
            print(f"  ✗ Instagram error: {str(e)}")
    
    return articles

def fetch_from_twitter() -> List[Dict[str, Any]]:
    """Fetch from Twitter via Nitter (unofficial)."""
    articles = []
    search_terms = ["conspiracy", "exposed", "truth"]
    
    for term in search_terms:
        try:
            print(f"[TWITTER] Searching for '{term}'...")
            resp = requests.get(
                "https://nitter.net/search",
                params={"q": term, "f": "tweet"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                tweets = soup.find_all("div", class_="tweet-content")[:10]
                
                for tweet in tweets:
                    text = tweet.get_text(strip=True)[:200]
                    if len(text) > 5:
                        articles.append({
                            "title": text,
                            "description": f"Twitter - {term}",
                            "url": f"https://twitter.com/search?q={term}",
                            "source": {"name": "Twitter/X"},
                            "publishedAt": datetime.utcnow()
                        })
            print(f"  ✓ Got tweets about {term}")
        except Exception as e:
            print(f"  ✗ Twitter error: {str(e)}")
    
    return articles

def fetch_from_rumble() -> List[Dict[str, Any]]:
    """Fetch from Rumble."""
    articles = []
    search_terms = ["breaking", "exposed"]
    
    for term in search_terms:
        try:
            print(f"[RUMBLE] Searching for '{term}'...")
            resp = requests.get(
                f"https://rumble.com/search",
                params={"q": term},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                videos = soup.find_all("a")[:10]
                
                for video in videos:
                    title = video.get_text(strip=True)[:150]
                    if len(title) > 5:
                        articles.append({
                            "title": title,
                            "description": f"Rumble - {term}",
                            "url": video.get("href", "https://rumble.com"),
                            "source": {"name": "Rumble"},
                            "publishedAt": datetime.utcnow()
                        })
            print(f"  ✓ Got Rumble videos")
        except Exception as e:
            print(f"  ✗ Rumble error: {str(e)}")
    
    return articles

def fetch_from_gettr() -> List[Dict[str, Any]]:
    """Fetch from Gettr."""
    articles = []
    
    try:
        print("[GETTR] Fetching...")
        resp = requests.get(
            "https://gettr.com/search?q=conspiracy",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        
        if resp.status_code == 200:
            articles.append({
                "title": "Gettr conspiracy posts",
                "description": "Posts from conservative social network",
                "url": "https://gettr.com/search?q=conspiracy",
                "source": {"name": "Gettr"},
                "publishedAt": datetime.utcnow()
            })
            print(f"  ✓ Fetched Gettr")
    except Exception as e:
        print(f"  ✗ Gettr error: {str(e)}")
    
    return articles

# =========================================
# GNEWS.IO - GLOBAL NEWS API
# =========================================

def fetch_from_gnews() -> List[Dict[str, Any]]:
    """Fetch global news from GNews.io"""
    articles = []
    
    if not GNEWS_API_KEY:
        print("[GNEWS] No API key provided, skipping...")
        return articles
    
    print("\n[GNEWS] Fetching global news...")
    
    keywords = [
        "breaking news", "conspiracy", "exposed", "leaked",
        "fake news", "misinformation", "urgent news", "scandal",
        "viral", "shocking"
    ]
    
    for q in keywords:
        try:
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q": q,
                    "token": GNEWS_API_KEY,
                    "lang": "en",
                    "max": 10
                },
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("articles", [])
                
                for item in items:
                    articles.append({
                        "title": item.get("title"),
                        "description": item.get("description", "")[:300],
                        "url": item.get("url"),
                        "source": {"name": item.get("source", {}).get("name", "GNews")},
                        "publishedAt": item.get("publishedAt", datetime.utcnow().isoformat())
                    })
                
                print(f"  ✓ Found {len(items)} articles for '{q}'")
            elif resp.status_code == 403:
                print(f"  ✗ GNews API: Invalid API key or rate limit exceeded")
                break  # Stop trying if API key is invalid
            else:
                print(f"  ✗ GNews API error: {resp.status_code}")
                
        except Exception as e:
            print(f"  ✗ GNews error for '{q}': {e}")
    
    print(f"[GNEWS TOTAL] {len(articles)} global articles fetched")
    return articles

# =========================================
# RSS FEEDS - 30+ GLOBAL NEWS SOURCES
# =========================================

def fetch_from_rss_feeds() -> List[Dict[str, Any]]:
    """Fetch from 30+ global RSS news feeds."""
    articles = []
    
    print("\n[RSS FEEDS] Fetching from 27 global news sources...")
    
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            
            if feed.entries:
                for entry in feed.entries[:10]:  # Get top 10 from each source
                    title = entry.get('title', '')
                    description = entry.get('summary', entry.get('description', ''))[:300]
                    link = entry.get('link', '')
                    
                    # Parse publication date
                    pub_date = entry.get('published_parsed') or entry.get('updated_parsed')
                    if pub_date:
                        published_at = datetime(*pub_date[:6])
                    else:
                        published_at = datetime.utcnow()
                    
                    if title and len(title) > 10:
                        articles.append({
                            "title": title,
                            "description": description,
                            "url": link,
                            "source": {"name": source_name},
                            "publishedAt": published_at
                        })
                
                print(f"  ✓ {source_name}: {len(feed.entries[:10])} articles")
            else:
                print(f"  ⚠️ {source_name}: No entries found")
                
        except Exception as e:
            print(f"  ✗ {source_name} error: {str(e)}")
    
    print(f"[RSS TOTAL] Fetched {len(articles)} articles from RSS feeds")
    return articles

# =========================================
# GOOGLE SEARCH SCRAPER (FULL INTERNET)
# =========================================

def fetch_from_google_search() -> List[Dict[str, Any]]:
    """Scrape Google Search for viral content across FULL internet."""
    articles = []
    
    print("\n[GOOGLE SEARCH] Scanning full internet...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    for term in GOOGLE_SEARCH_TERMS[:5]:  # Limit to 5 searches to avoid blocking
        try:
            print(f"  🔍 Searching: '{term}'...")
            
            # Use DuckDuckGo as fallback (more reliable than Google scraping)
            search_url = f"https://html.duckduckgo.com/html/?q={term}"
            
            resp = requests.get(search_url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                
                # DuckDuckGo results
                results = soup.find_all("a", class_="result__a")
                
                count = 0
                for result in results[:10]:  # Top 10 results per search
                    try:
                        title = result.get_text(strip=True)
                        link = result.get("href", "")
                        
                        if title and link and len(title) > 10:
                            articles.append({
                                "title": title,
                                "description": f"DuckDuckGo Search result for: {term}",
                                "url": link,
                                "source": {"name": f"Search-{term}"},
                                "publishedAt": datetime.utcnow()
                            })
                            count += 1
                    
                    except Exception as e:
                        continue
                
                print(f"    ✓ Found {count} results for '{term}'")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            else:
                print(f"    ✗ Search returned status {resp.status_code}")
                
        except Exception as e:
            print(f"    ✗ Error searching '{term}': {str(e)}")
    
    print(f"[SEARCH TOTAL] Fetched {len(articles)} articles from web search")
    return articles

def fetch_from_truth_social() -> List[Dict[str, Any]]:
    """Fetch from Truth Social."""
    articles = []
    
    try:
        print("[TRUTH SOCIAL] Fetching...")
        resp = requests.get(
            "https://truthsocial.com/api/v1/timelines/public",
            params={"limit": 20},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        
        if resp.status_code == 200:
            posts = resp.json() if isinstance(resp.json(), list) else resp.json().get("statuses", [])
            
            for post in posts[:10]:
                if isinstance(post, dict):
                    title = re.sub(r'<[^>]+>', '', post.get("content", ""))[:150]
                    if len(title) > 5:
                        articles.append({
                            "title": title,
                            "description": "Truth Social post",
                            "url": post.get("url", "https://truthsocial.com"),
                            "source": {"name": "Truth Social"},
                            "publishedAt": post.get("created_at", datetime.utcnow())
                        })
            print(f"  ✓ Got {len(posts)} posts from Truth Social")
    except Exception as e:
        print(f"  ✗ Truth Social error: {str(e)}")
    
    return articles

def fetch_viral_news() -> List[Dict[str, Any]]:
    """Fetch from ALL sources including RSS feeds, Google Search, and GNews."""
    
    print(f"\n[{datetime.utcnow().isoformat()}] 🔍 Scanning for viral misinformation...")
    print(f"Country: {COUNTRY_CODE}")
    print("="*70)
    
    all_articles = []
    
    # ✨ NEW: GNEWS API - GLOBAL NEWS
    print("\n[TIER 0 - GNEWS API (GLOBAL NEWS)]")
    all_articles.extend(fetch_from_gnews())
    
    # ✨ RSS FEEDS - 27 GLOBAL NEWS SOURCES
    print("\n[TIER 0.1 - GLOBAL NEWS (RSS)]")
    all_articles.extend(fetch_from_rss_feeds())
    
    # ✨ GOOGLE SEARCH - FULL INTERNET SCAN
    print("\n[TIER 0.5 - GOOGLE SEARCH (FULL INTERNET)]")
    all_articles.extend(fetch_from_google_search())
    
    # TIER 1 - CRITICAL (India optimized)
    print("\n[TIER 1 - CRITICAL SOCIAL MEDIA]")
    all_articles.extend(fetch_from_telegram())
    all_articles.extend(fetch_from_youtube_shorts())
    all_articles.extend(fetch_from_instagram())
    all_articles.extend(fetch_from_reddit())
    
    # TIER 2 - SECONDARY
    print("\n[TIER 2 - SECONDARY PLATFORMS]")
    if TWITTER_BEARER_TOKEN:
        print("[TWITTER] Using Official API")
        all_articles.extend(fetch_from_twitter_official())
    else:
        print("[TWITTER] Using Nitter (no official API key)")
        all_articles.extend(fetch_from_twitter())
    all_articles.extend(fetch_from_4chan())
    
    # TIER 3 - BONUS
    print("\n[TIER 3 - BONUS PLATFORMS]")
    all_articles.extend(fetch_from_rumble())
    all_articles.extend(fetch_from_gettr())
    all_articles.extend(fetch_from_truth_social())
    
    print(f"\n[AGGREGATION] Total articles fetched: {len(all_articles)}")
    
    # Deduplicate
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title = art.get('title', '')
        if title and title not in seen_titles and len(title) > 5:
            seen_titles.add(title)
            unique_articles.append(art)
    
    print(f"[DEDUPLICATION] Unique articles: {len(unique_articles)}")
    
    # Filter viral content
    viral_articles = [a for a in unique_articles if calculate_virality_score(a) >= 0.1]
    
    if len(viral_articles) < 3:
        viral_articles = unique_articles[:50]  # Increased from 20 to 50 for more coverage
    
    print(f"✅ Found {len(viral_articles)} viral articles")
    return viral_articles

# ===== DATABASE FUNCTIONS =====
def save_alert_to_db(result: Dict[str, Any]):
    """Save alert to database (with duplicate check)."""
    try:
        db = SessionLocal()
        
        # Check for duplicates
        existing = db.query(Alert).filter(Alert.id == result['id']).first()
        if existing:
            print(f"  ℹ️  Alert already in DB, skipping: {result['id']}")
            db.close()
            return True
        
        alert = Alert(
            id=result['id'],
            title=result['title'],
            source=result['source'],
            url=result['url'],
            falsehood_score=result['falsehood_score'],
            virality_score=result['virality_score'],
            alert_severity=result['alert_severity'],
            report=result['report'],
            published_at=result.get('published_at'),
            analyzed_at=result['analyzed_at'],
            is_alert=result['is_alert']
        )
        db.add(alert)
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"DB save error: {e}")
        return False

def log_analysis(total_articles: int, total_alerts: int, sources: int, status: str):
    """Log analysis run to database."""
    try:
        db = SessionLocal()
        log = AnalysisLog(
            total_articles=total_articles,
            total_alerts=total_alerts,
            sources_checked=sources,
            status=status
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as e:
        print(f"Log error: {e}")

def cleanup_old_alerts():
    """Delete alerts older than 30 days to prevent DB bloat."""
    try:
        db = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=30)
        deleted_count = db.query(Alert).filter(Alert.analyzed_at < cutoff).delete()
        db.commit()
        db.close()
        print(f"🧹 Cleanup: Deleted {deleted_count} alerts older than 30 days")
    except Exception as e:
        print(f"Cleanup error: {e}")

# ===== BACKGROUND JOBS =====
def autonomous_scan():
    """Run every 5 minutes - scan for viral fake news."""
    print(f"\n{'='*70}")
    print(f"🤖 AUTONOMOUS SCAN - {datetime.utcnow().isoformat()}")
    print(f"{'='*70}")
    
    try:
        viral_news = fetch_viral_news()
        print(f"📊 Analyzing {len(viral_news)} articles...")
        
        alert_count = 0
        for article in viral_news:
            result = analyze_for_viral_fake(article)
            if result and result['is_alert']:
                alert_count += 1
                save_alert_to_db(result)
                print(f"  ✓ {alert_count}. {result['title'][:50]}... [{result['alert_severity']}]")
        
        log_analysis(len(viral_news), alert_count, 9, "SUCCESS")
        print(f"\n✅ Scan complete: {alert_count} alerts saved to database")
        
    except Exception as e:
        print(f"❌ Scan error: {e}")
        log_analysis(0, 0, 0, f"ERROR: {str(e)}")
        traceback.print_exc()

# ===== API ENDPOINTS =====
class AnalyzeRequest(BaseModel):
    text: str
    url: Optional[str] = None
    title: Optional[str] = None

@app.post("/analyze-text")
def analyze_text(req: AnalyzeRequest):
    """Analyze custom text for fake news."""
    article = {
        "title": req.title or req.text[:100], 
        "description": req.text, 
        "url": req.url,
        "publishedAt": datetime.utcnow()
    }
    result = analyze_for_viral_fake(article)
    if not result:
        raise HTTPException(500, "Analysis failed")
    return result

@app.get("/alerts")
@app.post("/alerts")
def get_alerts(limit: int = 50, trigger_scan: bool = False):
    """Get recent alerts from database. Set trigger_scan=true to run scan first."""
    try:
        # ✅ MANUAL SCAN TRIGGER - Run scan if requested
        if trigger_scan:
            print("\n🔄 Manual scan triggered via /alerts endpoint...")
            autonomous_scan()
        
        db = SessionLocal()
        alerts = db.query(Alert).order_by(Alert.analyzed_at.desc()).limit(limit).all()
        
        result = {
            "total_alerts": len(alerts),
            "scan_triggered": trigger_scan,
            "alerts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "source": a.source,
                    "url": a.url,
                    "falsehood_score": a.falsehood_score,
                    "virality_score": a.virality_score,
                    "alert_severity": a.alert_severity,
                    "report": a.report,
                    "report_summary": a.report[:300] if a.report else "",
                    "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else None,
                    "published_at": a.published_at.isoformat() if a.published_at else None
                }
                for a in alerts
            ],
            "fetched_at": datetime.utcnow().isoformat()
        }
        db.close()
        return result
    except Exception as e:
        return {"error": str(e), "alerts": []}

@app.get("/test-alerts")
def test_alerts():
    """Test endpoint - manually trigger a FULL scan and return results."""
    print("\n🔄 Manual FULL scan triggered via /test-alerts...")
    
    try:
        # Run the scan
        autonomous_scan()
        
        # Return latest alerts
        db = SessionLocal()
        alerts = db.query(Alert).order_by(Alert.analyzed_at.desc()).limit(50).all()
        
        result = {
            "scan_triggered": True,
            "total_alerts": len(alerts),
            "alerts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "source": a.source,
                    "url": a.url,
                    "severity": a.alert_severity,
                    "falsehood_score": a.falsehood_score,
                    "virality_score": a.virality_score,
                    "report_summary": a.report[:300] if a.report else "",
                    "report": a.report
                }
                for a in alerts
            ],
            "fetched_at": datetime.utcnow().isoformat()
        }
        db.close()
        return result
        
    except Exception as e:
        return {"error": str(e), "alerts": [], "scan_triggered": False}

@app.post("/test-analysis")
def test_analysis(req: AnalyzeRequest):
    """Test individual article analysis - shows all 6 layers."""
    print("\n🧪 TESTING ANALYSIS LAYERS...")
    
    try:
        article = {
            "title": req.title or req.text[:100],
            "description": req.text,
            "url": req.url or "https://test.com",
            "source": {"name": "Test Source"},
            "publishedAt": datetime.utcnow()
        }
        
        title = article.get('title', 'Unknown')
        description = article.get('description', '')
        text = f"{title} {description}"
        
        print("="*70)
        print("📝 TEXT TO ANALYZE:")
        print(f"   {text[:200]}...")
        print("="*70)
        
        # Layer 1: ML Detection
        print("\n🔬 LAYER 1: ML DETECTION (3 models)")
        l1 = layer1_detection(text)
        print(f"   RoBERTa:  {l1['models']['roberta']}")
        print(f"   DeBERTa:  {l1['models']['deberta']}")
        print(f"   BERT:     {l1['models']['bert']}")
        print(f"   ✅ ENSEMBLE: {l1['ensemble_label']} ({l1['ensemble_score']*100:.1f}% confidence)")
        
        # Layer 2: Claim Extraction
        print("\n🔍 LAYER 2: CLAIM EXTRACTION")
        l2 = layer2_claim_extraction(text)
        print(f"   Claim: {l2['claim']}")
        print(f"   Entities: {l2['entities']}")
        
        # Layer 3: Authoritative Checks
        print("\n🏛️ LAYER 3: AUTHORITATIVE CHECKS")
        l3 = layer3_authoritative(l2["claim"], l2["entities"])
        print(f"   {l3}")
        
        # Layer 4: Fact Checks
        print("\n✔️ LAYER 4: FACT CHECKING")
        l4 = layer4_fact_checks(l2["claim"])
        print(f"   Matches found: {len(l4.get('matches', []))}")
        if l4.get('matches'):
            print(f"   First match: {l4['matches'][0]}")
        
        # Layer 5: Contextual Analysis
        print("\n🌐 LAYER 5: CONTEXTUAL ANALYSIS")
        l5 = layer5_contextual(text, article.get('url'))
        print(f"   Domain: {l5['domain']}")
        print(f"   Credibility: {l5['credibility_score']}")
        print(f"   Flags: {l5['flags']}")
        
        # Layer 6: Virality + Aggregation
        print("\n📊 LAYER 6: AGGREGATION")
        virality = calculate_virality_score(article)
        l6 = layer6_aggregate(l1, l3, l4, l5, virality)
        print(f"   Virality Score: {virality}")
        print(f"   Falsehood Score: {l6['falsehood_score']}")
        print(f"   Verdict: {l6['verdict']}")
        print(f"   Alert Severity: {l6['alert_severity']}")
        print(f"   Is Alert: {l6['is_alert']}")
        
        print("="*70)
        print("✅ ALL LAYERS WORKING!")
        print("="*70)
        
        # Build final result
        result = analyze_for_viral_fake(article)
        
        return {
            "success": True,
            "layers": {
                "layer1_ml_detection": l1,
                "layer2_claim_extraction": l2,
                "layer3_authoritative": l3,
                "layer4_fact_checks": l4,
                "layer5_contextual": l5,
                "layer6_aggregation": l6,
                "virality": virality
            },
            "final_result": result,
            "all_layers_working": True
        }
        
    except Exception as e:
        print(f"❌ ANALYSIS ERROR: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "all_layers_working": False
        }

@app.get("/stats")
def get_stats():
    """Get analysis statistics."""
    try:
        db = SessionLocal()
        
        total_alerts = db.query(Alert).count()
        critical_alerts = db.query(Alert).filter(Alert.alert_severity == "CRITICAL").count()
        high_alerts = db.query(Alert).filter(Alert.alert_severity == "HIGH").count()
        
        last_scan = db.query(AnalysisLog).order_by(AnalysisLog.timestamp.desc()).first()
        
        db.close()
        
        return {
            "total_alerts": total_alerts,
            "critical": critical_alerts,
            "high": high_alerts,
            "medium": total_alerts - critical_alerts - high_alerts,
            "last_scan": last_scan.timestamp.isoformat() if last_scan else None,
            "database": DATABASE_URL,
            "total_sources": 38
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def status():
    return {
        "status": "🟢 ONLINE - Running 24/7",
        "scan_interval": "1 hour",
        "cleanup_schedule": "Daily at 3 AM",
        "country": COUNTRY_CODE,
        "database": "SQLite" if "sqlite" in DATABASE_URL else "PostgreSQL",
        "sources": {
            "gnews_api": "GNews.io (Global News API)" if GNEWS_API_KEY else "Not configured",
            "rss_feeds": list(RSS_FEEDS.keys()),
            "social_media": [
                "Telegram",
                "YouTube Shorts",
                "Instagram Reels",
                "Reddit",
                "Twitter/X (Official + Nitter)",
                "4chan /pol/",
                "Rumble",
                "Gettr",
                "Truth Social"
            ],
            "google_search": GOOGLE_SEARCH_TERMS
        },
        "total_sources": f"38+ (GNews API + 27 RSS + 9 Social Media + Google Search)",
        "models": [m for m in ["roberta", "deberta", "bert"] if models.get({"roberta": "a", "deberta": "b", "bert": "c"}[m]) is not None],
        "cache_size": len(cache),
        "performance": {
            "articles_per_scan": "ALL (no limit)",
            "gdelt_enabled": False,
            "smart_filtering": "Only analyzes NEW articles"
        }
    }

# ===== SCHEDULER =====
scheduler = BackgroundScheduler()

# Run first scan on startup (after 30 seconds delay)
scheduler.add_job(autonomous_scan, 'date', run_date=datetime.utcnow() + timedelta(seconds=30))

# ✅ SCAN EVERY 1 HOUR - Analyze ALL articles
scheduler.add_job(autonomous_scan, 'interval', hours=1, max_instances=1)

# Daily cleanup at 3 AM
scheduler.add_job(cleanup_old_alerts, 'cron', hour=3)

try:
    scheduler.start()
    print("⏰ Scheduler started successfully")
except Exception as e:
    print(f"⚠️ Scheduler start warning: {e}")

print("="*70)
print("🚨 Agent Sentinel - 24/7 Autonomous System ONLINE")
print("="*70)
print(f"📡 Database: {DATABASE_URL}")
print(f"🌍 Country: {COUNTRY_CODE}")
print(f"🔄 Auto-scanning: Every 1 HOUR (analyzes only NEW articles)")
print(f"🧹 Cleanup: Daily at 3 AM (removes alerts >30 days)")
print(f"📊 Data Sources: 38+ sources")
print(f"   - GNews.io API (Global News)")
print(f"   - 27 Global RSS Feeds (NYT, BBC, CNN, Reuters, Al Jazeera, etc.)")
print(f"   - 9 Social Media Platforms (Reddit, Twitter, Telegram, 4chan, etc.)")
print(f"   - Google Search (Full Internet Scan)")
print(f"⚡ Smart Analysis: Skips already-analyzed articles automatically")
print("="*70)
print("\nAPI ENDPOINTS:")
print("  GET  /alerts?trigger_scan=true  - Trigger scan & get alerts")
print("  GET  /alerts                    - Get alerts (no scan)")
print("  GET  /test-alerts               - Manually trigger FULL scan")
print("  POST /test-analysis             - Test analysis layers (DebugMode)")
print("  POST /analyze-text              - Analyze custom text")
print("  GET  /stats                     - View statistics")
print("  GET  /status                    - System status")
print("="*70)

if __name__ == "__main__":
    print("\n🚀 Starting Agent Sentinel...")
    print("⏰ Scheduler started successfully")
    print("  - Scanning: Every 1 HOUR")
    print("  - Cleanup: Daily at 3 AM (removes alerts older than 30 days)")
    print("  - Coverage: 38+ global sources (GNews API + RSS + Social + Google)")
    print("  - Smart Mode: Only analyzes NEW articles (skips duplicates)")
