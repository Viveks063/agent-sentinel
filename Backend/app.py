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

# Scheduler for background viral news fetching
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

# --- CONFIG ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")  # ✅ ADD THIS LINE
COUNTRY_CODE = os.getenv("COUNTRY_CODE", "IN").upper()
USE_LIGHT_MODE = os.getenv("USE_LIGHT_MODE", "1") == "1"

# Models
LIGHT_MODEL_A = "roberta-large-mnli"
LIGHT_MODEL_B = "microsoft/deberta-v3-small"
LIGHT_MODEL_C = "distilbert-base-uncased-finetuned-sst-2-english"

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

app = FastAPI(title="Agent Sentinel — Viral Fake News Alert System")
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

print("Loading ML models...")
models["a"] = try_load_pipeline(LIGHT_MODEL_A, "text-classification")
models["b"] = try_load_pipeline(LIGHT_MODEL_B, "text-classification")
models["c"] = try_load_pipeline(LIGHT_MODEL_C, "text-classification")
print("Models loaded.")

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
    
    def is_fake_label(l):
        return any(x in (l or "").upper() for x in ["FAKE", "LABEL_1", "MISINFORMATION", "FALSE"])
    
    def model_fake_prob(m):
        lab = m.get("label", "")
        sc = m.get("score", 0.5)
        return sc if is_fake_label(lab) else (1.0 - sc)
    
    w_roberta, w_deberta, w_bert = 0.35, 0.40, 0.25
    ensemble = (model_fake_prob(outputs["roberta"]) * w_roberta + 
                model_fake_prob(outputs["deberta"]) * w_deberta + 
                model_fake_prob(outputs["bert"]) * w_bert)
    ensemble = round(max(0.0, min(1.0, ensemble)), 3)
    
    return {
        "ensemble_label": "FAKE" if ensemble >= 0.6 else "REAL",
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
    return {"matches": []}

def layer5_contextual(text: str, url: Optional[str] = None) -> Dict[str, Any]:
    domain = url.split("/")[2] if url else "unknown"
    credibility = TRUSTED_SOURCES.get(domain, {}).get("credibility", 0.4)
    return {"domain": domain, "credibility_score": round(credibility, 2), "flags": [], "is_trusted_domain": domain in TRUSTED_SOURCES}

def layer6_aggregate(l1, l3, l4, l5, virality) -> Dict[str, Any]:
    model_fake = l1.get("ensemble_score", 0.5)
    falsehood = model_fake * (1 - 0.3 * l5.get("credibility_score", 0.5))
    falsehood = round(max(0.0, min(1.0, falsehood)), 3)
    
    is_alert = (falsehood >= 0.35 and virality >= 0.25) or (falsehood >= 0.55)
    
    if falsehood >= 0.65 and virality >= 0.45:
        alert_severity = "CRITICAL"
    elif falsehood >= 0.5 or virality >= 0.5:
        alert_severity = "HIGH"
    elif is_alert:
        alert_severity = "MEDIUM"
    else:
        alert_severity = "LOW"
    
    return {
        "falsehood_score": falsehood,
        "virality_score": virality,
        "verdict": "FAKE" if falsehood >= 0.35 else "REAL",
        "is_alert": is_alert,
        "alert_severity": alert_severity,
    }

def build_alert_report(article: Dict[str, Any], l1, l2, l3, l4, l5, l6) -> str:
    lines = [
        "🚨 VIRAL FAKE NEWS ALERT 🚨\n",
        "=" * 70 + "\n\n",
        f"HEADLINE: {article.get('title', 'N/A')}\n",
        f"SOURCE: {article.get('source', 'Unknown')}\n",
        f"URL: {article.get('url', 'N/A')}\n\n",
        "VERDICT: " + ("🔴 FAKE NEWS" if l6['verdict'] == "FAKE" else "🟢 REAL NEWS") + "\n",
        f"Falsehood Score: {l6['falsehood_score']*100:.0f}%\n",
        f"Virality Score: {l6['virality_score']*100:.0f}%\n",
        f"Alert Severity: {l6['alert_severity']}\n\n",
        "Report Generated: " + datetime.now().isoformat() + "\n"
    ]
    return "\n".join(lines)

def analyze_for_viral_fake(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = short_hash(article.get('title', '') + article.get('url', ''))
    if key in cache:
        return cache[key]
    
    try:
        title = article.get('title', 'Unknown')
        description = article.get('description', '')
        text = f"{title} {description}"
        
        # Get source
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
        
        # Build report with actual data
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

Report Generated: {datetime.now().isoformat()}
================================================================================"""
        
        result = {
            "id": key,
            "title": title,
            "url": article.get('url', '#'),
            "source": {"name": source_name},
            "published_at": article.get('publishedAt', datetime.now().isoformat()),
            "virality_score": virality,
            "falsehood_score": l6['falsehood_score'],
            "is_alert": l6['is_alert'],
            "alert_severity": l6['alert_severity'],
            "report": report,
            "analyzed_at": datetime.now().isoformat()
        }
        
        cache[key] = result
        return result
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        return None

# =========================================
# ALL DATA SOURCES (INDIA OPTIMIZED)
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
                        "publishedAt": tweet["created_at"]
                    })
            
            print(f"  ✓ Got {len(articles)} conspiracy tweets from official API")
        elif response.status_code == 429:
            print("  ⚠️ Twitter API rate limit hit — switching to Nitter fallback...")
            return fetch_from_twitter()  # ✅ fallback to your existing unofficial Nitter scraper
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
                            "publishedAt": datetime.now().isoformat()
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
                        "source": {"name": f"Reddit - r/{sub}"},
                        "publishedAt": datetime.fromtimestamp(data.get("created_utc", 0)).isoformat()
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
                        "publishedAt": datetime.fromtimestamp(thread.get("time", 0)).isoformat()
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
                    "publishedAt": snippet.get("publishedAt", datetime.now().isoformat())
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
                    "publishedAt": datetime.now().isoformat()
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
                            "publishedAt": datetime.now().isoformat()
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
                            "publishedAt": datetime.now().isoformat()
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
                "publishedAt": datetime.now().isoformat()
            })
            print(f"  ✓ Fetched Gettr")
    except Exception as e:
        print(f"  ✗ Gettr error: {str(e)}")
    
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
                            "publishedAt": post.get("created_at", datetime.now().isoformat())
                        })
            print(f"  ✓ Got {len(posts)} posts from Truth Social")
    except Exception as e:
        print(f"  ✗ Truth Social error: {str(e)}")
    
    return articles

def fetch_viral_news() -> List[Dict[str, Any]]:
    """Fetch from ALL sources."""
    
    print("\n" + "="*70)
    print(f"[FETCH_VIRAL_NEWS] Starting for COUNTRY: {COUNTRY_CODE}")
    print("="*70)
    
    all_articles = []
    
    # TIER 1 - CRITICAL (India)
    print("\n[TIER 1 - CRITICAL]")
    all_articles.extend(fetch_from_telegram())
    all_articles.extend(fetch_from_youtube_shorts())
    all_articles.extend(fetch_from_instagram())
    all_articles.extend(fetch_from_reddit())
    
    # TIER 2 - SECONDARY
    print("\n[TIER 2 - SECONDARY]")
    # ✅ USE OFFICIAL TWITTER API if token available, else fallback to Nitter
    if TWITTER_BEARER_TOKEN:
        print("[TWITTER] Using Official API")
        all_articles.extend(fetch_from_twitter_official())
    else:
        print("[TWITTER] Using Nitter (no official API key)")
        all_articles.extend(fetch_from_twitter())
    all_articles.extend(fetch_from_4chan())
    
    # TIER 3 - BONUS
    print("\n[TIER 3 - BONUS]")
    all_articles.extend(fetch_from_rumble())
    all_articles.extend(fetch_from_gettr())
    all_articles.extend(fetch_from_truth_social())
    
    print(f"\n[AGGREGATION] Total articles: {len(all_articles)}")
    
    # Deduplicate
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title = art.get('title', '')
        if title and title not in seen_titles and len(title) > 5:
            seen_titles.add(title)
            unique_articles.append(art)
    
    print(f"[DEDUPLICATION] Unique: {len(unique_articles)}")
    
    # Filter virality
    viral_articles = [a for a in unique_articles if calculate_virality_score(a) >= 0.15]
    print(f"[VIRAL FILTER] {len(viral_articles)} viral articles")
    
    return viral_articles

# =========================================
# API ENDPOINTS
# =========================================

class AnalyzeRequest(BaseModel):
    text: str
    url: Optional[str] = None
    title: Optional[str] = None

@app.post("/analyze-text")
def analyze_text(req: AnalyzeRequest):
    """Analyze custom text for fake news."""
    article = {"title": req.title or req.text[:100], "description": req.text, "url": req.url}
    result = analyze_for_viral_fake(article)
    
    if not result:
        raise HTTPException(500, "Analysis failed")
    
    return result

@app.get("/alerts")
@app.post("/alerts")
def get_viral_alerts():
    """Get viral fake news alerts."""
    global viral_alerts
    
    print("\n" + "="*70)
    print("[ALERT ENDPOINT] Fetching viral misinformation...")
    
    try:
        viral_news = fetch_viral_news()
        print(f"[FETCH] Got {len(viral_news)} viral articles")
        
        alerts = []
        alert_count = 0
        
        for article in viral_news:
            result = analyze_for_viral_fake(article)
            if result and result['is_alert']:
                alert_count += 1
                
                # Get full report
                full_report = result.get('report', f"""
🚨 VIRAL FAKE NEWS ALERT 🚨

HEADLINE: {result.get('title', 'Untitled')}
SOURCE: {result.get('source', 'Unknown')}
URL: {result.get('url', 'N/A')}

VERDICT: {'🔴 FAKE NEWS' if result.get('is_alert') else '🟢 REAL NEWS'}
Falsehood Score: 33%
Virality Score: 33%
Alert Severity: {result.get('alert_severity', 'MEDIUM')}

Report Generated: {datetime.now().isoformat()}
""")
                
                alert_obj = {
                    "id": result.get('id', f'alert_{alert_count}'),
                    "title": result.get('title', 'Untitled Alert'),
                    "source": result.get('source', {'name': 'Unknown'}),
                    "url": result.get('url', '#'),
                    "severity": result.get('alert_severity', 'MEDIUM'),
                    "falsehood_score": result.get('falsehood_score', 0.33),  # ✅ Use actual score
                    "virality_score": result.get('virality_score', 0.33),  # ✅ Use actual score
                    "report_summary": full_report[:500],
                    "report": full_report,
                    "published_at": result.get('published_at', datetime.now().isoformat())
                }
                alerts.append(alert_obj)
                print(f"  ✓ ALERT {alert_count}: {alert_obj['title'][:60]}")
        
        viral_alerts = alerts
        print(f"[RESULT] Total alerts: {len(alerts)}\n")
        
        return {
            "total_alerts": len(alerts),
            "alerts": alerts,
            "fetched_at": datetime.now().isoformat(),
            "sources": 9
        }
    
    except Exception as e:
        print(f"[ERROR] {str(e)}\n")
        traceback.print_exc()
        return {"error": str(e), "alerts": [], "total_alerts": 0}

@app.get("/test-alerts")
def test_alerts():
    """Alias for /alerts."""
    return get_viral_alerts()

@app.get("/status")
def status():
    return {
        "status": "online",
        "country": COUNTRY_CODE,
        "sources": [
            "Telegram (CRITICAL)",
            "YouTube Shorts",
            "Instagram Reels",
            "Reddit Conspiracy",
            "Twitter/X (Nitter)",
            "4chan /pol/",
            "Rumble",
            "Gettr",
            "Truth Social"
        ],
        "total_sources": 9,
        "cache_size": len(cache),
        "alerts_cached": len(viral_alerts)
    }

scheduler = BackgroundScheduler()

def update_viral_alerts():
    print("[SCHEDULER] Fetching viral alerts...")
    try:
        viral_news = fetch_viral_news()
        for article in viral_news:
            analyze_for_viral_fake(article)
    except Exception as e:
        print(f"Scheduler error: {e}")

scheduler.add_job(update_viral_alerts, 'interval', minutes=30)
scheduler.start()

if __name__ == "__main__":
    print("🚨 Agent Sentinel - Viral Fake News Alert System")
    print(f"🇮🇳 Country: {COUNTRY_CODE}")
    print("📡 Sources: 9 (Telegram, Reddit, 4chan, YouTube, Instagram, Twitter, Rumble, Gettr, Truth Social)")
    print("GET /alerts - Fetch all viral fake news")
    print("GET /status - Check system status")