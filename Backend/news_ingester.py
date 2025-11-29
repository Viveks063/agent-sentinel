import feedparser
import asyncio
from datetime import datetime
import hashlib
from typing import List, Set

from agent_core import agent_core
from models import AlertLevel

class NewsIngester:
    """
    THE EYES: Autonomous News Ingestion System
    Continuously monitors RSS feeds and Social Streams
    """
    def __init__(self):
        self.seen_urls: Set[str] = set()
        self.is_running = False
        
        # Free Real-Time Sources (No API Key needed)
        self.sources = [
            # Google News (India/World) - The firehose
            "https://news.google.com/rss/search?q=when:1h&hl=en-IN&gl=IN&ceid=IN:en",
            
            # Specific Topic Feeds (Risk Monitoring)
            "https://news.google.com/rss/search?q=cyberattack+OR+outage&hl=en-IN&gl=IN&ceid=IN:en",
            "https://news.google.com/rss/search?q=riot+OR+protest+Mumbai&hl=en-IN&gl=IN&ceid=IN:en",
            
            # Tech/Crypto (High fraud areas)
            "https://feeds.feedburner.com/TheHackersNews",
        ]

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start the autonomous monitoring loop"""
        self.is_running = True
        print(f"👁️ Agent Sentinel Watchtower active. Monitoring {len(self.sources)} streams...")
        
        while self.is_running:
            try:
                await self._scan_feeds()
            except Exception as e:
                print(f"⚠️ Ingestion error: {e}")
            
            await asyncio.sleep(interval_seconds)

    async def _scan_feeds(self):
            for source in self.sources:
                try:
                    feed = feedparser.parse(source)
                    
                    # Limit to 3 items per feed to prevent flooding
                    for entry in feed.entries[:3]:
                        await self._process_entry(entry)
                        # CRITICAL: Wait 5 seconds between items to respect API limits
                        print(f"⏳ Cooling down for 5s to avoid rate limits...")
                        await asyncio.sleep(5) 
                        
                except Exception as e:
                    print(f"Feed error ({source}): {e}")

    async def _process_entry(self, entry):
            url = entry.link
            if url in self.seen_urls:
                return
            self.seen_urls.add(url)
            
            headline = entry.title
            content = getattr(entry, 'summary', '') + " " + getattr(entry, 'description', '')
            
            print(f"🔎 New Signal Detected: {headline[:50]}...")
            
            # Run analysis (Await it here instead of create_task to force sequential processing)
            await self._analyze_signal(headline, content, url)

    async def _analyze_signal(self, headline: str, content: str, url: str):
        try:
            analysis = await agent_core.analyze_news(
                headline=headline,
                content=content,
                source_url=url,
                enable_counter_narrative=True
            )
            
            # Log significant findings
            if analysis.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
                print(f"🚨 THREAT DETECTED: {headline} (Score: {analysis.falsehood_score})")
            else:
                print(f"✅ Verified Safe: {headline}")
                
        except Exception as e:
            print(f"Analysis failed for {headline}: {e}")

# Singleton
news_ingester = NewsIngester()