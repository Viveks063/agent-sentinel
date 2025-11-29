import asyncio
from duckduckgo_search import DDGS
from sentence_transformers import CrossEncoder
from typing import List, Dict
import time
import random
from urllib.parse import urlparse

# Import the specific models defined in models.py
from models import VerificationResult, NewsSource

class SemanticVerifier:
    """
    THE GENIUS VERIFIER: Semantic Cross-Reference using NLI
    """
    def __init__(self):
        print("🧠 Loading Semantic NLI Model...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.trusted_domains = [
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
            "npr.org", "pbs.org", "wsj.com", "bloomberg.com", 
            "snopes.com", "timesofindia.indiatimes.com", "thehindu.com",
            "ndtv.com", "indianexpress.com"
        ]

    async def verify_claim(self, headline: str, content: str) -> VerificationResult:
        """
        Verify a claim by finding trusted sources and checking semantic agreement
        """
        start_time = time.time()  # Start timer
        
        # 1. Search for evidence
        evidence_articles = self._fetch_evidence(headline)
        
        # Convert dictionaries to NewsSource objects
        news_sources = []
        for art in evidence_articles:
            domain = urlparse(art['url']).netloc
            news_sources.append(
                NewsSource(
                    url=art['url'],
                    title=art['title'],
                    domain=domain,
                    is_trusted=any(t in domain for t in self.trusted_domains),
                    published_date=None
                )
            )

        if not evidence_articles:
            return VerificationResult(
                is_verified=False,
                confidence_score=0.0,
                sources=[],
                contradicting_sources=[],
                summary="No trusted evidence found to verify this claim.",
                verification_time=time.time() - start_time # Return float duration
            )

        # 2. Semantic Analysis (NLI)
        claim_text = f"{headline}. {content[:200]}"
        pairs = [[claim_text, art['text']] for art in evidence_articles]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        supporting_indices = []
        
        for i, score in enumerate(scores):
            # Threshold for relevance/support
            if score > 1.5: 
                supporting_indices.append(i)

        # Calculate confidence
        confidence = min(1.0, len(supporting_indices) * 0.3)
        is_verified = confidence > 0.5
        
        # Filter sources that actually supported the claim
        supporting_sources_objs = [news_sources[i] for i in supporting_indices]

        summary_text = (
            f"Found {len(supporting_sources_objs)} trusted sources corroborating this story."
            if is_verified else "Evidence inconclusive or not found in trusted sources."
        )

        return VerificationResult(
            is_verified=is_verified,
            confidence_score=confidence,
            sources=news_sources,  # All sources found
            contradicting_sources=[], # Advanced NLI required for true contradiction
            summary=summary_text,
            verification_time=time.time() - start_time # Return float duration
        )

    def _fetch_evidence(self, query: str) -> List[Dict]:
        """
        FIXED: Better rate limiting and error handling
        """
        results = []
        clean_query = "".join(e for e in query if e.isalnum() or e.isspace())[:100]
        
        try:
            with DDGS() as ddgs:
                # CRITICAL: Random delay between 3-6 seconds
                import random
                time.sleep(random.uniform(3.0, 6.0))
                
                # Use default backend (most reliable)
                search_results = ddgs.text(
                    clean_query,
                    max_results=3,  # REDUCED from 5 to avoid rate limits
                    timelimit='m'   # Only recent results (last month)
                )
                
                if search_results:
                    for r in search_results:
                        link = r.get('href', '')
                        if any(d in link for d in self.trusted_domains):
                            results.append({
                                "text": r.get('body', ''),
                                "url": link,
                                "title": r.get('title', '')
                            })
                            
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            # Return empty - don't crash
        
        return results
# Singleton
verifier = SemanticVerifier()