import aiohttp
import urllib.parse
from typing import Dict, Any

class GDELTMonitor:
    """
    Global Event Monitor using GDELT Project API
    """
    
    # Correct API Endpoint for GDELT Doc API
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    async def check_event_coverage(self, headline: str) -> Dict[str, Any]:
        """
        Check if the world is talking about this headline
        """
        # Clean headline for query
        # Remove special chars and keep it short
        clean_query = "".join(e for e in headline if e.isalnum() or e.isspace())
        keywords = " ".join(clean_query.split()[:6]) # First 6 words usually contain the subject
        
        params = {
            "query": f'"{keywords}" sourcelang:eng',
            "mode": "artlist",
            "maxrecords": "10",
            "format": "json",
            "timespan": "24h"
        }
        
        # Headers are CRITICAL to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params, headers=headers) as response:
                    
                    if response.status != 200:
                        # Fail gracefully
                        print(f"GDELT API Status: {response.status}")
                        return self._empty_result()
                        
                    try:
                        data = await response.json()
                    except:
                        # If response is not JSON (e.g. HTML error page)
                        return self._empty_result()

                    articles = data.get("articles", [])
                    
                    trusted_domains = ["bbc", "reuters", "cnn", "aljazeera", "apnews", "hindu", "timesofindia"]
                    trusted_count = sum(1 for a in articles if any(t in a.get("domain", "") for t in trusted_domains))
                    
                    return {
                        "has_coverage": len(articles) > 0,
                        "total_articles": len(articles),
                        "trusted_articles": trusted_count,
                        "coverage_ratio": trusted_count / len(articles) if articles else 0,
                        "articles": articles[:3]
                    }
                    
        except Exception as e:
            print(f"GDELT Connection Error: {e}")
            return self._empty_result()

    def _empty_result(self):
        return {
            "has_coverage": False,
            "total_articles": 0,
            "trusted_articles": 0,
            "coverage_ratio": 0,
            "articles": []
        }

# Singleton
gdelt_monitor = GDELTMonitor()