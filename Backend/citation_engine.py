from typing import List
from models import NewsSource, VerificationResult

class CitationEngine:
    """
    THE GENIUS FEATURE: Automatically inject citations to counter false claims
    This is what makes Agent Sentinel government-ready
    """
    
    @staticmethod
    def generate_citations(verification: VerificationResult) -> List[str]:
        """
        Generate properly formatted citations from verification results
        """
        citations = []
        
        # Add supporting sources
        for source in verification.sources:
            citation = f"✓ {source.title} - {source.url}"
            citations.append(citation)
        
        # Add contradicting sources (these disprove false claims)
        for source in verification.contradicting_sources:
            citation = f"✗ Contradicted by: {source.title} - {source.url}"
            citations.append(citation)
        
        return citations
    
    @staticmethod
    def generate_press_release_citations(verification: VerificationResult) -> str:
        """
        Generate citation block for government press releases
        """
        citation_text = "\n\nSOURCES:\n"
        
        for idx, source in enumerate(verification.sources, 1):
            citation_text += f"[{idx}] {source.title}\n"
            citation_text += f"    {source.url}\n"
            if source.published_date:
                citation_text += f"    Published: {source.published_date}\n"
        
        for idx, source in enumerate(verification.contradicting_sources, 
                                     len(verification.sources) + 1):
            citation_text += f"[{idx}] {source.title} (Contradicts claim)\n"
            citation_text += f"    {source.url}\n"
        
        return citation_text
    
    @staticmethod
    def generate_social_media_citation(verification: VerificationResult, 
                                       platform: str = "twitter") -> str:
        """
        Generate citation optimized for social media character limits
        """
        if platform == "twitter":
            # Twitter/X optimized (280 chars)
            if verification.sources:
                source = verification.sources[0]
                return f"Verified by {source.domain}: {source.url}"
            elif verification.contradicting_sources:
                source = verification.contradicting_sources[0]
                return f"False. See {source.domain}: {source.url}"
        
        return "See official sources for verification"

citation_engine = CitationEngine()