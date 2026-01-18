# src/models/social_message.py
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"
    GROK = "grok"
    RESEARCH = "research"  # Morning Research Agent


# Common crypto tickers to exclude
CRYPTO_TICKERS = {
    "BTC", "ETH", "DOGE", "SOL", "XRP", "ADA", "AVAX", "DOT",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "BCH", "SHIB", "PEPE"
}


class SocialMessage(BaseModel):
    """Represents a message from any social media source."""

    source: SourceType
    source_id: str
    author: str
    content: str
    timestamp: datetime
    url: Optional[str] = None

    # Reddit-specific
    subreddit: Optional[str] = None
    upvotes: Optional[int] = None
    comment_count: Optional[int] = None

    # Twitter-specific
    retweet_count: Optional[int] = None
    like_count: Optional[int] = None

    # Stocktwits-specific
    sentiment: Optional[str] = None  # "bullish" or "bearish"

    # Extracted data (populated later)
    extracted_tickers: list[str] = Field(default_factory=list)

    def extract_tickers(self, exclude_crypto: bool = True) -> list[str]:
        """Extract stock tickers from content.

        Args:
            exclude_crypto: If True, excludes common crypto tickers.

        Returns:
            List of unique tickers in order of appearance.
        """
        # Match $TICKER pattern (1-5 uppercase letters)
        pattern = r'\$([A-Z]{1,5})\b'
        matches = re.findall(pattern, self.content)

        # Remove duplicates while preserving order
        seen = set()
        tickers = []
        for ticker in matches:
            if ticker not in seen:
                if exclude_crypto and ticker in CRYPTO_TICKERS:
                    continue
                seen.add(ticker)
                tickers.append(ticker)

        return tickers
