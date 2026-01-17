# tests/integration/test_validation_pipeline.py
"""Integration tests for the full Phase 2 → Phase 3 validation pipeline."""

import pytest
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime, timezone

from src.validators.technical_validator import TechnicalValidator
from src.validators.models import ValidationStatus, OptionsFlowData
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.models.social_message import SocialMessage, SourceType


class TestValidationPipelineIntegration:
    """Integration tests for Phase 2 → Phase 3 pipeline.

    Tests the full flow:
    1. AnalyzedMessage (from Phase 2) with sentiment analysis
    2. TechnicalValidator (Phase 3) validates against market data
    3. Returns ValidatedSignal with final decision
    """

    @pytest.fixture
    def mock_alpaca_client(self):
        """Create a mock Alpaca client that returns realistic OHLC data."""
        client = MagicMock()
        return client

    @pytest.fixture
    def bullish_message(self):
        """Create a bullish analyzed message."""
        return AnalyzedMessage(
            message=SocialMessage(
                source=SourceType.TWITTER,
                source_id="tweet_12345",
                author="WallStreetGuru",
                content="$AAPL breaking out above resistance! Strong momentum building. Price target $200. #bullish",
                timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
                url="https://twitter.com/WallStreetGuru/status/12345",
                like_count=523,
                retweet_count=87,
                extracted_tickers=["AAPL"],
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.92,
                confidence=0.88,
            ),
        )

    @pytest.fixture
    def bearish_message(self):
        """Create a bearish analyzed message."""
        return AnalyzedMessage(
            message=SocialMessage(
                source=SourceType.REDDIT,
                source_id="post_67890",
                author="BearishTrader",
                content="$TSLA looking weak here. Breaking down from support. Expecting further downside to $150.",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                url="https://reddit.com/r/wallstreetbets/comments/67890",
                subreddit="wallstreetbets",
                upvotes=342,
                comment_count=56,
                extracted_tickers=["TSLA"],
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BEARISH,
                score=0.15,
                confidence=0.85,
            ),
        )

    @pytest.fixture
    def neutral_ohlc_data(self):
        """Create neutral OHLC data (sideways market)."""
        # 50 bars of sideways trading
        return pd.DataFrame({
            'open': [150.0 + (i % 5) for i in range(50)],
            'high': [152.0 + (i % 5) for i in range(50)],
            'low': [148.0 + (i % 5) for i in range(50)],
            'close': [150.5 + (i % 5) for i in range(50)],
            'volume': [1000000 + (i * 10000) for i in range(50)],
        })

    @pytest.fixture
    def bullish_ohlc_data(self):
        """Create bullish OHLC data (uptrend with healthy RSI)."""
        # 50 bars of uptrend with realistic pullbacks (RSI ~55-65, ADX > 20)
        # Pattern: 2 up, 1 down with balanced moves keeps RSI healthy
        close_prices = []
        base = 100.0
        for i in range(50):
            if i % 3 == 2:  # Every 3rd bar is a pullback
                base -= 1.2  # Significant pullback
            else:
                base += 1.0  # Moderate gains
            close_prices.append(base)

        return pd.DataFrame({
            'open': [p - 0.4 for p in close_prices],
            'high': [p + 0.7 for p in close_prices],
            'low': [p - 0.7 for p in close_prices],
            'close': close_prices,
            'volume': [1000000 + (i * 15000) for i in range(50)],
        })

    @pytest.fixture
    def overbought_ohlc_data(self):
        """Create overbought OHLC data (strong rally, RSI > 70)."""
        # 50 bars with steep rally (will create RSI > 70)
        close_prices = [100.0]
        for i in range(1, 50):
            # Create steep uptrend that will push RSI above 70
            close_prices.append(close_prices[-1] * 1.02)  # 2% gains

        return pd.DataFrame({
            'open': [p - 0.5 for p in close_prices],
            'high': [p + 0.8 for p in close_prices],
            'low': [p - 0.3 for p in close_prices],
            'close': close_prices,
            'volume': [2000000 + (i * 50000) for i in range(50)],
        })

    @pytest.fixture
    def oversold_ohlc_data(self):
        """Create oversold OHLC data (heavy selloff, RSI < 30)."""
        # 50 bars with steep decline (will create RSI < 30)
        close_prices = [150.0]
        for i in range(1, 50):
            # Create steep downtrend that will push RSI below 30
            close_prices.append(close_prices[-1] * 0.98)  # 2% losses

        return pd.DataFrame({
            'open': [p + 0.5 for p in close_prices],
            'high': [p + 0.8 for p in close_prices],
            'low': [p - 0.8 for p in close_prices],
            'close': close_prices,
            'volume': [2500000 + (i * 60000) for i in range(50)],
        })

    @pytest.fixture
    def weak_trend_ohlc_data(self):
        """Create choppy OHLC data (no clear trend, low ADX)."""
        # 50 bars of choppy, range-bound trading
        import random
        random.seed(42)

        close_prices = []
        base = 100.0
        for i in range(50):
            # Random walk within tight range (creates low ADX)
            change = random.uniform(-0.5, 0.5)
            close_prices.append(base + change)

        return pd.DataFrame({
            'open': [p + random.uniform(-0.3, 0.3) for p in close_prices],
            'high': [p + abs(random.uniform(0.2, 0.6)) for p in close_prices],
            'low': [p - abs(random.uniform(0.2, 0.6)) for p in close_prices],
            'close': close_prices,
            'volume': [1000000 + random.randint(-100000, 100000) for _ in range(50)],
        })

    def test_full_pipeline_pass(self, mock_alpaca_client, bullish_message, bullish_ohlc_data):
        """Test full pipeline: bullish signal with favorable technicals → PASS.

        Scenario:
        - Bullish sentiment from Phase 2 (confidence 0.88)
        - Healthy uptrend with RSI around 55-65 (not overbought)
        - Strong ADX (trending market)
        - Expected result: PASS
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = bullish_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
            lookback_bars=50,
            timeframe="5Min",
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
        )

        # Assert - should PASS
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.veto_reasons) == 0
        assert result.should_trade() is True

        # Verify technical indicators are favorable
        indicators = result.validation.indicators
        assert 40.0 <= indicators.rsi <= 70.0  # Not overbought
        assert indicators.adx >= 20.0  # Strong trend

        # Verify the original message is preserved
        assert result.message == bullish_message
        assert result.message.sentiment.label == SentimentLabel.BULLISH
        assert result.message.sentiment.confidence == 0.88

        # Verify Alpaca was called correctly
        mock_alpaca_client.get_bars.assert_called_once()
        call_kwargs = mock_alpaca_client.get_bars.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["timeframe"] == "5Min"
        assert call_kwargs["limit"] == 50

    def test_full_pipeline_veto_overbought(self, mock_alpaca_client, bullish_message, overbought_ohlc_data):
        """Test full pipeline: bullish signal with overbought RSI → VETO.

        Scenario:
        - Bullish sentiment from Phase 2
        - Overbought conditions (RSI > 70) after steep rally
        - Expected result: VETO (prevent buying at the top)
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = overbought_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
        )

        # Assert - should VETO
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert result.should_trade() is False

        # Verify RSI veto reason
        veto_text = " ".join(result.validation.veto_reasons).lower()
        assert "rsi" in veto_text
        assert "overbought" in veto_text

        # Verify RSI is indeed overbought
        assert result.validation.indicators.rsi > 70.0

        # Original bullish sentiment should be preserved
        assert result.message.sentiment.label == SentimentLabel.BULLISH

    def test_full_pipeline_veto_oversold_bearish(self, mock_alpaca_client, bearish_message, oversold_ohlc_data):
        """Test full pipeline: bearish signal with oversold RSI → VETO.

        Scenario:
        - Bearish sentiment from Phase 2
        - Oversold conditions (RSI < 30) after heavy selloff
        - Expected result: VETO (prevent shorting at the bottom)
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = oversold_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bearish_message,
            symbol="TSLA",
        )

        # Assert - should VETO
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert result.should_trade() is False

        # Verify RSI veto reason
        veto_text = " ".join(result.validation.veto_reasons).lower()
        assert "rsi" in veto_text
        assert "oversold" in veto_text

        # Verify RSI is indeed oversold
        assert result.validation.indicators.rsi < 30.0

        # Original bearish sentiment should be preserved
        assert result.message.sentiment.label == SentimentLabel.BEARISH

    def test_full_pipeline_veto_weak_trend(self, mock_alpaca_client, bullish_message, weak_trend_ohlc_data):
        """Test full pipeline: bullish signal with weak trend (low ADX) → VETO.

        Scenario:
        - Bullish sentiment from Phase 2
        - Choppy, range-bound market with low ADX
        - Expected result: VETO (no clear trend to follow)
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = weak_trend_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
        )

        # Assert - should VETO
        assert result.validation.status == ValidationStatus.VETO
        assert len(result.validation.veto_reasons) > 0
        assert result.should_trade() is False

        # Verify ADX veto reason
        veto_text = " ".join(result.validation.veto_reasons).lower()
        assert "adx" in veto_text or "trend" in veto_text

        # Verify ADX is indeed weak
        assert result.validation.indicators.adx < 20.0

    def test_pipeline_with_options_data(self, mock_alpaca_client, bullish_message, bullish_ohlc_data):
        """Test pipeline with options flow data enhancement.

        Scenario:
        - Bullish signal with favorable technicals
        - Options data shows high volume and unusual activity
        - Expected: PASS with increased confidence modifier
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = bullish_ohlc_data

        # Create options data with volume spike and unusual activity
        options_data = OptionsFlowData(
            volume_ratio=3.5,  # 3.5x normal volume
            iv_rank=45.0,      # Moderate IV
            put_call_ratio=0.6,  # More calls than puts (bullish)
            unusual_activity=True,
        )

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            options_volume_spike_ratio=2.0,
            iv_rank_warning_threshold=80.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
            options_data=options_data,
        )

        # Assert - should PASS with confidence boost
        assert result.validation.status == ValidationStatus.PASS
        assert result.validation.options_flow == options_data
        assert result.should_trade() is True

        # Confidence modifier should be boosted due to:
        # - Volume spike (3.5x > 2.0 threshold)
        # - Unusual activity detected
        assert result.validation.confidence_modifier > 1.0

        # Should not have IV rank warning (45 < 80)
        warnings_text = " ".join(result.validation.warnings).lower()
        assert "iv" not in warnings_text or "rank" not in warnings_text

    def test_pipeline_with_high_iv_warning(self, mock_alpaca_client, bullish_message, bullish_ohlc_data):
        """Test pipeline with high IV rank generates warning.

        Scenario:
        - Bullish signal with favorable technicals
        - Options data shows very high IV rank (expensive options)
        - Expected: PASS but with IV warning
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = bullish_ohlc_data

        options_data = OptionsFlowData(
            volume_ratio=1.5,
            iv_rank=92.0,  # Very high IV (expensive options)
            put_call_ratio=0.8,
            unusual_activity=False,
        )

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            iv_rank_warning_threshold=80.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
            options_data=options_data,
        )

        # Assert - should PASS but with warning
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.warnings) > 0

        # Check for IV rank warning
        warnings_text = " ".join(result.validation.warnings).lower()
        assert "iv" in warnings_text or "rank" in warnings_text or "volatility" in warnings_text

    def test_warn_mode_converts_veto(self, mock_alpaca_client, bullish_message, overbought_ohlc_data):
        """Test warn mode converts VETO to WARN.

        Scenario:
        - Bullish signal with overbought conditions (would normally VETO)
        - Validator in warn mode (veto_mode=False)
        - Expected: WARN instead of VETO (allows trade but with caution)
        """
        # Arrange
        mock_alpaca_client.get_bars.return_value = overbought_ohlc_data

        # Create validator with veto_mode=False
        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=False,  # Warn mode
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            adx_trend_threshold=20.0,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
        )

        # Assert - should WARN instead of VETO
        assert result.validation.status == ValidationStatus.WARN
        assert len(result.validation.warnings) > 0
        assert len(result.validation.veto_reasons) == 0  # No veto reasons
        assert result.should_trade() is True  # Can still trade in WARN mode

        # Warnings should contain the veto information
        warnings_text = " ".join(result.validation.warnings).lower()
        assert "rsi" in warnings_text
        assert "overbought" in warnings_text

        # RSI should still be overbought
        assert result.validation.indicators.rsi > 70.0

    def test_pipeline_error_handling(self, mock_alpaca_client, bullish_message):
        """Test pipeline handles API errors gracefully.

        Scenario:
        - Bullish signal from Phase 2
        - Alpaca API fails with exception
        - Expected: PASS with warning (fail-safe behavior)
        """
        # Arrange
        mock_alpaca_client.get_bars.side_effect = Exception("Alpaca API connection timeout")

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        # Act
        result = validator.validate(
            analyzed_message=bullish_message,
            symbol="AAPL",
        )

        # Assert - should fail safely with PASS
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.warnings) > 0
        assert result.should_trade() is True  # Fail-safe allows trade

        # Warning should mention the error
        warnings_text = " ".join(result.validation.warnings).lower()
        assert "fail" in warnings_text or "error" in warnings_text

        # Indicators should be neutral/fallback values
        indicators = result.validation.indicators
        assert indicators.rsi == 50.0  # Neutral
        assert indicators.adx == 0.0   # No trend data

    def test_pipeline_neutral_sentiment_bypasses_veto(self, mock_alpaca_client, overbought_ohlc_data):
        """Test pipeline with neutral sentiment bypasses veto logic.

        Scenario:
        - Neutral sentiment from Phase 2
        - Overbought conditions (would veto bullish signal)
        - Expected: PASS (veto only applies to bullish/bearish signals)
        """
        # Arrange
        neutral_message = AnalyzedMessage(
            message=SocialMessage(
                source=SourceType.TWITTER,
                source_id="tweet_99999",
                author="MarketAnalyst",
                content="$AAPL trading sideways in a range. Waiting for direction.",
                timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                url="https://twitter.com/MarketAnalyst/status/99999",
                extracted_tickers=["AAPL"],
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                confidence=0.75,
            ),
        )

        mock_alpaca_client.get_bars.return_value = overbought_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        # Act
        result = validator.validate(
            analyzed_message=neutral_message,
            symbol="AAPL",
        )

        # Assert - should PASS (neutral bypasses veto)
        assert result.validation.status == ValidationStatus.PASS
        assert len(result.validation.veto_reasons) == 0
        assert result.should_trade() is True

        # Indicators should still show overbought
        assert result.validation.indicators.rsi > 70.0

    def test_pipeline_preserves_message_metadata(self, mock_alpaca_client, bullish_ohlc_data):
        """Test pipeline preserves all Phase 2 message metadata.

        Scenario:
        - Rich message with social engagement metrics
        - Should preserve all metadata through validation pipeline
        """
        # Arrange
        rich_message = AnalyzedMessage(
            message=SocialMessage(
                source=SourceType.REDDIT,
                source_id="reddit_post_123",
                author="DDKing",
                content="$GME mega DD: fundamentals looking strong, shorts haven't covered. PT $500+",
                timestamp=datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
                url="https://reddit.com/r/wallstreetbets/comments/123",
                subreddit="wallstreetbets",
                upvotes=15234,
                comment_count=892,
                extracted_tickers=["GME"],
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.95,
                confidence=0.92,
            ),
        )

        mock_alpaca_client.get_bars.return_value = bullish_ohlc_data

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
        )

        # Act
        result = validator.validate(
            analyzed_message=rich_message,
            symbol="GME",
        )

        # Assert - all metadata preserved
        original_msg = result.message.message
        assert original_msg.source == SourceType.REDDIT
        assert original_msg.source_id == "reddit_post_123"
        assert original_msg.author == "DDKing"
        assert original_msg.subreddit == "wallstreetbets"
        assert original_msg.upvotes == 15234
        assert original_msg.comment_count == 892
        assert original_msg.url == "https://reddit.com/r/wallstreetbets/comments/123"
        assert "GME" in original_msg.extracted_tickers

        # Sentiment preserved
        assert result.message.sentiment.label == SentimentLabel.BULLISH
        assert result.message.sentiment.score == 0.95
        assert result.message.sentiment.confidence == 0.92

    def test_pipeline_end_to_end_realistic_scenario(self, mock_alpaca_client):
        """Test realistic end-to-end scenario: Twitter alert → validation → trade decision.

        Scenario:
        - Popular trader tweets about $NVDA breakout
        - Phase 2 analyzed as highly bullish
        - Phase 3 validates with healthy uptrend
        - Options flow confirms with volume spike
        - Expected: PASS with confidence boost
        """
        # Arrange - Phase 2 output (AnalyzedMessage)
        twitter_alert = AnalyzedMessage(
            message=SocialMessage(
                source=SourceType.TWITTER,
                source_id="tweet_nvidia_breakout",
                author="ChipGuru",
                content="🚀 $NVDA breaking out of 6-month consolidation! AI demand strong. Resistance at $500 broken. Next stop $550! #NVDA #AI",
                timestamp=datetime(2024, 1, 15, 13, 45, 0, tzinfo=timezone.utc),
                url="https://twitter.com/ChipGuru/status/nvidia_breakout",
                like_count=1247,
                retweet_count=334,
                extracted_tickers=["NVDA"],
            ),
            sentiment=SentimentResult(
                label=SentimentLabel.BULLISH,
                score=0.94,
                confidence=0.91,
            ),
        )

        # Mock market data - healthy uptrend (not overbought)
        # Pattern: 2 up, 1 down with balanced moves for healthy RSI
        close_prices = []
        base = 450.0
        for i in range(50):
            if i % 3 == 2:  # Pullback every 3rd bar
                base -= 3.5  # Significant pullback
            else:
                base += 3.0  # Moderate gains
            close_prices.append(base)

        ohlc_data = pd.DataFrame({
            'open': [p - 2.0 for p in close_prices],
            'high': [p + 3.0 for p in close_prices],
            'low': [p - 2.5 for p in close_prices],
            'close': close_prices,
            'volume': [5000000 + (i * 100000) for i in range(50)],
        })
        mock_alpaca_client.get_bars.return_value = ohlc_data

        # Options data shows confirmation
        options_data = OptionsFlowData(
            volume_ratio=4.2,  # Strong volume spike
            iv_rank=55.0,      # Moderate IV (reasonable option prices)
            put_call_ratio=0.45,  # Heavy call buying
            unusual_activity=True,
        )

        validator = TechnicalValidator(
            alpaca_client=mock_alpaca_client,
            veto_mode=True,
            options_volume_spike_ratio=2.0,
            iv_rank_warning_threshold=80.0,
        )

        # Act - Phase 3 validation
        result = validator.validate(
            analyzed_message=twitter_alert,
            symbol="NVDA",
            options_data=options_data,
        )

        # Assert - Everything aligns for a strong trade signal
        assert result.validation.status == ValidationStatus.PASS
        assert result.should_trade() is True

        # Technical indicators should be favorable
        indicators = result.validation.indicators
        assert 40.0 <= indicators.rsi <= 70.0  # Not overbought
        assert indicators.adx >= 20.0  # Strong trend

        # Options flow should boost confidence
        assert result.validation.confidence_modifier > 1.0
        assert result.validation.options_flow.unusual_activity is True

        # No warnings or veto reasons
        assert len(result.validation.veto_reasons) == 0
        assert len(result.validation.warnings) == 0

        # Original high-confidence bullish signal preserved
        assert result.message.sentiment.confidence > 0.90
        assert result.message.message.like_count == 1247  # High social engagement

        # Final decision: This is a high-quality trade signal
        # - Strong bullish sentiment (0.91 confidence)
        # - Healthy technical setup (RSI not overbought, strong trend)
        # - Options flow confirmation (4.2x volume, unusual activity)
        # - High social engagement (1247 likes, 334 retweets)
