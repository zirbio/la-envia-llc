# tests/scoring/test_time_factors.py
"""Tests for time factors calculator."""
from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from src.scoring.time_factors import TimeFactorCalculator


class TestTimeFactorCalculator:
    """Tests for TimeFactorCalculator."""

    def test_market_hours_returns_full_factor(self):
        """Test that 10:00 AM ET during market hours returns 1.0 factor."""
        calculator = TimeFactorCalculator()
        # 10:00 AM ET on a weekday (Monday)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)  # Monday

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 1.0
        assert "market_hours" in reasons

    def test_premarket_returns_reduced_factor(self):
        """Test that 6:00 AM ET during pre-market returns 0.9 factor."""
        calculator = TimeFactorCalculator()
        # 6:00 AM ET on a weekday (Tuesday)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 16, 6, 0, 0, tzinfo=et_tz)  # Tuesday

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 0.9
        assert "premarket" in reasons

    def test_afterhours_returns_reduced_factor(self):
        """Test that 5:00 PM ET during after-hours returns 0.8 factor."""
        calculator = TimeFactorCalculator()
        # 5:00 PM ET on a weekday (Wednesday)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 17, 17, 0, 0, tzinfo=et_tz)  # Wednesday

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 0.8
        assert "afterhours" in reasons

    def test_closed_hours_returns_lowest_factor(self):
        """Test that 10:00 PM ET during closed hours returns 0.5 factor."""
        calculator = TimeFactorCalculator()
        # 10:00 PM ET on a weekday (Thursday)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 18, 22, 0, 0, tzinfo=et_tz)  # Thursday

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 0.5
        assert "closed" in reasons

    def test_earnings_proximity_applies_penalty(self):
        """Test that earnings within proximity days applies earnings factor."""
        calculator = TimeFactorCalculator(earnings_proximity_days=3, earnings_factor=0.7)
        et_tz = ZoneInfo("America/New_York")
        # Signal at 10:00 AM ET on Monday
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)
        # Earnings is 2 days away (within 3 day proximity)
        earnings_dates = {
            "AAPL": datetime(2024, 1, 17, tzinfo=et_tz),  # 2 days away
        }

        factor, reasons = calculator.calculate_factor(
            timestamp, "AAPL", earnings_dates=earnings_dates
        )

        # Market hours (1.0) * earnings factor (0.7) = 0.7
        assert factor == pytest.approx(0.7, rel=1e-3)
        assert "market_hours" in reasons
        assert "earnings_proximity" in reasons

    def test_weekend_returns_lowest_factor(self):
        """Test that Saturday/Sunday returns 0.5 factor (closed)."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")
        # Saturday at noon
        saturday = datetime(2024, 1, 20, 12, 0, 0, tzinfo=et_tz)  # Saturday
        # Sunday at noon
        sunday = datetime(2024, 1, 21, 12, 0, 0, tzinfo=et_tz)  # Sunday

        sat_factor, sat_reasons = calculator.calculate_factor(saturday, "AAPL")
        sun_factor, sun_reasons = calculator.calculate_factor(sunday, "AAPL")

        assert sat_factor == 0.5
        assert "weekend" in sat_reasons
        assert sun_factor == 0.5
        assert "weekend" in sun_reasons


class TestIsMarketHours:
    """Tests for is_market_hours method."""

    def test_is_market_hours_true_during_regular_hours(self):
        """Test is_market_hours returns True during 9:30 AM - 4:00 PM ET."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")

        # 9:30 AM ET (market open)
        market_open = datetime(2024, 1, 15, 9, 30, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(market_open) is True

        # 12:00 PM ET (mid-day)
        midday = datetime(2024, 1, 15, 12, 0, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(midday) is True

        # 3:59 PM ET (just before close)
        before_close = datetime(2024, 1, 15, 15, 59, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(before_close) is True

    def test_is_market_hours_false_outside_regular_hours(self):
        """Test is_market_hours returns False outside 9:30 AM - 4:00 PM ET."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")

        # 9:29 AM ET (just before open)
        before_open = datetime(2024, 1, 15, 9, 29, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(before_open) is False

        # 4:00 PM ET (market close)
        market_close = datetime(2024, 1, 15, 16, 0, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(market_close) is False

        # 6:00 AM ET (pre-market)
        premarket = datetime(2024, 1, 15, 6, 0, 0, tzinfo=et_tz)
        assert calculator.is_market_hours(premarket) is False


class TestIsPremarket:
    """Tests for is_premarket method."""

    def test_is_premarket_true_during_premarket(self):
        """Test is_premarket returns True during 4:00 AM - 9:30 AM ET."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")

        # 4:00 AM ET (premarket open)
        premarket_open = datetime(2024, 1, 15, 4, 0, 0, tzinfo=et_tz)
        assert calculator.is_premarket(premarket_open) is True

        # 6:00 AM ET (mid-premarket)
        mid_premarket = datetime(2024, 1, 15, 6, 0, 0, tzinfo=et_tz)
        assert calculator.is_premarket(mid_premarket) is True

        # 9:29 AM ET (just before regular open)
        before_regular = datetime(2024, 1, 15, 9, 29, 0, tzinfo=et_tz)
        assert calculator.is_premarket(before_regular) is True

    def test_is_premarket_false_outside_premarket(self):
        """Test is_premarket returns False outside 4:00 AM - 9:30 AM ET."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")

        # 3:59 AM ET (before premarket)
        before_premarket = datetime(2024, 1, 15, 3, 59, 0, tzinfo=et_tz)
        assert calculator.is_premarket(before_premarket) is False

        # 9:30 AM ET (regular market open)
        regular_open = datetime(2024, 1, 15, 9, 30, 0, tzinfo=et_tz)
        assert calculator.is_premarket(regular_open) is False

        # 5:00 PM ET (after-hours)
        afterhours = datetime(2024, 1, 15, 17, 0, 0, tzinfo=et_tz)
        assert calculator.is_premarket(afterhours) is False


class TestTimezoneConversion:
    """Tests for timezone handling."""

    def test_utc_timestamp_converted_correctly(self):
        """Test that UTC timestamp is converted to ET correctly."""
        calculator = TimeFactorCalculator()
        utc_tz = ZoneInfo("UTC")
        # 3:00 PM UTC = 10:00 AM ET (during EST, -5 hours)
        # Note: In January, ET is EST (UTC-5)
        utc_timestamp = datetime(2024, 1, 15, 15, 0, 0, tzinfo=utc_tz)

        factor, reasons = calculator.calculate_factor(utc_timestamp, "AAPL")

        # 10:00 AM ET is during market hours
        assert factor == 1.0
        assert "market_hours" in reasons


class TestCustomFactors:
    """Tests for custom factor configuration."""

    def test_custom_premarket_factor(self):
        """Test that custom premarket factor is applied."""
        calculator = TimeFactorCalculator(premarket_factor=0.85)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 6, 0, 0, tzinfo=et_tz)

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 0.85
        assert "premarket" in reasons

    def test_custom_afterhours_factor(self):
        """Test that custom afterhours factor is applied."""
        calculator = TimeFactorCalculator(afterhours_factor=0.75)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 17, 0, 0, tzinfo=et_tz)

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 0.75
        assert "afterhours" in reasons


class TestEarningsProximity:
    """Tests for earnings proximity logic."""

    def test_earnings_outside_proximity_no_penalty(self):
        """Test that earnings outside proximity days has no penalty."""
        calculator = TimeFactorCalculator(earnings_proximity_days=3, earnings_factor=0.7)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)
        # Earnings is 5 days away (outside 3 day proximity)
        earnings_dates = {
            "AAPL": datetime(2024, 1, 20, tzinfo=et_tz),  # 5 days away
        }

        factor, reasons = calculator.calculate_factor(
            timestamp, "AAPL", earnings_dates=earnings_dates
        )

        # Market hours (1.0), no earnings penalty
        assert factor == 1.0
        assert "earnings_proximity" not in reasons

    def test_earnings_for_different_symbol_no_penalty(self):
        """Test that earnings for different symbol has no penalty."""
        calculator = TimeFactorCalculator(earnings_proximity_days=3, earnings_factor=0.7)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)
        # Earnings for MSFT, not AAPL
        earnings_dates = {
            "MSFT": datetime(2024, 1, 16, tzinfo=et_tz),  # 1 day away
        }

        factor, reasons = calculator.calculate_factor(
            timestamp, "AAPL", earnings_dates=earnings_dates
        )

        # Market hours (1.0), no earnings penalty for AAPL
        assert factor == 1.0
        assert "earnings_proximity" not in reasons

    def test_no_earnings_dates_provided(self):
        """Test that None earnings_dates works correctly."""
        calculator = TimeFactorCalculator()
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)

        factor, reasons = calculator.calculate_factor(timestamp, "AAPL")

        assert factor == 1.0
        assert "earnings_proximity" not in reasons

    def test_earnings_same_day_applies_penalty(self):
        """Test that earnings on same day applies penalty."""
        calculator = TimeFactorCalculator(earnings_proximity_days=3, earnings_factor=0.7)
        et_tz = ZoneInfo("America/New_York")
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=et_tz)
        # Earnings same day
        earnings_dates = {
            "AAPL": datetime(2024, 1, 15, tzinfo=et_tz),
        }

        factor, reasons = calculator.calculate_factor(
            timestamp, "AAPL", earnings_dates=earnings_dates
        )

        assert factor == pytest.approx(0.7, rel=1e-3)
        assert "earnings_proximity" in reasons
