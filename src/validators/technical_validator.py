# src/validators/technical_validator.py
"""TechnicalValidator orchestrator for trading signal validation."""

import logging
from typing import Optional
import pandas as pd

from src.analyzers.analyzed_message import AnalyzedMessage
from src.validators.indicator_engine import IndicatorEngine
from src.validators.veto_logic import VetoLogic
from src.validators.options_validator import OptionsValidator
from src.validators.models import (
    TechnicalIndicators,
    TechnicalValidation,
    ValidationStatus,
    ValidatedSignal,
    OptionsFlowData,
)

logger = logging.getLogger(__name__)


class TechnicalValidator:
    """Orchestrates technical validation of trading signals.

    This class coordinates the validation pipeline:
    1. Fetch market data via Alpaca API
    2. Calculate technical indicators
    3. Apply veto logic based on sentiment
    4. Process options data if provided
    5. Return validated signal

    Attributes:
        alpaca_client: Client for fetching market data (duck-typed).
        veto_mode: If True, apply veto logic; if False, convert VETOs to WARNs.
        indicator_engine: Engine for calculating technical indicators.
        veto_logic: Logic for evaluating veto conditions.
        options_validator: Validator for options flow data.
        lookback_bars: Number of historical bars to fetch.
        timeframe: Timeframe for bar data (e.g., "5Min", "1Hour").
    """

    def __init__(
        self,
        alpaca_client,
        veto_mode: bool = True,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        adx_trend_threshold: float = 20.0,
        lookback_bars: int = 50,
        timeframe: str = "5Min",
        options_volume_spike_ratio: float = 2.0,
        iv_rank_warning_threshold: float = 80.0,
    ):
        """Initialize the TechnicalValidator.

        Args:
            alpaca_client: Client for fetching market data (duck-typed).
            veto_mode: If True, apply veto logic; if False, convert VETOs to WARNs.
            rsi_overbought: RSI threshold for overbought conditions.
            rsi_oversold: RSI threshold for oversold conditions.
            adx_trend_threshold: ADX threshold for trend strength.
            lookback_bars: Number of historical bars to fetch.
            timeframe: Timeframe for bar data.
            options_volume_spike_ratio: Volume spike threshold for options.
            iv_rank_warning_threshold: IV rank threshold for warnings.
        """
        self.alpaca_client = alpaca_client
        self.veto_mode = veto_mode
        self.lookback_bars = lookback_bars
        self.timeframe = timeframe

        # Initialize components
        self.indicator_engine = IndicatorEngine()
        self.veto_logic = VetoLogic(
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            adx_trend_threshold=adx_trend_threshold,
        )
        self.options_validator = OptionsValidator(
            volume_spike_ratio=options_volume_spike_ratio,
            iv_rank_warning_threshold=iv_rank_warning_threshold,
        )

    def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch market data for the symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            DataFrame with OHLC data.

        Raises:
            Exception: If API call fails (propagated to caller).
        """
        return self.alpaca_client.get_bars(
            symbol=symbol,
            timeframe=self.timeframe,
            limit=self.lookback_bars,
        )

    def _get_fallback_indicators(self) -> TechnicalIndicators:
        """Return neutral technical indicators for error cases.

        Returns:
            TechnicalIndicators with neutral/default values.
        """
        return TechnicalIndicators(
            rsi=50.0,
            macd_histogram=0.0,
            macd_trend="flat",
            stochastic_k=50.0,
            stochastic_d=50.0,
            adx=0.0,
        )

    def validate(
        self,
        analyzed_message: AnalyzedMessage,
        symbol: str,
        options_data: Optional[OptionsFlowData] = None,
    ) -> ValidatedSignal:
        """Validate an analyzed message with technical indicators.

        Pipeline:
        1. Fetch market data via alpaca_client.get_bars()
        2. Calculate indicators via IndicatorEngine
        3. Apply veto logic via VetoLogic
        4. Process options data if provided
        5. If not veto_mode, convert VETO to WARN
        6. Return ValidatedSignal

        Args:
            analyzed_message: Analyzed message from Phase 2.
            symbol: Stock ticker symbol.
            options_data: Optional options flow data.

        Returns:
            ValidatedSignal with validation results.

        Note:
            On error, returns PASS status with warning to fail safely.
        """
        try:
            # Step 1: Fetch market data
            ohlc_data = self._fetch_market_data(symbol)

            # Step 2: Calculate indicators
            indicators = self.indicator_engine.calculate_all(ohlc_data)

        except Exception as e:
            # Handle errors gracefully - return PASS with warning
            logger.warning(
                f"Failed to fetch/calculate indicators for {symbol}: {e}"
            )
            return ValidatedSignal(
                message=analyzed_message,
                validation=TechnicalValidation(
                    status=ValidationStatus.PASS,
                    indicators=self._get_fallback_indicators(),
                    warnings=[
                        f"Technical validation failed: {str(e)}. "
                        "Signal passed without technical confirmation."
                    ],
                ),
            )

        # Step 3: Apply veto logic
        sentiment = analyzed_message.sentiment.label
        status, veto_reasons = self.veto_logic.evaluate(indicators, sentiment)

        # Initialize validation result
        warnings: list[str] = []
        confidence_modifier = 1.0

        # Step 4: Process options data if provided
        if options_data is not None:
            is_enhanced, options_warnings = self.options_validator.validate(
                options_data
            )
            warnings.extend(options_warnings)
            confidence_modifier = self.options_validator.get_confidence_modifier(
                options_data
            )

        # Step 5: Convert VETO to WARN if not in veto mode
        if not self.veto_mode and status == ValidationStatus.VETO:
            status = ValidationStatus.WARN
            warnings.extend(veto_reasons)
            veto_reasons = []

        # Step 6: Build and return ValidatedSignal
        validation = TechnicalValidation(
            status=status,
            indicators=indicators,
            options_flow=options_data,
            veto_reasons=veto_reasons,
            warnings=warnings,
            confidence_modifier=confidence_modifier,
        )

        return ValidatedSignal(
            message=analyzed_message,
            validation=validation,
        )
