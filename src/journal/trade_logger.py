# src/journal/trade_logger.py
"""Trade logger for persisting trades to JSON files."""
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import aiofiles

from src.execution.models import TrackedPosition
from src.journal.models import JournalEntry
from src.journal.settings import JournalSettings
from src.scoring.models import Direction, ScoreComponents, TradeRecommendation


class TradeLogger:
    """Logger for persisting trades to JSON files.

    Stores trades in daily JSON files with format: {data_dir}/{YYYY-MM-DD}.json
    Trade IDs follow format: YYYY-MM-DD-SYMBOL-NNN
    """

    def __init__(self, settings: JournalSettings) -> None:
        """Initialize the trade logger.

        Args:
            settings: Journal configuration settings.
        """
        self._settings = settings
        self._data_dir = Path(settings.data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, trade_date: date) -> Path:
        """Get the JSON file path for a specific date."""
        return self._data_dir / f"{trade_date.isoformat()}.json"

    async def _read_entries(self, trade_date: date) -> list[dict]:
        """Read entries from the JSON file for a date."""
        file_path = self._get_file_path(trade_date)
        if not file_path.exists():
            return []

        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def _write_entries(self, trade_date: date, entries: list[dict]) -> None:
        """Write entries to the JSON file for a date."""
        file_path = self._get_file_path(trade_date)
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(entries, indent=2, default=str))

    def _generate_trade_id(
        self, trade_date: date, symbol: str, existing_count: int
    ) -> str:
        """Generate a trade ID in format YYYY-MM-DD-SYMBOL-NNN."""
        sequence = existing_count + 1
        return f"{trade_date.isoformat()}-{symbol}-{sequence:03d}"

    def _entry_to_dict(self, entry: JournalEntry) -> dict:
        """Convert a JournalEntry to a dictionary for JSON storage."""
        return {
            "trade_id": entry.trade_id,
            "symbol": entry.symbol,
            "direction": entry.direction.value,
            "entry_time": entry.entry_time.isoformat(),
            "entry_price": entry.entry_price,
            "entry_quantity": entry.entry_quantity,
            "entry_reason": entry.entry_reason,
            "entry_score": entry.entry_score,
            "stop_loss": entry.stop_loss,
            "exit_time": entry.exit_time.isoformat() if entry.exit_time else None,
            "exit_price": entry.exit_price,
            "exit_quantity": entry.exit_quantity,
            "exit_reason": entry.exit_reason,
            "pnl_dollars": entry.pnl_dollars,
            "pnl_percent": entry.pnl_percent,
            "r_multiple": entry.r_multiple,
            "market_conditions": entry.market_conditions,
            "score_components": {
                "sentiment_score": entry.score_components.sentiment_score,
                "technical_score": entry.score_components.technical_score,
                "sentiment_weight": entry.score_components.sentiment_weight,
                "technical_weight": entry.score_components.technical_weight,
                "confluence_bonus": entry.score_components.confluence_bonus,
                "credibility_multiplier": entry.score_components.credibility_multiplier,
                "time_factor": entry.score_components.time_factor,
            },
            "emotion_tag": entry.emotion_tag,
            "notes": entry.notes,
        }

    def _dict_to_entry(self, data: dict) -> JournalEntry:
        """Convert a dictionary from JSON to a JournalEntry."""
        return JournalEntry(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            direction=Direction(data["direction"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            entry_quantity=data["entry_quantity"],
            entry_reason=data["entry_reason"],
            entry_score=data["entry_score"],
            stop_loss=data["stop_loss"],
            exit_time=(
                datetime.fromisoformat(data["exit_time"])
                if data["exit_time"]
                else None
            ),
            exit_price=data["exit_price"],
            exit_quantity=data["exit_quantity"],
            exit_reason=data["exit_reason"],
            pnl_dollars=data["pnl_dollars"],
            pnl_percent=data["pnl_percent"],
            r_multiple=data["r_multiple"],
            market_conditions=data["market_conditions"],
            score_components=ScoreComponents(
                sentiment_score=data["score_components"]["sentiment_score"],
                technical_score=data["score_components"]["technical_score"],
                sentiment_weight=data["score_components"]["sentiment_weight"],
                technical_weight=data["score_components"]["technical_weight"],
                confluence_bonus=data["score_components"]["confluence_bonus"],
                credibility_multiplier=data["score_components"]["credibility_multiplier"],
                time_factor=data["score_components"]["time_factor"],
            ),
            emotion_tag=data["emotion_tag"],
            notes=data["notes"],
        )

    async def log_entry(
        self, position: TrackedPosition, recommendation: TradeRecommendation
    ) -> str:
        """Log a trade entry to the journal.

        Args:
            position: The tracked position that was opened.
            recommendation: The trade recommendation that triggered the entry.

        Returns:
            The generated trade ID.
        """
        trade_date = position.entry_time.date()
        existing_entries = await self._read_entries(trade_date)

        trade_id = self._generate_trade_id(
            trade_date, position.symbol, len(existing_entries)
        )

        entry = JournalEntry(
            trade_id=trade_id,
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            entry_quantity=position.quantity,
            entry_reason=recommendation.reasoning,
            entry_score=recommendation.score,
            stop_loss=position.stop_loss,
            exit_time=None,
            exit_price=None,
            exit_quantity=0,
            exit_reason=None,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            r_multiple=0.0,
            market_conditions="",
            score_components=recommendation.components,
            emotion_tag=None,
            notes=None,
        )

        existing_entries.append(self._entry_to_dict(entry))
        await self._write_entries(trade_date, existing_entries)

        return trade_id

    async def log_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str,
    ) -> None:
        """Log a trade exit and calculate PnL.

        Args:
            trade_id: The trade ID to update.
            exit_time: When the position was closed.
            exit_price: The exit price.
            exit_quantity: Number of shares exited.
            exit_reason: Reason for exiting.
        """
        trade_date = date.fromisoformat(trade_id[:10])
        entries = await self._read_entries(trade_date)

        for entry in entries:
            if entry["trade_id"] == trade_id:
                entry["exit_time"] = exit_time.isoformat()
                entry["exit_price"] = exit_price
                entry["exit_quantity"] = exit_quantity
                entry["exit_reason"] = exit_reason

                entry_price = entry["entry_price"]
                quantity = entry["entry_quantity"]
                stop_loss = entry["stop_loss"]
                direction = Direction(entry["direction"])

                if direction == Direction.LONG:
                    pnl_dollars = (exit_price - entry_price) * quantity
                    risk_per_share = entry_price - stop_loss
                    gain_per_share = exit_price - entry_price
                else:
                    pnl_dollars = (entry_price - exit_price) * quantity
                    risk_per_share = stop_loss - entry_price
                    gain_per_share = entry_price - exit_price

                cost_basis = entry_price * quantity
                pnl_percent = (pnl_dollars / cost_basis) * 100 if cost_basis else 0.0

                r_multiple = (
                    gain_per_share / risk_per_share if risk_per_share else 0.0
                )

                entry["pnl_dollars"] = pnl_dollars
                entry["pnl_percent"] = round(pnl_percent, 3)
                entry["r_multiple"] = round(r_multiple, 3)
                break

        await self._write_entries(trade_date, entries)

    async def add_emotion_tag(self, trade_id: str, tag: str) -> None:
        """Add an emotion tag to a trade entry.

        Args:
            trade_id: The trade ID to update.
            tag: The emotion tag to add.
        """
        trade_date = date.fromisoformat(trade_id[:10])
        entries = await self._read_entries(trade_date)

        for entry in entries:
            if entry["trade_id"] == trade_id:
                entry["emotion_tag"] = tag
                break

        await self._write_entries(trade_date, entries)

    async def add_notes(self, trade_id: str, notes: str) -> None:
        """Add notes to a trade entry.

        Args:
            trade_id: The trade ID to update.
            notes: The notes to add.
        """
        trade_date = date.fromisoformat(trade_id[:10])
        entries = await self._read_entries(trade_date)

        for entry in entries:
            if entry["trade_id"] == trade_id:
                entry["notes"] = notes
                break

        await self._write_entries(trade_date, entries)

    async def get_entries_for_date(self, query_date: date) -> list[JournalEntry]:
        """Get all journal entries for a specific date.

        Args:
            query_date: The date to retrieve entries for.

        Returns:
            List of JournalEntry objects for that date.
        """
        entries = await self._read_entries(query_date)
        return [self._dict_to_entry(e) for e in entries]

    async def get_entries_for_period(
        self, start_date: date, end_date: date
    ) -> list[JournalEntry]:
        """Get all journal entries within a date range.

        Args:
            start_date: Start of the period (inclusive).
            end_date: End of the period (inclusive).

        Returns:
            List of JournalEntry objects within the period.
        """
        all_entries: list[JournalEntry] = []
        current_date = start_date

        while current_date <= end_date:
            entries = await self.get_entries_for_date(current_date)
            all_entries.extend(entries)
            current_date += timedelta(days=1)

        return all_entries
