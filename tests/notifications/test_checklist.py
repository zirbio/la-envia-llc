# tests/notifications/test_checklist.py
"""Tests for ChecklistHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChecklistHandlerInit:
    """Tests for ChecklistHandler initialization."""

    def test_init(self):
        """Handler initializes with settings."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()
        mock_bot = MagicMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        assert handler._settings == settings
        assert handler._bot == mock_bot
        assert handler._chat_id == "12345"
        assert handler._checked_items == set()
        assert handler._message_id is None


class TestChecklistHandlerState:
    """Tests for checklist state management."""

    def test_is_checklist_complete_false_initially(self):
        """is_checklist_complete returns False initially."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )

        assert handler.is_checklist_complete() is False

    def test_is_checklist_complete_true_when_all_checked(self):
        """is_checklist_complete returns True when all items checked."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )
        handler._checked_items = {0, 1, 2}

        assert handler.is_checklist_complete() is True

    def test_reset_checklist_clears_state(self):
        """reset_checklist clears checked items and message ID."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()
        handler = ChecklistHandler(
            settings=settings,
            bot=MagicMock(),
            chat_id="12345",
        )
        handler._checked_items = {0, 1, 2}
        handler._message_id = 123

        handler.reset_checklist()

        assert handler._checked_items == set()
        assert handler._message_id is None


class TestChecklistHandlerSend:
    """Tests for sending checklist."""

    @pytest.mark.asyncio
    async def test_send_checklist_sends_message_with_keyboard(self):
        """send_checklist sends message with inline keyboard."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_message = MagicMock()
        mock_message.message_id = 999
        mock_bot.send_message = AsyncMock(return_value=mock_message)

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        await handler.send_checklist()

        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        assert call_args.kwargs["chat_id"] == "12345"
        assert "PRE-MARKET CHECKLIST" in call_args.kwargs["text"]
        assert call_args.kwargs["reply_markup"] is not None
        assert handler._message_id == 999


class TestChecklistHandlerCheck:
    """Tests for checking items."""

    @pytest.mark.asyncio
    async def test_on_item_checked_updates_state(self):
        """on_item_checked adds item to checked set."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )
        handler._message_id = 999

        await handler.on_item_checked(0)

        assert 0 in handler._checked_items

    @pytest.mark.asyncio
    async def test_on_item_checked_updates_message(self):
        """on_item_checked updates the message."""
        from notifications.checklist_handler import ChecklistHandler
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2"]
        )
        mock_bot = MagicMock()
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )
        handler._message_id = 999

        await handler.on_item_checked(0)

        mock_bot.edit_message_text.assert_called_once()
