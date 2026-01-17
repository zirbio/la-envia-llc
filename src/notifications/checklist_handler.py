# src/notifications/checklist_handler.py
"""Pre-market checklist handler with interactive buttons."""

import logging

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from .settings import NotificationSettings

logger = logging.getLogger(__name__)


class ChecklistHandler:
    """Handles interactive pre-market checklist.

    Sends a checklist message with inline keyboard buttons.
    Users can tap buttons to mark items as checked.
    """

    def __init__(
        self,
        settings: NotificationSettings,
        bot: Bot,
        chat_id: str,
    ):
        """Initialize the checklist handler.

        Args:
            settings: Notification settings with checklist items.
            bot: Telegram bot instance.
            chat_id: Chat ID to send checklist to.
        """
        self._settings = settings
        self._bot = bot
        self._chat_id = chat_id
        self._checked_items: set[int] = set()
        self._message_id: int | None = None

    async def send_checklist(self) -> None:
        """Send the pre-market checklist with inline buttons."""
        text = self._format_checklist_message()
        keyboard = self._build_keyboard()

        message = await self._bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )
        self._message_id = message.message_id
        logger.info("Pre-market checklist sent")

    async def on_item_checked(self, item_index: int) -> None:
        """Handle when user checks an item.

        Args:
            item_index: Index of the item that was checked.
        """
        self._checked_items.add(item_index)

        if self._message_id is None:
            return

        text = self._format_checklist_message()
        keyboard = self._build_keyboard()

        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            logger.info(f"Checklist item {item_index} checked")
        except Exception as e:
            logger.error(f"Failed to update checklist: {e}")

    def is_checklist_complete(self) -> bool:
        """Check if all items have been checked.

        Returns:
            True if all items are checked.
        """
        return len(self._checked_items) >= len(self._settings.checklist_items)

    def reset_checklist(self) -> None:
        """Reset checklist for next day."""
        self._checked_items = set()
        self._message_id = None
        logger.info("Checklist reset")

    def _build_keyboard(self) -> InlineKeyboardMarkup:
        """Build inline keyboard with current checked states.

        Returns:
            InlineKeyboardMarkup with buttons for each item.
        """
        buttons = []
        for i, item in enumerate(self._settings.checklist_items):
            if i in self._checked_items:
                # Item is checked - show checkmark
                text = f"✅ {i + 1}"
            else:
                # Item not checked - show number
                text = f"☐ {i + 1}"

            buttons.append(
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"checklist_{i}",
                )
            )

        # Arrange buttons in rows of 3
        rows = []
        for i in range(0, len(buttons), 3):
            rows.append(buttons[i : i + 3])

        return InlineKeyboardMarkup(rows)

    def _format_checklist_message(self) -> str:
        """Format the checklist message showing checked/unchecked items.

        Returns:
            Formatted message string.
        """
        if self.is_checklist_complete():
            return """✅ <b>CHECKLIST COMPLETE</b>

All items verified. Ready to trade!"""

        lines = ["📋 <b>PRE-MARKET CHECKLIST</b>", ""]
        lines.append("Ready for trading day? Complete all items:")
        lines.append("")

        for i, item in enumerate(self._settings.checklist_items):
            if i in self._checked_items:
                lines.append(f"✅ {item}")
            else:
                lines.append(f"☐ {item}")

        return "\n".join(lines)
