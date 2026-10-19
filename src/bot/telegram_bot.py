"""
src/bot/telegram_bot.py - Re-export and mirror of telegram bot module for backwards and cross-module compatibility.
"""
from src.gateways.telegram_bot import *
from src.gateways.telegram_bot import (
    main,
    format_fallback_card,
    get_fallback_keyboard,
    create_card,
    button_callback,
    simulate_command,
    handle_message,
    toggle_mode_command,
)

if __name__ == "__main__":
    main()
