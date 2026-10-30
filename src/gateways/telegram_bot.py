import os
import re
import uuid
import asyncio
import sys
from typing import Optional
from dotenv import load_dotenv

# Ensure the root directory is in the path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters, ContextTypes
from src.core.orchestrator import start_message_pipeline, resume_message_pipeline
from src.core.fallbacks import get_random_fallback, OFFLINE_STALL_DRAFTS
from src.core.database import get_config, set_config
from src.core.memory_manager import (
    get_pending_card,
    get_pending_card_by_thread,
    update_pending_card_draft,
    save_pending_card,
)

from src.core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TARGET_NAME, TARGET_PHONE

TARGET_NAME = os.getenv("TARGET_CONTACT_NAME", TARGET_NAME or "PersonName")

TOKEN = TELEGRAM_BOT_TOKEN
AUTHORIZED_USER_ID = int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else 0

if not TOKEN or not AUTHORIZED_USER_ID:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment / config.")

# In-memory store for awaiting feedback
# Key: user_id, Value: (thread_id, card_msg_id)
awaiting_feedback = {}

def get_approval_keyboard(thread_id: str):
    keyboard = [
        [
            InlineKeyboardButton("Approve & Send ✅", callback_data=f"approve:{thread_id}"),
            InlineKeyboardButton("Regenerate 🔄", callback_data=f"edit:{thread_id}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def format_fallback_card(incoming_text: str, draft_text: str, sender_id: str = "") -> str:
    """Formats the Telegram approval card when Gemini API is offline."""
    sender_header = f"📩 Incoming from: {sender_id}\n\n" if sender_id else ""
    return (
        "⚠️ [API OFFLINE] Google Servers Down.\n\n"
        f"{sender_header}"
        f"Text:\n\"{incoming_text}\"\n\n"
        f"🤖 Proposed Stall (Local): \"{draft_text}\"\n\n"
        "(A background worker is polling the API and will notify you when it recovers.)"
    )

def get_fallback_keyboard(target_id: str):
    """Inline keyboard for offline stall draft: Approve or Swap."""
    keyboard = [
        [
            InlineKeyboardButton("Send Stall Excuse ✅", callback_data=f"approve:{target_id}"),
            InlineKeyboardButton("🎲 Swap Excuse", callback_data=f"cycle_fallback:{target_id}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_card(incoming_text: str, draft_text: str, thread_id: str, sender_id: str = "", is_fallback: bool = False, server_recovered: bool = False, is_revision: bool = False):
    """
    Builds the card text and inline keyboard depending on whether is_fallback or normal/recovered.
    Explicitly displays the sender's identity in the header: 📩 Incoming from: [Sender Phone Number].
    """
    from src.core.config import get_contact_display_name
    sender_label = get_contact_display_name(sender_id or TARGET_PHONE or TARGET_NAME)
    if is_fallback:
        text = format_fallback_card(incoming_text, draft_text, sender_id=sender_label)
        keyboard = get_fallback_keyboard(thread_id)
    else:
        sender_header = f"📩 Incoming from: {sender_label}"
        if server_recovered:
            header = f"🟢 [SERVER RECOVERED]\n{sender_header}"
        elif is_revision:
            header = f"🔄 Revised AI Draft\n{sender_header}"
        else:
            header = sender_header
            
        text = (
            f"{header}\n\n"
            f"Text:\n\"{incoming_text}\"\n\n"
            "🤖 Proposed AI Draft:\n"
            f"\"{draft_text}\""
        )
        keyboard = get_approval_keyboard(thread_id)
    return text, keyboard

async def simulate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return
        
    thread_id = str(uuid.uuid4())[:8]
    sender_id = TARGET_PHONE or TARGET_NAME
    incoming_text = "how is your day going? milo just knocked over another plant 😭"
    
    await update.message.reply_text(f"Starting simulation. Thread ID: {thread_id}")
    
    # Run orchestrator in a separate thread to prevent blocking the asyncio event loop
    state = await asyncio.to_thread(start_message_pipeline, thread_id, sender_id, incoming_text)
    
    draft = state.get("draft_text", "")
    is_fallback = bool(state.get("is_fallback"))
    text, keyboard = create_card(incoming_text, draft, thread_id, sender_id=sender_id, is_fallback=is_fallback)
    
    await update.message.reply_text(text, reply_markup=keyboard)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await query.answer("Unauthorized.", show_alert=True)
        return

    await query.answer()
    
    data = query.data
    action, target_id = data.split(":")
    card_msg_id = query.message.message_id
    
    if action == "approve":
        await asyncio.to_thread(resume_message_pipeline, target_id, "approve", None, card_msg_id)
        await query.edit_message_text(text="✅ [Approved & Dispatched]")
    elif action == "discard":
        await asyncio.to_thread(resume_message_pipeline, target_id, "discard", None, card_msg_id)
        await query.edit_message_text(text="❌ Discarded.")
    elif action == "edit":
        awaiting_feedback[update.effective_user.id] = (target_id, card_msg_id)
        await query.edit_message_text(text=f"{query.message.text}\n\n*Awaiting Revision Notes...*")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply with your revision notes (e.g., 'Make it shorter and mention I\\'ll be late')."
        )
    elif action == "cycle_fallback":
        # Retrieve card to get current draft and incoming text
        card = get_pending_card(card_msg_id)
        if not card:
            card = get_pending_card_by_thread(target_id)
        if not card and str(target_id).isdigit():
            card = get_pending_card(int(target_id))

        current_draft = card.get("draft_text", "") if card else ""
        incoming_text = card.get("incoming_text", "") if card else ""
        sender_id = card.get("sender_id", "") if card else ""
        
        # Fallback if card wasn't found in DB: extract from current message text
        if query.message.text:
            if not incoming_text:
                m = re.search(r'Text:\s*\n?"([^"]+)"', query.message.text)
                if not m:
                    m = re.search(r'📩 Incoming(?:\s*from:[^\n]+)?:\s*"([^"]+)"', query.message.text)
                if m:
                    incoming_text = m.group(1)
            if not current_draft:
                m = re.search(r'🤖 Proposed Stall \(Local\):\s*"([^"]+)"', query.message.text)
                if m:
                    current_draft = m.group(1)
            if not sender_id:
                m_sender = re.search(r'📩 Incoming from:\s*([^\n]+)', query.message.text)
                if m_sender:
                    sender_id = m_sender.group(1).strip()

        target_lower = TARGET_NAME.lower()
        err_type = 429 if (f"sorryy {target_lower}" in current_draft or "sorryy aarushii" in current_draft or "rest now" in current_draft or "13 hours" in current_draft) else 503
        new_draft = get_random_fallback(error_type=err_type, exclude=current_draft)
        effective_thread = card.get("thread_id", target_id) if card else target_id
        
        if card:
            update_pending_card_draft(card["telegram_message_id"], new_draft)
        
        new_text = format_fallback_card(incoming_text, new_draft, sender_id=sender_id)
        fallback_kb = get_fallback_keyboard(effective_thread)
        
        try:
            await query.edit_message_text(text=new_text, reply_markup=fallback_kb)
        except Exception as e:
            if "message is not modified" not in str(e).lower():
                print(f"[Telegram Bot] Error editing fallback card: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != AUTHORIZED_USER_ID:
        return
        
    feedback = update.message.text
    if user_id in awaiting_feedback:
        item = awaiting_feedback.pop(user_id)
        if isinstance(item, tuple):
            thread_id, card_msg_id = item
        else:
            thread_id, card_msg_id = item, None
        
        await asyncio.to_thread(resume_message_pipeline, thread_id, "edit", feedback, card_msg_id)
    else:
        # Check if there is an active pending card (e.g. second revision note)
        from src.core.memory_manager import get_latest_pending_card
        latest = get_latest_pending_card()
        if latest and latest.get("status") == "pending":
            print(f"[Telegram Bot] Applying revision notes to active card #{latest['telegram_message_id']}: '{feedback}'")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"🔄 Applying revision notes: \"{feedback}\"..."
            )
            await asyncio.to_thread(resume_message_pipeline, latest["thread_id"], "edit", feedback, latest["telegram_message_id"])

async def toggle_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Toggles between 'training' mode (requires Telegram approval)
    and 'autopilot' mode (replies directly to WhatsApp).
    """
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return
        
    current_mode = get_config("mode")
    new_mode = "autopilot" if current_mode == "training" else "training"
    set_config("mode", new_mode)
    
    print(f"[Telegram Bot] 🔄 Persistent mode toggled: {current_mode} -> {new_mode}")
    await update.message.reply_text(
        f"🔄 Bot is now in *{new_mode.upper()}* mode.",
        parse_mode="Markdown"
    )

def main():
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("toggle_mode", toggle_mode_command))
    application.add_handler(CommandHandler("mode", toggle_mode_command))
    application.add_handler(CommandHandler("simulate", simulate_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Starting Telegram Bot with LangGraph Integration...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
