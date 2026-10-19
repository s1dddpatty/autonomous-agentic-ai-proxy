import os
import asyncio
import datetime
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, CommandHandler

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_TG_USER_ID")

if not TOKEN or not AUTHORIZED_USER_ID:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or AUTHORIZED_TG_USER_ID in .env")

# Convert to integer if necessary
AUTHORIZED_USER_ID = int(AUTHORIZED_USER_ID)

async def send_mock_approval_card(app: Application):
    """Sends the mock approval card to the authorized user immediately upon startup."""
    text = (
        "*[Sister Incoming Message]*\n\n"
        "Message: \"milo just knocked over another plant 😭\"\n"
        "Draft Reply: \"same old, milo's still making messes everywhere 🤣\""
    )
    
    keyboard = [
        [
            InlineKeyboardButton("Approve", callback_data="approve"),
            InlineKeyboardButton("Suggest Changes", callback_data="edit"),
            InlineKeyboardButton("Discard", callback_data="discard")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    print("Sending mock approval card to user...")
    await app.bot.send_message(
        chat_id=AUTHORIZED_USER_ID,
        text=text,
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
    print("Message sent! Waiting for your interaction...")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query
    
    # Security: Ignore unauthorized users
    if update.effective_user.id != AUTHORIZED_USER_ID:
        print(f"Unauthorized interaction attempt from user ID: {update.effective_user.id}")
        await query.answer("You are not authorized to perform this action.", show_alert=True)
        return

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    
    action = query.data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] User clicked button: {action}")
    
    response_text = ""
    if action == "approve":
        response_text = "✅ Action recorded: Approved. The message would now be sent via WhatsApp."
    elif action == "edit":
        response_text = "✏️ Action recorded: Suggest Changes. The bot would now prompt you for feedback."
    elif action == "discard":
        response_text = "🗑️ Action recorded: Discarded. The draft has been dropped."

    await query.edit_message_text(text=response_text)
    
    # Gracefully stop the application after an action is taken for testing purposes
    print("Interaction complete. Stopping application.")
    asyncio.create_task(context.application.stop())


if __name__ == "__main__":
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # We use post_init to send the message right after the application starts
    async def post_init(app: Application):
        await send_mock_approval_card(app)
        
    application.post_init = post_init
    
    print("Starting Telegram Gateway Sandbox...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
