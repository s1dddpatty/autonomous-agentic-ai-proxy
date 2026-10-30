import os
import asyncio
from typing import Dict, List
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from src.core.config import VERIFY_TOKEN
from src.core.memory_manager import add_pending_message, pop_pending_messages

app = FastAPI()

# -------------------------------------------------------------
# SLIDING WINDOW ASYNC DEBOUNCE LOCK FOR MESSAGE BURSTS
# -------------------------------------------------------------
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "7.0"))
ACTIVE_TIMERS: Dict[str, asyncio.Task] = {}

async def debounce_and_dispatch(chat_id: str):
    """
    Sliding window debounce timer.
    Waits for DEBOUNCE_SECONDS of silence after the most recent message.
    If a follow-up message arrives while sleeping, this task is cancelled and rescheduled.
    Once DEBOUNCE_SECONDS of silence elapses without new messages:
    1. Removes chat_id from ACTIVE_TIMERS.
    2. Pulls all pending messages for that chat_id from the SQLite buffer.
    3. Concatenates them into a single string (joined_text = "\n".join(messages)).
    4. Passes the unified burst block to handle_incoming_whatsapp_burst.
    """
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        # Cancelled because a follow-up message arrived before silence duration elapsed
        return
    finally:
        current_task = asyncio.current_task()
        # Only remove if this task is still the registered active timer (prevent cancelling task from wiping a newer task)
        if ACTIVE_TIMERS.get(chat_id) == current_task:
            ACTIVE_TIMERS.pop(chat_id, None)

    # Pull all pending messages for that chat_id from the SQLite buffer
    messages = pop_pending_messages(chat_id)
    if not messages:
        return

    # Concatenate them into a single string
    joined_text = "\n".join(messages)
    print(f"[Debounce Lock] ⚡ Sliding debounce elapsed ({DEBOUNCE_SECONDS}s silence). Flushed {len(messages)} message(s) for {chat_id}:\n{joined_text}")

    from src.core.orchestrator import handle_incoming_whatsapp_burst
    # Pass this single joined_text block to Gemini so it processes the entire burst as one prompt
    await asyncio.to_thread(handle_incoming_whatsapp_burst, chat_id, joined_text, messages)

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """
    Handles Meta's initial verify-token handshake.
    """
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Parses incoming JSON payloads from WhatsApp with 10-second debounce lock.
    """
    body = await request.json()
    
    # Safely inspect the payload for incoming messages
    try:
        entries = body.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                
                # Skip read/delivery status updates (they contain "statuses" instead of "messages")
                if "messages" in value:
                    for message in value.get("messages", []):
                        sender_id = message.get("from")
                        
                        if message.get("type") == "text":
                            text = message.get("text", {}).get("body")
                            if text and sender_id:
                                chat_id = str(sender_id).strip()
                                print(f"[Webhook Received] Dynamic routing from {chat_id}: {text}")
                                
                                # 1. Instantly save to SQLite pending buffer for this sender
                                add_pending_message(chat_id, text)
                                
                                # 2. Sliding window debounce: cancel active timer so timer resets from latest message
                                existing_timer = ACTIVE_TIMERS.get(chat_id)
                                if existing_timer and not existing_timer.done():
                                    existing_timer.cancel()
                                    print(f"[Debounce Lock] 🔄 Follow-up received for {chat_id}. Sliding timer reset ({DEBOUNCE_SECONDS}s).")
                                else:
                                    print(f"[Debounce Lock] ⏱️ New burst started for {chat_id}. Timer set ({DEBOUNCE_SECONDS}s).")

                                # 3. Schedule fresh sliding countdown task for this sender
                                ACTIVE_TIMERS[chat_id] = asyncio.create_task(debounce_and_dispatch(chat_id))
                            
    except Exception as e:
        print(f"Error parsing webhook: {e}")

    # Return 200 OK to acknowledge receipt
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # For testing locally
    print("Starting FastAPI Webhook Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
