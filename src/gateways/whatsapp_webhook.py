import os
import asyncio
from typing import Dict, List
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from src.core.config import VERIFY_TOKEN

app = FastAPI()

# -------------------------------------------------------------
# 3-SECOND ASYNC DEBOUNCE BUFFER
# -------------------------------------------------------------
DEBOUNCE_SECONDS = 3.0
debounce_buffers: Dict[str, List[str]] = {}
debounce_tasks: Dict[str, asyncio.Task] = {}

async def flush_debounce_buffer(sender_id: str):
    """
    Waits 3.0 seconds. If no new message arrives from this sender,
    pops and clears the buffer and triggers one single pipeline execution.
    """
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return
        
    lines = debounce_buffers.pop(sender_id, [])
    debounce_tasks.pop(sender_id, None)
    
    if not lines:
        return
        
    combined_burst = "\n".join(lines)
    print(f"[Debounce Buffer] ⚡ Flushed {len(lines)} message(s) for {sender_id}:\n{combined_burst}")
    
    from src.core.orchestrator import handle_incoming_whatsapp_burst
    # Run in a thread worker so the asyncio event loop remains responsive
    await asyncio.to_thread(handle_incoming_whatsapp_burst, sender_id, combined_burst, lines)

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
    Parses incoming JSON payloads from WhatsApp with 3-second debouncing.
    """
    body = await request.json()
    
    # Safely inspect the payload for incoming messages
    try:
        # WhatsApp payloads can be deeply nested
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Skip read/delivery status updates (they contain "statuses" instead of "messages")
        if "messages" in value:
            message = value["messages"][0]
            sender_id = message.get("from")
            
            if message.get("type") == "text":
                text = message.get("text", {}).get("body")
                if text:
                    print(f"[Webhook Received] From {sender_id}: {text}")
                    
                    # 1. Cancel existing debounce countdown for this sender if running
                    existing_task = debounce_tasks.get(sender_id)
                    if existing_task and not existing_task.done():
                        existing_task.cancel()
                        
                    # 2. Append to sender's burst buffer
                    if sender_id not in debounce_buffers:
                        debounce_buffers[sender_id] = []
                    debounce_buffers[sender_id].append(text)
                    
                    # 3. Schedule fresh 3.0-second countdown
                    debounce_tasks[sender_id] = asyncio.create_task(flush_debounce_buffer(sender_id))
                
    except IndexError:
        pass
    except Exception as e:
        print(f"Error parsing webhook: {e}")

    # Return 200 OK to acknowledge receipt
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # For testing locally
    print("Starting FastAPI Webhook Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
