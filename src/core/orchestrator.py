import os
import sqlite3
import uuid
import time
import asyncio
import threading
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.memory_manager import (
    get_pending_session,
    get_pending_session_by_thread,
    set_pending_session,
    update_pending_telegram_msg_id,
    pop_superseded_tg_msg_id,
    delete_pending_session,
    save_pending_card,
    get_pending_card,
    get_pending_card_by_thread,
    update_pending_card_status,
    update_pending_card_draft,
    delete_pending_card,
    add_pending_message,
    pop_pending_messages,
)

try:
    from src.gateways.whatsapp_webhook import ACTIVE_TIMERS, debounce_and_dispatch
except ImportError:
    ACTIVE_TIMERS = {}
    debounce_and_dispatch = None

from src.core.fallbacks import get_random_fallback, OFFLINE_STALL_DRAFTS, QUOTA_EXHAUSTED_DRAFTS, is_stall_draft
from src.core.database import get_config
from src.core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TARGET_NAME, TARGET_PHONE, CHECKPOINT_DB_PATH, DATA_DIR

TARGET_NAME = os.getenv("TARGET_CONTACT_NAME", TARGET_NAME or "PersonName")
TARGET_PHONE = os.getenv("TARGET_PHONE_NUMBER", TARGET_PHONE or "")

# 1. Define AgentState Schema
class AgentState(TypedDict, total=False):
    message_id: str
    sender_id: str
    incoming_text: str
    messages: list
    draft_text: Optional[str]
    human_feedback: Optional[str]
    status: Optional[str]
    is_fallback: Optional[bool]
    server_recovered: Optional[bool]

# 2. Define Nodes
from src.core.gemini_node import (
    gemini_proxy_node,
    extract_clean_text,
    sanitize_draft,
    build_quota_exhausted_draft,
    generate_with_key_rotation,
    is_quota_fallback_text,
    MODEL_NAME,
)

async def background_recovery_worker(incoming_text: str, vibe_context=None, db_id: str = None, sender_id: str = None, messages: list = None):
    """
    Ghost worker loop that polls gemini-flash-latest every 60 seconds during an outage.
    Once the servers recover, dispatches a brand NEW Telegram card with the real draft.
    """
    if isinstance(vibe_context, dict):
        if not sender_id:
            sender_id = vibe_context.get("sender_id", TARGET_NAME)
        if not messages:
            messages = vibe_context.get("messages", [])
        if not db_id:
            db_id = vibe_context.get("message_id", str(uuid.uuid4())[:8])

    print(f"[Recovery Worker] 👻 Ghost worker started for incoming: '{incoming_text[:40]}...' (thread {db_id})")

    while True:
        await asyncio.sleep(300)
        print(f"[Recovery Worker] 🔄 Polling Gemini API for thread {db_id}...")

        try:
            state_input = {
                "incoming_text": incoming_text,
                "messages": messages or [],
                "sender_id": sender_id or TARGET_NAME,
                "message_id": db_id
            }
            res = await asyncio.to_thread(gemini_proxy_node, state_input)
            
            # Safe unpacking: handle res is None or draft is None/empty
            if res is None:
                print(f"[Recovery Worker] ❌ Generator returned None for thread {db_id}!")
                failed_thread_id = str(uuid.uuid4())[:8]
                failed_state = {
                    "message_id": failed_thread_id,
                    "sender_id": sender_id or TARGET_NAME,
                    "incoming_text": incoming_text,
                    "draft_text": "[GENERATION FAILED] Generator returned NoneType.",
                    "messages": messages or [],
                    "is_fallback": True,
                    "server_recovered": False
                }
                await asyncio.to_thread(send_to_telegram, failed_state)
                break

            draft = res.get("draft_text", "") if isinstance(res, dict) else ""
            if draft is None or not str(draft).strip():
                print(f"[Recovery Worker] ❌ Generator returned empty draft for thread {db_id}!")
                failed_thread_id = str(uuid.uuid4())[:8]
                failed_state = {
                    "message_id": failed_thread_id,
                    "sender_id": sender_id or TARGET_NAME,
                    "incoming_text": incoming_text,
                    "draft_text": "[GENERATION FAILED] Model returned empty response (Safety filter or NoneType).",
                    "messages": messages or [],
                    "is_fallback": True,
                    "server_recovered": False
                }
                await asyncio.to_thread(send_to_telegram, failed_state)
                break

            if str(draft).startswith("Error:") or "[GENERATION FAILED]" in str(draft):
                print(f"[Recovery Worker] ❌ Model blocked or returned error for thread {db_id}: {draft}")
                failed_thread_id = str(uuid.uuid4())[:8]
                failed_state = {
                    "message_id": failed_thread_id,
                    "sender_id": sender_id or TARGET_NAME,
                    "incoming_text": incoming_text,
                    "draft_text": f"[GENERATION FAILED] {draft}",
                    "messages": messages or [],
                    "is_fallback": True,
                    "server_recovered": False
                }
                await asyncio.to_thread(send_to_telegram, failed_state)
                break

            if draft.startswith("⚠️ [GEMINI ERROR:") or is_quota_fallback_text(draft):
                print(f"[Recovery Worker] ⏳ API still unavailable ({draft[:60]}...). Retrying in 60s.")
                continue

            print(f"[Recovery Worker] 🟢 Server recovered! Real draft generated: '{draft[:60]}...'")
            recovered_thread_id = str(uuid.uuid4())[:8]
            recovered_state = {
                "message_id": recovered_thread_id,
                "sender_id": sender_id or TARGET_NAME,
                "incoming_text": incoming_text,
                "draft_text": draft,
                "messages": messages or [],
                "is_fallback": False,
                "server_recovered": True
            }
            
            # Seed checkpoint into LangGraph for recovered_thread_id so approve/edit flows succeed
            try:
                config = {"configurable": {"thread_id": recovered_thread_id}}
                app.update_state(config, recovered_state, as_node="wait_for_human_action")
            except Exception as e:
                print(f"[Recovery Worker] State checkpoint notice: {e}")

            await asyncio.to_thread(send_to_telegram, recovered_state)
            break
        except Exception as e:
            print(f"[Recovery Worker] ⏳ Poll attempt exception ({e}). Sending failure alert & retrying in 60s.")
            try:
                failed_thread_id = str(uuid.uuid4())[:8]
                failed_state = {
                    "message_id": failed_thread_id,
                    "sender_id": sender_id or TARGET_NAME,
                    "incoming_text": incoming_text,
                    "draft_text": f"[GENERATION FAILED] {type(e).__name__}: {str(e)[:100]}",
                    "messages": messages or [],
                    "is_fallback": True,
                    "server_recovered": False
                }
                await asyncio.to_thread(send_to_telegram, failed_state)
            except Exception as tg_err:
                print(f"[Recovery Worker] Error sending failure card to Telegram: {tg_err}")
            continue

def process_incoming_message(state: AgentState):
    """
    Generation node: wraps gemini_proxy_node with Zero-Token local fallback handling
    and launches background recovery when Google APIs are completely down.
    """
    incoming_text = state.get("incoming_text", "")
    sender_id = state.get("sender_id", "")
    thread_id = state.get("message_id", "")
    messages = state.get("messages", [])

    try:
        res = gemini_proxy_node(state)
        draft = res.get("draft_text", "") if isinstance(res, dict) else ""
        if not draft or draft.startswith("⚠️ [GEMINI ERROR:") or draft.startswith("Error:") or is_quota_fallback_text(draft):
            raise Exception(f"Gemini API unavailable: {draft}")
        return {
            "draft_text": draft,
            "status": "pending",
            "is_fallback": False
        }
    except Exception as e:
        err_str = str(e).lower()
        target_lower = TARGET_NAME.lower()
        is_429 = any(k in err_str for k in ["429", "quota", "resource_exhausted", f"m rlyy sorryy {target_lower}", "m rlyy sorryy aarushii"])
        error_type = 429 if is_429 else 503

        print(f"[Orchestrator] ⚠️ Gemini proxy failed across keys (error_type={error_type}, err={e}). Triggering Zero-Token Fallback...")
        draft_text = get_random_fallback(error_type=error_type)
        print(f"[Orchestrator] 🛡️ Selected {'429 Quota' if is_429 else '503 Stall'} excuse: '{draft_text[:50]}...'")

        if is_429:
            # 429 Quota Exhausted: do NOT launch background_recovery_worker
            print(f"[Orchestrator] 🛑 Quota exhausted (429). Background recovery worker suppressed.")
        else:
            # 503 or transient outage: launch background recovery worker
            print(f"[Orchestrator] 🔄 503/transient outage. Launching background recovery worker...")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(background_recovery_worker(
                    incoming_text=incoming_text,
                    vibe_context=state,
                    db_id=thread_id,
                    sender_id=sender_id,
                    messages=messages
                ))
            except (RuntimeError, AttributeError):
                threading.Thread(
                    target=lambda: asyncio.run(background_recovery_worker(
                        incoming_text=incoming_text,
                        vibe_context=state,
                        db_id=thread_id,
                        sender_id=sender_id,
                        messages=messages
                    )),
                    daemon=True
                ).start()

        return {
            "draft_text": draft_text,
            "status": "pending",
            "is_fallback": True
        }

def send_to_telegram(state: AgentState):
    mode = get_config("mode")
    if mode == "autopilot":
        print("[Telegram Dispatch] 🚀 AUTOPILOT MODE ACTIVE: Bypassing Telegram payload dispatch.")
        return dispatch_message(state)

    import requests
    bot_token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    
    if not bot_token or not chat_id:
        print("[Telegram Dispatch] ERROR: Missing TELEGRAM_BOT_TOKEN or AUTHORIZED_TG_USER_ID in environment.")
        return state
        
    sender_id = state.get("sender_id", "")
    incoming = state.get("incoming_text", "")
    draft = state.get("draft_text", "")
    thread_id = state.get("message_id")
    is_fallback = bool(state.get("is_fallback"))
    server_recovered = bool(state.get("server_recovered"))
    is_revision = bool(state.get("human_feedback"))
    
    print(f"[Telegram Dispatch] Preparing approval card for thread {thread_id} (is_fallback={is_fallback}, server_recovered={server_recovered})...")
    
    if is_fallback:
        if "[GENERATION FAILED]" in str(draft):
            text = (
                "❌ [GENERATION FAILED]\n\n"
                f"📩 Incoming: \"{incoming}\"\n\n"
                f"⚠️ Error: {draft}\n\n"
                "(The background generator encountered an error or empty model response.)"
            )
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "Regenerate 🔄", "callback_data": f"edit:{thread_id}"}
                    ]
                ]
            }
        else:
            text = (
                "⚠️ [API OFFLINE] Google Servers Down.\n\n"
                f"📩 Incoming: \"{incoming}\"\n\n"
                f"🤖 Proposed Stall (Local): \"{draft}\"\n\n"
                "(A background worker is polling the API and will notify you when it recovers.)"
            )
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "Send Stall Excuse ✅", "callback_data": f"approve:{thread_id}"},
                        {"text": "🎲 Swap Excuse", "callback_data": f"cycle_fallback:{thread_id}"}
                    ]
                ]
            }
    else:
        if server_recovered:
            header = "🟢 [SERVER RECOVERED] Fresh AI Draft"
        elif is_revision:
            header = "🔄 Revised AI Draft"
        else:
            header = "📩 Incoming WhatsApp Message"
            
        text = (
            f"{header}\n"
            f"From: {sender_id}\n"
            f"Text:\n\"{incoming}\"\n\n"
            "🤖 Proposed AI Draft:\n"
            f"\"{draft}\""
        )
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve & Send ✅", "callback_data": f"approve:{thread_id}"},
                    {"text": "Regenerate 🔄", "callback_data": f"edit:{thread_id}"}
                ]
            ]
        }

    # Always deliver as a fresh message to the bottom of the Telegram chat
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": keyboard
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        resp_data = response.json()
        new_msg_id = resp_data.get("result", {}).get("message_id")
        if new_msg_id:
            save_pending_card(
                telegram_message_id=new_msg_id,
                thread_id=thread_id,
                sender_id=sender_id,
                incoming_text=incoming,
                draft_text=draft,
                status="pending"
            )
            card_type = "fallback" if is_fallback else ("recovered" if server_recovered else ("revised" if is_revision else "new"))
            print(f"[Telegram Dispatch] Success! Delivered {card_type} approval card #{new_msg_id}.")
    except requests.exceptions.HTTPError as e:
        print(f"[Telegram Dispatch] HTTPError: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"[Telegram Dispatch] Exception: {e}")
        
    return state

def wait_for_human_action(state: AgentState):
    # This node acts as a dummy node for the interrupt.
    # Execution halts BEFORE this node due to interrupt_before.
    # When resumed via update_state(as_node="wait_for_human_action"), it simply passes through.
    return {}

def process_human_feedback(state: AgentState):
    import os
    from google import genai
    from google.genai import types
    print(f"[Orchestrator] Processing feedback: '{state.get('human_feedback')}'")
    
    incoming_text = state.get("incoming_text", "")
    old_draft = state.get("draft_text", "") or ""
    feedback = state.get("human_feedback", "") or ""
    
    is_error_draft = old_draft.startswith("⚠️ [GEMINI ERROR") or is_quota_fallback_text(old_draft)
    feedback_clean = feedback.strip().lower()
    is_generic_retry = feedback_clean in [
        "retry", "try again", "regenerate", "redo", "again", "pls retry", "please retry", ""
    ] or feedback_clean.startswith("retry")

    if is_error_draft:
        print(f"[Orchestrator] Regenerating from error/quota draft (is_generic_retry={is_generic_retry})...")
        if is_generic_retry:
            proxy_res = gemini_proxy_node({
                "incoming_text": incoming_text,
                "messages": state.get("messages", [])
            })
            new_draft = proxy_res.get("draft_text", "")
        else:
            prompt = (
                f"{TARGET_NAME} sent this message:\n'{incoming_text}'\n\n"
                f"Please draft Siddharth's reply following these instructions:\n'{feedback}'\n\n"
                f"Output ONLY Siddharth's raw text message reply (multi-line bursts are natural). Do not include quotes."
            )
            feedback_config = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.65
            )
            new_draft = generate_with_key_rotation(prompt, feedback_config)
    else:
        prompt = (
            f"You previously drafted this reply to '{incoming_text}':\n'{old_draft}'\n\n"
            f"The user wants you to revise it with these exact instructions:\n'{feedback}'\n\n"
            f"Please output ONLY the revised raw text message. Do not include quotes."
        )
        print(f"[Orchestrator] Regenerating draft via {MODEL_NAME} with key rotation (4096 tokens headroom)...")
        feedback_config = types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.65
        )
        new_draft = generate_with_key_rotation(prompt, feedback_config)
    
    # ---------------------------------------------------------
    # CONTINUOUS LEARNING: Log the correction to memory ONLY if valid
    # (Never log error drafts, quota fallbacks, or generic retries)
    # ---------------------------------------------------------
    from src.core.memory_manager import log_feedback
    if not is_error_draft and not is_quota_fallback_text(new_draft) and not new_draft.startswith("⚠️ [GEMINI ERROR"):
        log_feedback(old_draft, feedback, new_draft)
    
    return {"draft_text": new_draft, "status": "pending", "human_feedback": None}

def dispatch_message(state: AgentState):
    mode = get_config("mode")
    status = state.get("status")

    # If training mode and not yet approved by human, dispatch to Telegram
    if mode == "training" and status not in ["approve", "approved"]:
        return send_to_telegram(state)

    import time
    from src.gateways.whatsapp_client import send_whatsapp_message
    from src.core.memory_manager import save_message
    sender_id = state.get("sender_id")
    final_draft = state.get("draft_text", "")
    thread_id = state.get("message_id")
    
    # If state draft_text is missing, check pending_cards table by thread_id
    if not final_draft and thread_id:
        card = get_pending_card_by_thread(thread_id)
        if card:
            final_draft = card.get("draft_text", "")
            if not sender_id:
                sender_id = card.get("sender_id")
    
    # CRITICAL SAFETY GUARD: Never dispatch empty or error drafts to WhatsApp
    if not final_draft or final_draft.startswith("⚠️ [GEMINI ERROR:"):
        print(f"[Orchestrator] BLOCKED DISPATCH: Draft is empty or error message: '{final_draft}'")
        return {"status": "error_blocked"}

    print(f"[Orchestrator] Dispatching message to {sender_id} (mode={mode})")
    
    # Split by newline (ignoring empty lines) to dispatch sequential bubbles
    bubbles = [line.strip() for line in final_draft.split("\n") if line.strip()]
    if not bubbles and final_draft.strip():
        bubbles = [final_draft.strip()]

    for i, bubble in enumerate(bubbles):
        send_whatsapp_message(sender_id, bubble)
        if i < len(bubbles) - 1:
            time.sleep(0.4)
    
    # Permanently log the AI's dispatched message directly to the chats table (messages)
    # (unless it is a quota excuse fallback or stall draft)
    if is_quota_fallback_text(final_draft) or is_stall_draft(final_draft):
        print(f"[Orchestrator] 🛡️ Fallback/Stall detected ('{final_draft[:40]}...'). Dispatched to WhatsApp but NOT logged to messages database.")
    else:
        save_message("Siddharth", final_draft)
        
        # Trigger background learning engine every 8th message
        try:
            from src.core.memory_manager import get_total_messages_count
            total_messages = get_total_messages_count()
            if total_messages > 0 and total_messages % 8 == 0:
                print(f"[Orchestrator] 🚀 Modulo 8 triggered (total_messages={total_messages}). Launching background fact extraction...")
                try:
                    loop = asyncio.get_running_loop()
                    from src.core.fact_extractor import extract_and_store_facts
                    loop.create_task(extract_and_store_facts())
                except (RuntimeError, AttributeError):
                    from src.core.fact_extractor import extract_and_store_facts_sync
                    threading.Thread(target=extract_and_store_facts_sync, daemon=True).start()
        except Exception as e:
            print(f"[Orchestrator] Fact extraction trigger error: {e}")
    
    return {"status": "dispatched"}

# 3. Define Conditional Routing
def route_after_generation(state: AgentState):
    mode = get_config("mode")
    if mode == "autopilot":
        print("[Orchestrator] 🚀 Autopilot active: routing directly to dispatch_message (bypassing Telegram)")
        return "dispatch_message"
    return "send_to_telegram"

def route_after_human(state: AgentState):
    status = state.get("status")
    if status in ["approve", "approved"]:
        return "dispatch_message"
    elif status in ["edit", "revise", "suggest_changes"]:
        return "process_human_feedback"
    elif status in ["discard", "discarded"]:
        return END
    # Fallback to end if status is unrecognized
    return END

# 4. Build StateGraph
builder = StateGraph(AgentState)

builder.add_node("gemini_proxy_node", process_incoming_message)
builder.add_node("send_to_telegram", send_to_telegram)
builder.add_node("wait_for_human_action", wait_for_human_action)
builder.add_node("process_human_feedback", process_human_feedback)
builder.add_node("dispatch_message", dispatch_message)

# Define edges
builder.add_edge(START, "gemini_proxy_node")
builder.add_conditional_edges("gemini_proxy_node", route_after_generation)
builder.add_edge("send_to_telegram", "wait_for_human_action")
builder.add_edge("process_human_feedback", "send_to_telegram") # Loop back to telegram after editing
builder.add_edge("dispatch_message", END)

# Define conditional edges leaving the wait node
builder.add_conditional_edges("wait_for_human_action", route_after_human)

# 5. Set up Persistence and Compile
os.makedirs(DATA_DIR, exist_ok=True)
conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
sqlite_saver = SqliteSaver(conn)
sqlite_saver.setup()

app = builder.compile(
    checkpointer=sqlite_saver, 
    interrupt_before=["wait_for_human_action"]
)

# 6. Helper Functions for Integration
def handle_incoming_whatsapp_burst(sender_id: str, combined_incoming: str, lines: Optional[list] = None):
    """
    Handles a debounced WhatsApp burst from the target contact.
    1. Saves each incoming line to SQLite short-term memory in order.
    2. Does NOT re-append previously carded messages.
    3. Invokes Gemini proxy node with this fresh burst.
    4. Delivers an independent Telegram approval card.
    """
    from src.core.memory_manager import save_message, get_recent_history
    
    # 1. Log each incoming message line to short-term memory in order
    if lines:
        for line in lines:
            if line.strip():
                save_message(TARGET_NAME, line.strip())
    else:
        for line in combined_incoming.split("\n"):
            if line.strip():
                save_message(TARGET_NAME, line.strip())
                
    thread_id = str(uuid.uuid4())[:8]
    print(f"[Orchestrator] Fresh debounced burst from {sender_id} (thread {thread_id}): '{combined_incoming[:60]}...'")
    
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "message_id": thread_id,
        "sender_id": sender_id,
        "incoming_text": combined_incoming,
        "messages": get_recent_history(limit=20)
    }
    return app.invoke(initial_state, config)

def handle_incoming_whatsapp_message(sender_id: str, incoming_text: str):
    """Single-message entry point that processes without re-appending old cards."""
    return handle_incoming_whatsapp_burst(sender_id, incoming_text, [incoming_text])

def start_message_pipeline(message_id: str, sender_id: str, incoming_text: str):
    return handle_incoming_whatsapp_message(sender_id, incoming_text)

def resume_message_pipeline(thread_id: str, action: str, feedback: Optional[str] = None, card_msg_id: Optional[int] = None):
    """
    Resumes message pipeline for a specific card thread without any superseded blocking.
    Every card clicked on Telegram executes independently.
    """
    print(f"[Orchestrator] Resuming thread {thread_id} with action='{action}' (card_msg_id={card_msg_id})")
    
    # Retrieve card details to ensure state integrity
    card = None
    if card_msg_id:
        card = get_pending_card(card_msg_id)
    if not card:
        card = get_pending_card_by_thread(thread_id)
    if not card and str(thread_id).isdigit():
        card = get_pending_card(int(thread_id))
        
    actual_thread_id = card["thread_id"] if card else thread_id
    config = {"configurable": {"thread_id": actual_thread_id}}
    
    update_dict = {"human_feedback": feedback, "status": action}
    if card:
        update_dict["message_id"] = actual_thread_id
        update_dict["sender_id"] = card["sender_id"]
        update_dict["draft_text"] = card["draft_text"]
        update_dict["incoming_text"] = card["incoming_text"]

    app.update_state(
        config, 
        update_dict, 
        as_node="wait_for_human_action"
    )
    result = app.invoke(None, config)
    
    # Update card status in independent card table
    target_msg_id = card_msg_id or (card["telegram_message_id"] if card else None)
    if target_msg_id:
        if action == "approve":
            update_pending_card_status(target_msg_id, "approved")
        elif action == "discard":
            update_pending_card_status(target_msg_id, "discarded")
        
    return result
