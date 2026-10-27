import os
import re
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types

from src.core.config import GEMINI_KEYS, TARGET_NAME
from src.core.vector_memory import chroma_db, classify_message_vibe

def analyze_vibe(incoming_text: str, messages: Optional[List[Dict]] = None) -> str:
    """
    Identifies the current vibe / energy of the incoming message and recent context.
    Matches the specific energy required for the drafting system prompt.
    """
    text_lower = incoming_text.lower().strip()
    
    # 1. Check for memory quiz / question testing
    quiz_triggers = [
        "kab tha", "kab thi", "bata then", "bata na", "yaad hai",
        "kaha tha", "kaha mile", "first date", "pehli baar",
        "which date", "tell me when", "what date", "remember when",
        "kis din", "kab gaye"
    ]
    if any(q in text_lower for q in quiz_triggers):
        return "playful testing / checking memory, slightly teasing"

    # 2. Use vector memory's mood classifier
    vibe_data = classify_message_vibe(incoming_text)
    mood = vibe_data.get("mood", "casual")

    if mood == "affectionate":
        return "warm, affectionate, relaxed"
    elif mood == "playful":
        return "playful Hinglish banter, light teasing"
    elif mood == "sarcastic":
        return "dry banter, casual irony"
    elif mood == "serious_upset":
        return "mature, attentive, serious reassurance"
    
    return "dry, casual, and natural"

def generate_standalone_query(incoming_text: str, messages: Optional[List[Dict]] = None) -> str:
    """
    Pre-Retrieval Query Rewriter:
    Ingests the last 3-4 messages from conversation history + incoming text
    and generates a standalone search query for Memories.md retrieval.
    """
    clean_incoming = incoming_text.strip()
    if not clean_incoming:
        return "couple relationship history timeline memories"

    recent_history = messages or []
    history_lines = []
    target_name = os.getenv("TARGET_CONTACT_NAME", TARGET_NAME or "PersonName")

    # Ingest the last 3-4 messages from history
    for msg in recent_history[-4:]:
        role = msg.get("role", "")
        role_name = "Siddharth" if role in ["Siddharth", "ai", "model", "gpt"] else target_name
        content = (msg.get("content") or msg.get("text") or "").strip()
        if content:
            history_lines.append(f"{role_name}: {content}")

    # If no prior history, return the incoming text directly
    if not history_lines:
        return clean_incoming

    history_str = "\n".join(history_lines)

    # Fast heuristic check: if the message is clearly self-contained small talk with no question, return as is
    t_lower = clean_incoming.lower()
    is_question_or_reference = any(k in t_lower for k in [
        "kab", "kaha", "when", "where", "date", "pehle", "yaad", "kaun",
        "bata", "kis din", "first", "timeline", "kiske", "kisne", "then",
        "which", "what", "place", "location"
    ]) or len(clean_incoming.split()) <= 5

    if not is_question_or_reference:
        return clean_incoming

    # Call Gemini model for fast query rewriting
    prompt = f"""You are a query rewriter for retrieving facts and dates from a couple's relationship diary (Memories.md).
Based on the recent conversation context and the latest message, write a concise standalone search query (3 to 8 words) that captures what past fact, date, event, or memory is being asked about.

CRITICAL: Do not hallucinate or invent context, dates, or events. If the user's message is extremely brief or ambiguous (e.g., 'boll', 'kab', 'tell me'), and the exact entity is not explicitly clear from the recent chat history, DO NOT invent a search query. Instead, output the standalone query exactly as the literal translation of the user's message without adding any guessed details.

Recent Conversation Context (Last 3-4 messages):
{history_str}

Latest Message:
"{clean_incoming}"

If the latest message is an ongoing question or pronoun-heavy reference to the previous messages (e.g., "kab tha?", "kaha gaye the?", "tell me the date"), resolve the reference using the prior messages (e.g., "Date of our first date at Versova Cat Cafe").
If the message is pure small talk with no reference to past memories, output the message itself.

Output ONLY the standalone search query in 3-8 words. No explanations, no quotes, no markdown."""

    for key in GEMINI_KEYS:
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt
            )
            raw_text = (getattr(response, "text", "") or "").strip()
            if raw_text:
                cleaned = raw_text.splitlines()[0].strip().strip("'\"`*")
                if cleaned:
                    return cleaned
        except Exception:
            continue

    # Fallback heuristic if API call is skipped or fails:
    last_partner_msg = ""
    for line in reversed(history_lines):
        if target_name in line:
            last_partner_msg = line.replace(f"{target_name}:", "").strip()
            break
            
    if last_partner_msg and last_partner_msg != clean_incoming:
        return f"{last_partner_msg} {clean_incoming}"[:80]

    return clean_incoming

def analyzer_node(state: dict) -> dict:
    """
    LangGraph Pre-Retrieval / Analyzer Node:
    1. Ingests the last 3-4 messages from the conversation history and incoming text.
    2. Identifies the conversation vibe/energy.
    3. Rewrites the raw incoming follow-up text into a contextualized standalone_search_query.
    4. Passes standalone_search_query to chroma_db.similarity_search() to retrieve rag_context from Memories.md.
    """
    incoming_text = state.get("incoming_text", "")
    messages = state.get("messages", [])
    if not messages:
        from src.core.memory_manager import get_recent_history
        messages = get_recent_history(limit=10)

    # 1. Determine conversation vibe
    vibe = analyze_vibe(incoming_text, messages)

    # 2. Pre-Retrieval Query Rewriting
    standalone_query = generate_standalone_query(incoming_text, messages)
    print(f"[Analyzer Node] Raw incoming: '{incoming_text}' -> Standalone query: '{standalone_query}' | Vibe: '{vibe}'")

    # 3. Retrieve RAG context from ChromaDB using the rewritten standalone query
    from src.core.vector_memory import chroma_db
    retrieved_docs = chroma_db.similarity_search(standalone_query, k=3)
    rag_context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant memories found in Memories.md."

    return {
        "vibe": vibe,
        "standalone_search_query": standalone_query,
        "rag_context": rag_context
    }
