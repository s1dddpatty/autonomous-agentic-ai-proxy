import os
import re
import json
import sqlite3
import asyncio
from typing import List
from google.genai import types

from src.core.gemini_node import generate_with_key_rotation, is_quota_fallback_text
from src.core.vector_memory import save_semantic_fact

FACT_EXTRACTOR_SYSTEM_PROMPT = (
    "You are the analytical memory engine for a digital twin. "
    "Analyze this recent chat transcript and extract ONLY permanent, significant facts, "
    "relationship details, or major ongoing events (e.g., family hobbies, allergies, upcoming trips, ongoing rules). "
    "STRICTLY IGNORE trivial daily updates (e.g., what they ate for lunch today, feeling tired, going to sleep). "
    "Return ONLY a valid JSON list of extracted fact strings. "
    "If there are no new significant facts, return an empty list []."
)

def get_recent_chat_transcript(limit: int = 20) -> str:
    """
    Connects to chats.db and fetches the last 15-20 messages in chronological order.
    """
    from src.core.config import DB_PATH
    if not os.path.exists(DB_PATH):
        return ""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    lines = []
    for role, content in reversed(rows):
        clean_content = (content or "").strip()
        if clean_content:
            lines.append(f"{role}: {clean_content}")
    return "\n".join(lines)

def parse_facts_json(raw_text: str) -> List[str]:
    """
    Robustly parses a JSON list of fact strings from the model output,
    handling markdown fences and surrounding commentary.
    """
    if not raw_text:
        return []
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE).strip()
    
    match = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(item).strip() for item in data if item and str(item).strip()]
        except Exception as e:
            print(f"[Fact Extractor] JSON parsing warning: {e}")
    return []

def extract_and_store_facts_sync(transcript: str = None, limit: int = 20) -> List[str]:
    """
    Synchronous implementation of permanent fact extraction using gemini-flash-latest with 5-key rotation.
    """
    if transcript is None:
        transcript = get_recent_chat_transcript(limit=limit)

    if not transcript or not transcript.strip():
        print("[Fact Extractor] No chat transcript available to extract facts from.")
        return []

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
        system_instruction=FACT_EXTRACTOR_SYSTEM_PROMPT,
        response_mime_type="application/json"
    )

    contents = f"Recent chat transcript:\n{transcript}\n\nExtracted Facts JSON:"
    print(f"[Fact Extractor] Analyzing recent chat transcript ({len(transcript.splitlines())} messages) for permanent facts...")
    raw_response = generate_with_key_rotation(contents=contents, config=config)

    if not raw_response or is_quota_fallback_text(raw_response) or raw_response.startswith("⚠️ [GEMINI ERROR"):
        print(f"[Fact Extractor] Skipping fact storage due to LLM error/fallback: {raw_response[:80]}")
        return []

    facts = parse_facts_json(raw_response)
    saved_facts = []
    for fact in facts:
        save_semantic_fact(fact)
        saved_facts.append(fact)

    print(f"[Fact Extractor] 🧠 Learned {len(saved_facts)} new permanent facts.")
    for f in saved_facts:
        print(f"  - {f}")
    return saved_facts

async def extract_and_store_facts(transcript: str = None, limit: int = 20) -> List[str]:
    """
    Asynchronous entry point that offloads the blocking LLM call to a worker thread.
    Never blocks or delays the active event loop or message dispatches.
    """
    return await asyncio.to_thread(extract_and_store_facts_sync, transcript, limit)
