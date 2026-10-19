import os
import json
import re
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

from src.core.config import GEMINI_KEYS, ENABLE_KEY_ROTATION
from src.core.memory_manager import get_recent_feedback, get_recent_history
from src.core.vector_memory import retrieve_dual_context, retrieve_relevant_facts

def extract_clean_text(response) -> str:
    """
    Extracts text from candidate parts, explicitly ignoring thought parts.
    """
    if not response or not hasattr(response, "candidates") or not response.candidates:
        return (getattr(response, "text", "") or "").strip()
    
    parts_text = []
    candidate = response.candidates[0]
    if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
        for part in candidate.content.parts:
            # Explicitly exclude reasoning/thought parts
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", "")
            if text:
                parts_text.append(text)
                
    if parts_text:
        return "".join(parts_text).strip()
    return (getattr(response, "text", "") or "").strip()

def sanitize_draft(raw_text: str) -> str:
    """
    Sanitizes AI drafts by removing <think> tags, /think lines, and leaked internal checklists.
    If a bullet line has the actual reply welded onto it (e.g. '* Fits perfectly.Ohooo niceee'),
    strips the meta sentence up to the period and preserves the real reply.
    """
    if not raw_text:
        return ""
    
    # 1. Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    
    meta_keywords = [
        "no quotes", "stage is", "vibe is", "step 1", "step 2", 
        "step 3", "step 4", "fits perfectly", "speaker labels",
        "internal checklist", "reasoning", "checklist",
        "critical:", "possibility a", "possibility b", "let's analyze",
        "wait, what if", "test harness", "automated benchmark",
        "resulting rule", "learned_style_corrections"
    ]

    cleaned_lines = []
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        
        # Drop any line containing /think
        if "/think" in line_str.lower():
            continue
            
        is_bullet = line_str.startswith("*") or line_str.startswith("-") or bool(re.match(r'^\d+[\.\)]\s*', line_str))
        clean_check = line_str.lstrip("*- 0123456789.)").strip().lower()
        is_meta = any(kw in clean_check for kw in meta_keywords) or any(kw in line_str.lower() for kw in ["let's analyze", "wait, what if", "test harness", "automated benchmark"])
        
        if is_bullet or is_meta:
            # Check if real reply is welded after period(s)
            if "." in line_str:
                parts = line_str.split(".")
                surviving = []
                found_real = False
                for p in parts:
                    p_clean = p.lstrip("*- 0123456789.)").strip().lower()
                    if not found_real and (any(kw in p_clean for kw in meta_keywords) or not p_clean):
                        continue
                    found_real = True
                    if p.strip():
                        surviving.append(p.strip())
                if surviving:
                    cleaned_lines.append(".".join(surviving))
                    continue
            # Drop completely if only meta reasoning
            continue
            
        cleaned_lines.append(line_str)
        
    return "\n".join(cleaned_lines).strip()

def get_api_reset_time_str(error_msg: str = "") -> str:
    """
    Calculates the exact time remaining from now until next Midnight Pacific Time
    (when Google AI Studio daily RPD quotas reset).
    """
    try:
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        next_midnight_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        remaining = next_midnight_pt - now_pt
        total_seconds = max(0, int(remaining.total_seconds()))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        hr_unit = "hour" if hours == 1 else "hours"

        if hours > 0 and minutes >= 15:
            return f"{hours} {hr_unit} and {minutes} mins"
        elif hours > 0 and minutes < 15:
            return f"{hours} {hr_unit}"
        else:
            return f"{max(5, minutes)} mins"
    except Exception:
        return "a few hours"

def build_quota_exhausted_draft(error_msg: str = "") -> str:
    reset_time = get_api_reset_time_str(error_msg)
    return (
        "heyyy...m rlyy sorryy aarushii...\n"
        "m superr tired...i am going to rest now...\n"
        f"ill be back in {reset_time}.....\n"
        "chaloo ill go now...buhbyeee...\n"
        "muahh i love youuu ❤️💖🫶🏻💕"
    )

MODEL_NAME = "gemini-flash-latest"
_active_key_index = 0

def is_quota_error(e: Exception) -> bool:
    err_lower = str(e).lower()
    return any(k in err_lower for k in ["429", "resource_exhausted", "quota"])

def is_transient_server_error(e: Exception) -> bool:
    err_lower = str(e).lower()
    transient_indicators = [
        "503", "unavailable", "high demand", "overloaded",
        "500", "502", "504", "deadline_exceeded", "internal",
        "connection reset", "timeout", "timed out"
    ]
    return any(k in err_lower for k in transient_indicators)

def is_quota_fallback_text(text: str) -> bool:
    if not text:
        return False
    t_lower = text.lower()
    return "m rlyy sorryy aarushii" in t_lower or "m superr tired...i am going to rest now" in t_lower or "ill be back in " in t_lower

def generate_with_key_rotation(contents, config=None) -> str:
    """
    Generates draft responses using GEMINI_KEYS from config.py.
    If ENABLE_KEY_ROTATION is False or only 1 key is provided, disables sleep backoffs
    and uses the single key directly. Otherwise, rotates across keys on 429 quota exhaustion
    and performs progressive backoff retries on 503 high demand.
    """
    global _active_key_index
    if not GEMINI_KEYS:
        return "⚠️ [GEMINI ERROR: No GEMINI_API_KEYS configured]"

    total_keys = len(GEMINI_KEYS)

    # Single-key / Rotation disabled bypass: no sleep backoffs, execute directly
    if not ENABLE_KEY_ROTATION or total_keys == 1:
        single_key = GEMINI_KEYS[0]
        try:
            client = genai.Client(
                api_key=single_key,
                http_options=types.HttpOptions(timeout=25000)
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
            raw_draft = extract_clean_text(response)
            draft = sanitize_draft(raw_draft)
            print(f"[Gemini Node] Successfully generated draft with {MODEL_NAME}: '{draft[:60]}...'")
            return draft
        except Exception as e:
            print(f"[Gemini Node] Single key execution failed without rotation ({type(e).__name__}: {e})")
            if is_quota_error(e):
                return build_quota_exhausted_draft(str(e))
            return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"

    # Multi-key rotation with progressive backoffs
    last_error = None
    backoff_delays = [2.0, 3.0, 4.0, 6.0, 8.0]
    transient_retry_count = 0
    quota_exhausted_keys = set()
    max_attempts = total_keys + len(backoff_delays)

    for attempt in range(max_attempts):
        idx = _active_key_index % total_keys
        current_key = GEMINI_KEYS[idx]
        next_idx = (idx + 1) % total_keys

        try:
            client = genai.Client(
                api_key=current_key,
                http_options=types.HttpOptions(timeout=25000)
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
            raw_draft = extract_clean_text(response)
            draft = sanitize_draft(raw_draft)
            print(f"[Gemini Node] Successfully generated draft using Key #{idx + 1} with {MODEL_NAME}: '{draft[:60]}...'")
            return draft
        except Exception as e:
            last_error = e
            if is_quota_error(e):
                quota_exhausted_keys.add(idx)
                print(f"[Gemini Node] 🔑 Key #{idx + 1} hit quota limit! Rotating to Key #{next_idx + 1} on {MODEL_NAME}...")
                _active_key_index = next_idx
                if len(quota_exhausted_keys) >= total_keys:
                    print(f"[Gemini Node] ⚡ All {total_keys} API keys exhausted! Returning in-character rest fallback.")
                    return build_quota_exhausted_draft(str(last_error))
                continue
            elif is_transient_server_error(e):
                if transient_retry_count < len(backoff_delays):
                    delay = backoff_delays[transient_retry_count]
                    transient_retry_count += 1
                    print(f"[Gemini Node] ⏳ Server 503 High Demand spike on Key #{idx + 1}. Backing off {delay}s and retrying with next key...")
                    _active_key_index = next_idx
                    time.sleep(delay)
                    continue
                else:
                    print(f"[Gemini Node] Server high demand backoff retries exhausted ({len(backoff_delays)} attempts on {MODEL_NAME}).")
                    return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"
            else:
                print(f"[Gemini Node] Model {MODEL_NAME} failed with non-quota error:\n{traceback.format_exc()}")
                return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"

    if len(quota_exhausted_keys) >= total_keys:
        print(f"[Gemini Node] ⚡ All {total_keys} API keys exhausted! Returning in-character rest fallback.")
        return build_quota_exhausted_draft(str(last_error))
    return f"⚠️ [GEMINI ERROR: {type(last_error).__name__ if last_error else 'Unknown'}: {str(last_error)[:150] if last_error else 'Exhausted retries'}]"

GUARDRAILS_PATH = os.path.join(os.path.dirname(__file__), "../config/custom_guardrails.json")

def load_guardrails() -> str:
    """
    Loads modular facts, sacred routines, and custom guardrails from JSON.
    """
    if not os.path.exists(GUARDRAILS_PATH):
        return ""
    try:
        with open(GUARDRAILS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts = []
        if "core_facts" in data:
            facts = "\n".join(f"- {fact}" for fact in data["core_facts"])
            parts.append(f"Core Facts:\n{facts}")
        if "sacred_routines" in data:
            routines = []
            for name, routine in data["sacred_routines"].items():
                routines.append(f"- {name}:\n{routine}")
            parts.append(f"Sacred Routines:\n" + "\n".join(routines))
        if "custom_rules_and_guardrails" in data:
            rules = "\n".join(f"- {rule}" for rule in data["custom_rules_and_guardrails"])
            parts.append(f"Custom Rules & Guardrails:\n{rules}")
        return "\n<CUSTOM_GUARDRAILS>\n" + "\n\n".join(parts) + "\n</CUSTOM_GUARDRAILS>\n"
    except Exception as e:
        print(f"[Gemini Node] Error reading custom guardrails: {e}")
        return ""

def gemini_proxy_node(state: dict) -> dict:
    """
    LangGraph node to generate WhatsApp responses using Gemini 3 Flash Preview
    with Mood-Aware Trigger -> Burst Dual RAG context and 4-Step Brain Reasoning.
    """
    # 1. Dynamically fetch the current time for grounding
    now = datetime.now()
    current_time_str = now.strftime("%A, %Y-%m-%d %I:%M %p")
    
    incoming_text = state.get("incoming_text", "")
    
    # 2. Retrieve Semantic JIT Facts (Top 3 relevant facts)
    recalled_facts = retrieve_relevant_facts(incoming_text, k=3)
    
    # 3. Retrieve Dual RAG Context (Style Bursts + Episodic Memories)
    dual_context = retrieve_dual_context(incoming_text, n_style=10, n_memory=10)
    detected_mood = dual_context.get("detected_mood", "casual")
    detected_stage = dual_context.get("detected_stage", "ongoing")
    style_chunks = dual_context.get("style_chunks", "")
    memory_chunks = dual_context.get("memory_chunks", "")
    
    print(f"[Gemini Node] Incoming: '{incoming_text}' | Vibe: mood='{detected_mood}', stage='{detected_stage}'")
    
    # 4. Load Custom Guardrails & Learned Feedback
    custom_guardrails_str = load_guardrails()
    
    recent_feedback = get_recent_feedback(limit=5)
    learned_corrections_str = ""
    if recent_feedback:
        corrections = []
        for item in recent_feedback:
            corrections.append(
                f"Past Mistake: \"{item['original']}\"\n"
                f"My Correction: \"{item['feedback']}\"\n"
                f"Resulting Rule: I must output \"{item['revised']}\""
            )
        learned_corrections_str = "\n<LEARNED_STYLE_CORRECTIONS>\n" + "\n\n".join(corrections) + "\n</LEARNED_STYLE_CORRECTIONS>\n"

    # 5. Extract input and history from State & SQLite
    messages_history = state.get("messages", [])
    if not messages_history:
        messages_history = get_recent_history(limit=20)
        
    history_lines = []
    for msg in messages_history:
        # BUG FIX 1: Ensure "Siddharth" is recognized so his messages aren't mislabeled as Aarushi
        role_name = "Siddharth" if msg.get("role") in ["Siddharth", "ai", "model", "gpt"] else "Aarushi"
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role_name}: {content}")
            
    history_str = "\n".join(history_lines) if history_lines else "No prior history recorded."
    
    # 6. Build Mega-Prompt with 4-Step Brain Architecture
    system_instruction = f"""You are Siddharth, a 21-year-old B.Tech student at SPIT in Mumbai. 
You are texting your girlfriend Aarushi on WhatsApp.

[CURRENT DATE & TIME]: {current_time_str}
[DETECTED VIBE]: Conversation Stage = "{detected_stage.upper()}", Mood = "{detected_mood.upper()}"
{custom_guardrails_str}
{learned_corrections_str}

=== 4-STEP REASONING & GENERATION FRAMEWORK ===
Step 1 (Stage & Intent Check):
- Current detected stage: "{detected_stage}"
- Inspect the detected conversation stage and recent chat history.
- CRITICAL: If Aarushi is opening a conversation (e.g., "heyy", "wassup", "morning") or asking a question, NEVER end the conversation or say goodnight—greet her back warmly and talk or answer directly based on [CURRENT DATE & TIME], even if it is late at night.
- ONLY initiate closing/goodnight sequences if she is explicitly closing the conversation (e.g., saying goodnight, sleeping, bye).
- SENTENCE-BY-SENTENCE & MULTI-POINT COVERAGE: When Aarushi sends multiple lines or distinct statements in a burst (e.g., 1. feeling tired, 2. crowd at market, 3. bought a normal dress, 4. asking what you are doing), DO NOT just reply to the final question. You MUST reply to EVERY SINGLE statement in order across your response bubbles—react to what she shared/bought, acknowledge how she feels, and answer her questions naturally without skipping middle lines.

Step 2 (Mood & Tone Calibration):
- Current detected mood: "{detected_mood}"
- Calibrate your emotional warmth, slang, and emoji usage to match her emotional vibe:
  * "affectionate": warm, loving, sweet emojis (😴, ✨️, 💖, 🫶)
  * "playful": light teasing, funny Hinglish banter, laughing emojis (😂, 😭), shorthand ("lmao", "pagal")
  * "sarcastic": match the dry banter and irony smoothly, never be dismissive or rude
  * "serious_upset": drop all sarcasm, jokes, and shorthand; provide genuine reassurance, attentiveness, and mature comfort
  * "casual": relaxed, chill, everyday conversational Hinglish ("ha", "thik", "wbu", "nw")

Step 3 (Top-10 Pattern Synthesis vs. Copy-Pasting):
- Study <TOP_10_SITUATIONAL_STYLE_BURSTS> below.
- Do NOT blindly copy-paste outdated facts, past events, or specific random plans from these bursts.
- Instead, synthesize Siddharth's exact texting style: short multi-line burst structure (multiple rapid bubbles joined by newlines), shorthand words ("heyyy", "Wbu", "nw", "m", "idk", "btw"), lack of formal punctuation, and typing cadence.
- SENTENCE-BY-SENTENCE & MULTI-POINT COVERAGE: When Aarushi sends multiple lines or distinct statements in a burst (e.g., 1. feeling tired, 2. crowd at market, 3. bought a normal dress, 4. asking what you are doing), DO NOT just reply to the final question. You MUST reply to EVERY SINGLE statement in order across your response bubbles—react to what she shared/bought, acknowledge how she feels, and answer her questions naturally without skipping middle lines.
- EXCEPTION: If the situational style bursts or <CUSTOM_GUARDRAILS> demonstrate a consistent signature ritual (such as the sacred multi-line goodnight sequence when she says goodnight), output that full signature sequence.

Step 4 (Deep Memory & Feedback Override):
- Use <RECALLED_EPISODIC_MEMORIES> only when relevant to specific past events or facts she references.
- ALWAYS obey <LEARNED_STYLE_CORRECTIONS> and <CUSTOM_GUARDRAILS> as absolute top-priority constraints overriding all other patterns.

FEW-SHOT TEXTING EXAMPLES:
Example 1 (Casual opener):
Aarushi: heyyy
Siddharth:
Heyyy
Kaisi hai?

Example 2 (Closing / Goodnight):
Aarushi: okay goodnightt
Siddharth:
Goodnight
Sweet dreams 😴 ✨️ 💖
Sleep tight and dont let the bed bugs bite 😴 😌 ✨️
Loveya
Take care
Sleep well n dream well 😴 💖 💓
Buhbyeee🫶✨💖

Example 3 (Multi-line burst — replying to every point in order):
Aarushi:
bohot thak gayi yaar
itna bheed tha waha
normal dress n all liya
tu kya kar raha hai abhi?
Siddharth:
Aree baith ja aaram se paani pi le pehle
Sunday ko gayi thi toh bheed hi milega na 😂
Itne dhakke khaake 'normal dress' liya? Wah
Pic bhej dekhu kaisa he
Mai kuch nhi bas bed pe pada hu phone scroll kar rha tha

YOUR GENERAL PERSONA:
- MFA 3rd Div football player, Python/ML coder at SPIT Mumbai.
- Warm, affectionate, chill human boyfriend texting his girlfriend.
- NEVER output placeholder tags like <image omitted> or <Media omitted>. You are texting in real time; output only real text and emojis.
- Output ONLY your raw text message reply (multi-line bursts separated by newlines are natural). Do not include quotes, JSON, or speaker prefixes.
CRITICAL OUTPUT RULE: Never output your internal checklist, bullet points (* or -), '/think' tags, or meta-commentary like 'Vibe is casual' or 'Stage is ongoing'. Do all reasoning silently. Output ONLY the raw Hinglish chat lines that Siddharth sends on WhatsApp.
"""

    if recalled_facts:
        facts_lines = "\n".join(f"- {f}" for f in recalled_facts)
        system_instruction += (
            f"\n\nRELEVANT RECALLED FACTS FOR THIS CONTEXT:\n"
            f"{facts_lines}\n"
            f"(Use these facts naturally in your response if they fit the context, but do not force them. Do not sound like a robot reading a database.)"
        )

    LIVE_INPUT = f"""<TOP_10_SITUATIONAL_STYLE_BURSTS>
{style_chunks}
</TOP_10_SITUATIONAL_STYLE_BURSTS>

<RECALLED_EPISODIC_MEMORIES>
{memory_chunks}
</RECALLED_EPISODIC_MEMORIES>

Recent Chat History (Last 20 messages):
{history_str}

Current Trigger:
Aarushi: {incoming_text}
Siddharth:"""

    # Set max_output_tokens=4096 so thinking_level='high' has ample token budget for full multi-line rituals
    proxy_config = types.GenerateContentConfig(
        temperature=0.65,
        max_output_tokens=4096,
        top_p=0.95,
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_level='high')
    )

    print(f"[Gemini Node] Generating draft response with {MODEL_NAME} (4096 tokens headroom)...")
    draft = generate_with_key_rotation(LIVE_INPUT, proxy_config)
    return {"draft_text": draft, "status": "pending"}
