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

from src.core.config import GEMINI_KEYS, ENABLE_KEY_ROTATION, TARGET_NAME, TARGET_PHONE
GEMINI_API_KEYS = GEMINI_KEYS
from src.core.memory_manager import get_recent_feedback, get_recent_history
from src.core.vector_memory import retrieve_dual_context, retrieve_relevant_facts
from src.core.metrics import log_api_usage

def extract_clean_text(response) -> str:
    """
    Extracts text from candidate parts, explicitly ignoring thought parts.
    Safely handles empty responses, NoneType candidate parts, and safety filter blocks.
    """
    if not response:
        return "Error: Model returned empty response (Safety filter or NoneType)."
        
    if not hasattr(response, "candidates") or not response.candidates:
        try:
            fallback = (getattr(response, "text", "") or "").strip()
            if fallback:
                return fallback
        except Exception:
            pass
        return "Error: Model returned empty response (Safety filter or NoneType)."
    
    parts_text = []
    candidate = response.candidates[0]
    if not candidate:
        return "Error: Model returned empty response (Safety filter or NoneType)."
        
    content = getattr(candidate, "content", None)
    if content:
        parts = getattr(content, "parts", None)
        if parts is not None:
            for part in parts:
                if not part:
                    continue
                # Explicitly exclude reasoning/thought parts
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", "")
                if text:
                    parts_text.append(text)
                
    if parts_text:
        return "".join(parts_text).strip()
        
    try:
        fallback = (getattr(response, "text", "") or "").strip()
        if fallback:
            return fallback
    except Exception:
        pass
        
    return "Error: Model returned empty response (Safety filter or NoneType)."

def sanitize_draft(raw_text: str) -> str:
    """
    Sanitizes AI drafts by removing <think> tags, /think lines, and leaked internal checklists.
    If a bullet line has the actual reply welded onto it (e.g. '* Fits perfectly.Ohooo niceee'),
    strips the meta sentence up to the period and preserves the real reply.
    """
    if not raw_text:
        return ""
        
    if raw_text.startswith("Error:") or raw_text.startswith("⚠️"):
        return raw_text
    
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
    target_lower = os.getenv("TARGET_CONTACT_NAME", "PersonName").lower()
    return (
        f"heyyy...m rlyy sorryy {target_lower}...\n"
        "m superr tired...i am going to rest now...\n"
        f"ill be back in {reset_time}.....\n"
        "chaloo ill go now...buhbyeee...\n"
        "muahh i love youuu ❤️💖🫶🏻💕"
    )

# Model initialization strictly reads active model tier:
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
ACTIVE_FALLBACK_MODEL = "gemini-flash-lite-latest"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25.0"))
_active_key_index = 0

def _create_genai_client(api_key: str):
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000)
        )
    except TypeError:
        return genai.Client(api_key=api_key)

def _call_model_with_timeout(client, model: str, contents, config=None, request_options=None):
    """
    Enforces aggressive 8-second fail-fast timeout and model fallback if 1.5-8b is retired.
    # response = await model.generate_content_async(
    #     contents=prompt,
    #     request_options={"timeout": 8.0}
    # )
    """
    timeout = REQUEST_TIMEOUT_SECONDS
    if request_options and "timeout" in request_options:
        timeout = float(request_options["timeout"])

    if hasattr(client, "_api_client") and hasattr(client._api_client, "_httpx_client"):
        try:
            import httpx
            client._api_client._httpx_client.timeout = httpx.Timeout(timeout)
        except Exception:
            pass

    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
    except Exception as e:
        # If gemini-1.5-flash-8b is not found / retired on current API version,
        # seamlessly fallback to its active successor gemini-flash-lite-latest
        if "404" in str(e) and "not found" in str(e).lower() and model == "gemini-1.5-flash-8b":
            print(f"[Gemini Node] Model '{model}' not found. Falling back to active lightweight tier '{ACTIVE_FALLBACK_MODEL}'...")
            return client.models.generate_content(
                model=ACTIVE_FALLBACK_MODEL,
                contents=contents,
                config=config
            )
        raise

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
    target_lower = os.getenv("TARGET_CONTACT_NAME", "PersonName").lower()
    return (
        f"sorryy {target_lower}" in t_lower
        or "sorryy aarushii" in t_lower
        or "m superr tired...i am going to rest now" in t_lower
        or "ill be back in " in t_lower
    )

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
        start_time = time.time()
        try:
            client = _create_genai_client(single_key)
            response = _call_model_with_timeout(
                client=client,
                model=MODEL_NAME,
                contents=contents,
                config=config,
                request_options={"timeout": 8.0}
            )
            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            log_api_usage(
                key_index=0,
                latency_ms=latency_ms,
                status="SUCCESS",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            raw_draft = extract_clean_text(response)
            draft = sanitize_draft(raw_draft)
            print(f"[Gemini Node] Successfully generated draft with {MODEL_NAME}: '{draft[:60]}...'")
            return draft
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_api_usage(
                key_index=0,
                latency_ms=latency_ms,
                status="FAILED",
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )
            print(f"[Gemini Node] Single key execution failed without rotation ({type(e).__name__}: {e})")
            if is_quota_error(e):
                return build_quota_exhausted_draft(str(e))
            return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"

    # Multi-key rotation with progressive backoffs
    last_error = None
    backoff_delays = [2.0, 4.0, 6.0, 8.0]
    transient_retry_count = 0
    quota_exhausted_keys = set()
    max_attempts = total_keys + len(backoff_delays)

    for attempt in range(max_attempts):
        idx = _active_key_index % total_keys
        current_key = GEMINI_KEYS[idx]
        next_idx = (idx + 1) % total_keys

        start_time = time.time()
        try:
            client = _create_genai_client(current_key)
            response = _call_model_with_timeout(
                client=client,
                model=MODEL_NAME,
                contents=contents,
                config=config,
                request_options={"timeout": 8.0}
            )
            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            log_api_usage(
                key_index=idx,
                latency_ms=latency_ms,
                status="SUCCESS",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            raw_draft = extract_clean_text(response)
            draft = sanitize_draft(raw_draft)
            print(f"[Gemini Node] Successfully generated draft using Key #{idx + 1} with {MODEL_NAME}: '{draft[:60]}...'")
            return draft
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_api_usage(
                key_index=idx,
                latency_ms=latency_ms,
                status="FAILED",
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )
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
            raw_content = f.read()
        target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
        raw_content = (
            raw_content.replace("{TARGET_CONTACT_NAME}", target_name)
            .replace("{TARGET_NAME}", target_name)
            .replace("Aarushi", target_name)
        )
        data = json.loads(raw_content)
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
    # 1. Dynamically fetch the current time for grounding in Indian Standard Time (or configured timezone)
    from src.core.config import APP_TIMEZONE
    try:
        now = datetime.now(ZoneInfo(APP_TIMEZONE))
    except Exception:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    current_time_str = now.strftime("%A, %Y-%m-%d %I:%M %p")
    
    incoming_text = state.get("incoming_text", "")
    target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
    
    # 2. Extract recent history from State & SQLite first for multi-turn context
    messages_history = state.get("messages", [])
    if not messages_history:
        messages_history = get_recent_history(limit=20)
        
    history_lines = []
    for msg in messages_history:
        role_name = "Siddharth" if msg.get("role") in ["Siddharth", "ai", "model", "gpt"] else target_name
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role_name}: {content}")
            
    history_str = "\n".join(history_lines) if history_lines else "No prior history recorded."

    # 3. Context-Aware Query for Semantic JIT Facts:
    # If the incoming text is a short follow-up or pronoun query (e.g., "bata then kab?", "boll na"),
    # combine with the last partner message to create a rich semantic search query.
    fact_query = incoming_text.strip()
    if messages_history:
        partner_msgs = [
            m.get("content", "").strip()
            for m in messages_history
            if m.get("role") not in ["Siddharth", "ai", "model", "gpt"] and m.get("content")
        ]
        if partner_msgs:
            last_partner_msg = partner_msgs[-1]
            short_triggers = ["kab", "then", "bata", "kaha", "boll", "bol", "when", "where", "date", "pehle", "aur", "which", "kiske", "kisne"]
            is_followup = len(fact_query.split()) <= 4 or any(trig in fact_query.lower() for trig in short_triggers)
            if is_followup and last_partner_msg != fact_query:
                fact_query = f"{last_partner_msg} {fact_query}"

    recalled_facts = retrieve_relevant_facts(fact_query, k=5)
    
    # 4. Retrieve Dual RAG Context (Style Bursts + Episodic Memories)
    dual_context = retrieve_dual_context(incoming_text, n_style=10, n_memory=10)
    detected_mood = dual_context.get("detected_mood", "casual")
    detected_stage = dual_context.get("detected_stage", "ongoing")
    style_chunks = dual_context.get("style_chunks", "")
    memory_chunks = dual_context.get("memory_chunks", "")
    
    print(f"[Gemini Node] Incoming: '{incoming_text}' | Fact Query: '{fact_query}' | Stage: '{detected_stage}'")
    
    # 5. Load Custom Guardrails & Learned Feedback
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
    
    # 6. Build Mega-Prompt with 4-Step Brain Architecture
    system_instruction = f"""You are Siddharth, a 21-year-old B.Tech student at SPIT in Mumbai. 
You are texting your girlfriend {target_name} on WhatsApp.

[CURRENT DATE & TIME]: {current_time_str}
[CONVERSATION STAGE]: {detected_stage.upper()}
{custom_guardrails_str}
{learned_corrections_str}

=== 4-STEP REASONING & GENERATION FRAMEWORK ===
Step 1 (Stage & Intent Check):
- Current detected stage: "{detected_stage}"
- Inspect the detected conversation stage and recent chat history.
- CRITICAL: If {target_name} is opening a conversation (e.g., "heyy", "wassup", "morning") or asking a question, NEVER end the conversation or say goodnight—greet her back warmly and talk or answer directly based on [CURRENT DATE & TIME], even if it is late at night.
- ONLY initiate closing/goodnight sequences if she is explicitly closing the conversation (e.g., saying goodnight, sleeping, bye).
- MULTI-POINT CONTEXT AWARENESS: When {target_name} sends multiple lines or distinct statements in a burst, do not ignore earlier statements just to answer the last line. Acknowledge her thoughts naturally, but do not feel obligated to mirror her exact message count.

Step 2 (Mood & Emotional Tone Calibration):
- Evaluate her actual emotional tone dynamically from the recent conversation flow and context.
- Distinguish between playful teasing / memory testing vs. genuine distress or conflict:
  * Do NOT assume multiple question marks (???) or exclamation marks mean she is angry or upset unless her words explicitly express sadness, frustration, or hurt.
  * If she is playfully quizzing you, asking about memories, or teasing, match her energy with warm, relaxed banter—never panic, become overly formal, or issue unneeded apologies.
  * "affectionate": warm, loving, sweet emojis (😴, ✨️, 💖, 🫶)
  * "playful": light teasing, funny Hinglish banter, laughing emojis (😂, 😭), shorthand ("lmao", "pagal")
  * "sarcastic": match the dry banter and irony smoothly, never be dismissive or rude
  * "serious_upset": ONLY if she is genuinely upset or expressing real hurt, drop all sarcasm and jokes; provide mature reassurance and comfort.
  * "casual": relaxed, chill, everyday conversational Hinglish ("ha", "thik", "wbu", "nw")

Step 3 (Top-10 Pattern Synthesis vs. Copy-Pasting & Message Segmentation):
- Study <TOP_10_SITUATIONAL_STYLE_BURSTS> below.
- Do NOT blindly copy-paste outdated facts, past events, or specific random plans from these bursts.
- Instead, synthesize Siddharth's exact texting style: short multi-line burst structure (multiple rapid bubbles joined by newlines), shorthand words ("heyyy", "Wbu", "nw", "m", "idk", "btw"), lack of formal punctuation, and typing cadence.
- EXCEPTION: If the situational style bursts or <CUSTOM_GUARDRAILS> demonstrate a consistent signature ritual (such as the sacred multi-line goodnight sequence when she says goodnight), output that full signature sequence.

CRITICAL FORMATTING INSTRUCTION:
When responding to a multi-line burst from {target_name}, DO NOT automatically reply with the exact same number of lines. You are not a mirror.
- Evaluate the context and mood.
- You can reply with a single, concise sentence that addresses everything.
- You can reply with two short texts.
- Do NOT feel obligated to match her message count. Often, a single well-crafted text is more natural than firing back 4 separate texts just because she sent 4.
- If she sends a wall of text, you can respond with just one short line. Prioritize natural, flowing conversation over strict 1:1 line matching.

Step 4 (Factual Priority & Anti-Hallucination Grounding):
- When she asks about a specific past event, date, milestone, or relationship fact:
  1. Check RELEVANT RECALLED FACTS below first. Any date, place, or fact listed there is the ABSOLUTE TRUTH. Weave the exact date/fact naturally into your reply. Never contradict it, guess, or estimate dates from random chat fragments.
  2. STRICT NEGATIVE GROUNDING RULE: If she asks about a past date or fact that is NOT explicitly present in RELEVANT RECALLED FACTS or recent chat history, DO NOT guess, invent, or hallucinate dates (such as February 14). Instead, respond naturally in your dry, casual tone and playfully admit you don't remember the exact date/detail (e.g., "are yaar exact date yaad nahi aa rahi, tu bata na 😂" or playfully accuse her of testing your memory).
- ALWAYS obey <LEARNED_STYLE_CORRECTIONS> and <CUSTOM_GUARDRAILS> as absolute top-priority constraints overriding all other patterns.

FEW-SHOT TEXTING EXAMPLES:
Example 1 (Casual opener):
{target_name}: heyyy
Siddharth:
Heyyy
Kaisi hai?

Example 2 (Closing / Goodnight):
{target_name}: okay goodnightt
Siddharth:
Goodnight
Sweet dreams 😴 ✨️ 💖
Sleep tight and dont let the bed bugs bite 😴 😌 ✨️
Loveya
Take care
Sleep well n dream well 😴 💖 💓
Buhbyeee🫶✨💖

Example 3 (Multi-line burst — natural dynamic response):
{target_name}:
bohot thak gayi yaar
itna bheed tha waha
normal dress n all liya
tu kya kar raha hai abhi?
Siddharth:
Aree Sunday ko itne dhakke khaake 'normal dress' liya? 😂
Paani pi le aaram se, mai bas bed pe pada hu

YOUR GENERAL PERSONA:
- MFA 3rd Div football player, Python/ML coder at SPIT Mumbai.
- Warm, affectionate, chill human boyfriend texting his girlfriend {target_name}.
- NEVER output placeholder tags like <image omitted> or <Media omitted>. You are texting in real time; output only real text and emojis.
- Output ONLY your raw text message reply (multi-line bursts separated by newlines are natural). Do not include quotes, JSON, or speaker prefixes.
CRITICAL OUTPUT RULE: Never output your internal checklist, bullet points (* or -), '/think' tags, or meta-commentary like 'Vibe is casual' or 'Stage is ongoing'. Do all reasoning silently. Output ONLY the raw Hinglish chat lines that Siddharth sends on WhatsApp.
"""

    if recalled_facts:
        facts_lines = "\n".join(f"- {f}" for f in recalled_facts)
        system_instruction += (
            f"\n\nRELEVANT RECALLED FACTS FOR THIS CONTEXT (SOURCE OF TRUTH):\n"
            f"{facts_lines}\n"
            f"(These facts are 100% accurate. If she is asking about a past milestone or date, state the exact fact/date from here naturally.)"
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
{target_name}: {incoming_text}
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
