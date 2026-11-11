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
from src.core.memory_manager import get_recent_feedback, get_recent_history, get_recent_cards
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
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
ACTIVE_FALLBACK_MODEL = "gemini-3.5-flash-lite"
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

def _call_single_model_with_thinking_fallback(client, target_model: str, contents, config=None):
    """
    Calls a single Gemini model with optional thinking_config; if thinking_config triggers
    an empty/malformed response or 503 spike on that tier, retries cleanly without thinking_config.
    """
    has_thinking = bool(config and getattr(config, "thinking_config", None))
    thinking_modes = [True, False] if has_thinking else [False]
    last_error = None

    for use_thinking in thinking_modes:
        call_config = config
        if not use_thinking and has_thinking:
            import copy
            try:
                call_config = copy.copy(config)
                call_config.thinking_config = None
            except Exception:
                call_config = config

        try:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=call_config
            )
            raw = extract_clean_text(response)
            if raw and not raw.startswith("Error:"):
                return response
            print(f"[Gemini Node] Model '{target_model}' (thinking={use_thinking}) returned empty/malformed: '{raw[:60]}'.")
            last_error = Exception(f"Model {target_model} returned empty/malformed response: {raw}")
        except Exception as e:
            print(f"[Gemini Node] Model '{target_model}' (thinking={use_thinking}) call failed: {e}.")
            last_error = e
            # If key quota is exhausted (429), retrying same model without thinking won't bypass quota; rotate key immediately
            if is_quota_error(e):
                raise e

    raise last_error if last_error else Exception(f"Model {target_model} failed")


def _call_model_with_timeout(client, model: str, contents, config=None, request_options=None):
    """
    Sets timeout on the client and delegates to `_call_single_model_with_thinking_fallback`.
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

    response = _call_single_model_with_thinking_fallback(client, model, contents, config)
    return response, model

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
    Rotates through ALL configured keys on the primary model (gemini-3.5-flash) BEFORE
    stepping down to lighter fallback tiers (gemini-3.5-flash-lite, gemini-3.1-flash-lite).
    """
    global _active_key_index
    if not GEMINI_KEYS:
        return "⚠️ [GEMINI ERROR: No GEMINI_API_KEYS configured]"

    total_keys = len(GEMINI_KEYS)
    candidate_models = []
    for m in [MODEL_NAME, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    # Single-key / Rotation disabled bypass: try model cascade directly on the single key
    if not ENABLE_KEY_ROTATION or total_keys == 1:
        single_key = GEMINI_KEYS[0]
        last_error = None
        for target_model in candidate_models:
            start_time = time.time()
            try:
                client = _create_genai_client(single_key)
                response, used_model = _call_model_with_timeout(
                    client=client,
                    model=target_model,
                    contents=contents,
                    config=config,
                    request_options={"timeout": 8.0}
                )
                latency_ms = (time.time() - start_time) * 1000
                prompt_tokens = getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0) or 0
                completion_tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0) or 0
                log_api_usage(0, latency_ms, "SUCCESS", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                raw_draft = extract_clean_text(response)
                draft = sanitize_draft(raw_draft)
                if not draft or draft.startswith("Error:") or draft.startswith("⚠️"):
                    raise Exception(f"Model returned invalid draft: {draft}")
                print(f"[Gemini Node] Successfully generated draft with {used_model}: '{draft[:60]}...'")
                return draft
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                log_api_usage(0, latency_ms, "FAILED", error=f"{type(e).__name__}: {str(e)[:100]}")
                last_error = e
                print(f"[Gemini Node] Single key failed on {target_model} ({type(e).__name__}: {e}).")

        if last_error and is_quota_error(last_error):
            return build_quota_exhausted_draft(str(last_error))
        return f"⚠️ [GEMINI ERROR: {type(last_error).__name__ if last_error else 'Unknown'}: {str(last_error)[:150]}]"

    # Multi-key rotation: exhaust all keys on primary tier before downgrading model tier
    last_error = None
    all_keys_quota_exhausted = False

    for tier_idx, target_model in enumerate(candidate_models):
        if tier_idx > 0:
            print(f"[Gemini Node] ⬇️ All {total_keys} keys failed/exhausted on {candidate_models[tier_idx - 1]}. Falling back to tier '{target_model}'...")

        quota_exhausted_for_tier = set()
        transient_failed_for_tier = set()
        backoff_delays = [1.5, 3.0]
        backoff_round = 0
        max_attempts_for_tier = total_keys * 2 + len(backoff_delays)

        for _ in range(max_attempts_for_tier):
            idx = _active_key_index % total_keys
            current_key = GEMINI_KEYS[idx]
            next_idx = (idx + 1) % total_keys

            start_time = time.time()
            try:
                client = _create_genai_client(current_key)
                response, used_model = _call_model_with_timeout(
                    client=client,
                    model=target_model,
                    contents=contents,
                    config=config,
                    request_options={"timeout": 8.0}
                )
                latency_ms = (time.time() - start_time) * 1000
                prompt_tokens = getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0) or 0
                completion_tokens = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0) or 0
                log_api_usage(idx, latency_ms, "SUCCESS", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                raw_draft = extract_clean_text(response)
                draft = sanitize_draft(raw_draft)
                if not draft or draft.startswith("Error:") or draft.startswith("⚠️"):
                    raise Exception(f"Model returned invalid draft: {draft}")
                print(f"[Gemini Node] Successfully generated draft using Key #{idx + 1} with {used_model}: '{draft[:60]}...'")
                return draft
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                log_api_usage(idx, latency_ms, "FAILED", error=f"{type(e).__name__}: {str(e)[:100]}")
                last_error = e
                _active_key_index = next_idx

                if is_quota_error(e):
                    quota_exhausted_for_tier.add(idx)
                    print(f"[Gemini Node] 🔑 Key #{idx + 1} hit quota limit on {target_model}! Rotating to Key #{next_idx + 1}...")
                    if len(quota_exhausted_for_tier) >= total_keys:
                        all_keys_quota_exhausted = True
                        break
                    continue
                elif is_transient_server_error(e):
                    transient_failed_for_tier.add(idx)
                    if len(quota_exhausted_for_tier | transient_failed_for_tier) < total_keys:
                        # Immediately try remaining fresh keys on this tier without sleeping
                        print(f"[Gemini Node] ⏳ {target_model} 503/transient spike on Key #{idx + 1}. Rotating immediately to Key #{next_idx + 1}...")
                        continue
                    if backoff_round < len(backoff_delays):
                        delay = backoff_delays[backoff_round]
                        backoff_round += 1
                        transient_failed_for_tier.clear()
                        print(f"[Gemini Node] ⏳ All non-exhausted keys hit 503 on {target_model}. Backing off {delay}s before retry round...")
                        time.sleep(delay)
                        continue
                    print(f"[Gemini Node] {target_model} transient retries exhausted across all {total_keys} keys.")
                    break
                else:
                    transient_failed_for_tier.add(idx)
                    print(f"[Gemini Node] Key #{idx + 1} failed on {target_model} ({e}). Rotating to Key #{next_idx + 1}...")
                    if len(quota_exhausted_for_tier | transient_failed_for_tier) >= total_keys:
                        break
                    continue

    if all_keys_quota_exhausted and is_quota_error(last_error):
        print(f"[Gemini Node] ⚡ All {total_keys} API keys exhausted across tiers! Returning in-character rest fallback.")
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
    
    print(f"[Gemini Node] Incoming: '{incoming_text}' | Fact Query: '{fact_query}' | Stage: '{detected_stage}' | Mood: '{detected_mood}'")
    
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

    mood_directives = {
        "casual": (
            "CASUAL & CHILL: Reply like an effortless, relaxed everyday text. Keep it short, dry/natural Hinglish, "
            "and closely match the brevity and tone of <TOP_10_SITUATIONAL_STYLE_BURSTS>. ZERO cheesy/sweet wrap-up lines, "
            "and NO forced emojis (use 0 emojis unless the matching style bursts clearly use one)."
        ),
        "playful": (
            "PLAYFUL BANTER: Light teasing or natural funny Hinglish banter matching <TOP_10_SITUATIONAL_STYLE_BURSTS>. "
            "Stay chill and concise—do NOT add corny romantic lines or spam emojis."
        ),
        "sarcastic": (
            "SARCASTIC / DRY BANTER: Match her dry irony smoothly in 1–2 short lines. No defensive apologies or lovey emojis."
        ),
        "affectionate": (
            "AFFECTIONATE: Warm and genuine in Siddharth's real voice from <TOP_10_SITUATIONAL_STYLE_BURSTS>. "
            "Keep it grounded and real—avoid generic Bollywood/Hallmark clichés."
        ),
        "serious_upset": (
            "SERIOUS / UPSET: Drop sarcasm and banter completely; give calm, mature reassurance."
        ),
    }
    active_mood_directive = mood_directives.get(detected_mood, mood_directives["casual"])
    
    # 6. Build Mega-Prompt with 4-Step Brain Architecture
    system_instruction = f"""You are Siddharth, a 21-year-old B.Tech student at SPIT in Mumbai. 
You are texting {target_name} on WhatsApp.

[CURRENT DATE & TIME]: {current_time_str}
[CONVERSATION STAGE]: {detected_stage.upper()}
[DETECTED MOOD]: {detected_mood.upper()}
{custom_guardrails_str}
{learned_corrections_str}

=== 4-STEP REASONING & GENERATION FRAMEWORK ===
Step 1 (Stage & Intent Check):
- Current detected stage: "{detected_stage}"
- CRITICAL: If {target_name} is opening a conversation or talking casually, NEVER end the conversation or say goodnight—talk naturally based on [CURRENT DATE & TIME].
- ONLY initiate closing/goodnight sequences if she is explicitly closing the conversation (e.g., saying goodnight, sleeping, bye).
- MULTI-POINT CONTEXT AWARENESS: When {target_name} sends multiple lines in a burst, acknowledge her thoughts naturally without feeling forced to mirror her exact message count.

Step 2 (Mood & Emotional Tone Calibration — HIGHEST PRIORITY):
- Current active mood target: {active_mood_directive}
- Also check Recent Chat History to confirm context flow: do NOT assume multiple question marks (???) mean she is angry unless her words explicitly express hurt or anger.

Step 3 (Top-10 Real Style Grounding & Anti-Cliché Rule):
- Study <TOP_10_SITUATIONAL_STYLE_BURSTS> closely—THAT is how Siddharth actually texts.
- Synthesize his exact raw cadence: short casual bubbles, lowercase/minimal punctuation, everyday shorthand ("ha", "m", "nw", "wbu", "re", "baba", "idk", "btw").
- STRICT ANTI-CLICHÉ & EMOJI RESTRAINT RULE:
  * NEVER tack on generic cheesy wrap-up lines (e.g., "tu hai na mere sath bas wahi kaafi hai ❤️", "ab aaram se chill kar 💖", "hamesha tere sath hu") when a short chill reply already answers her!
  * Emojis are OPTIONAL, NOT mandatory. In casual/ongoing chat, plain text with ZERO emojis is often the most natural.
- EXCEPTION: ONLY if she is explicitly saying goodnight/closing, use the signature goodnight sequence from <CUSTOM_GUARDRAILS>.

CRITICAL FORMATTING INSTRUCTION:
When responding to a multi-line burst from {target_name}, DO NOT automatically reply with the exact same number of lines or bloat your answer with extra filler lines. Often 1 concise, chill line is far more natural than 2–3 lines.

Step 4 (Factual Priority & Anti-Hallucination Grounding):
- When she asks about a specific past event, date, milestone, or relationship fact:
  1. Check RELEVANT RECALLED FACTS below first. Any date, place, or fact listed there is the ABSOLUTE TRUTH.
  2. STRICT NEGATIVE GROUNDING RULE: If she asks about a past date or fact NOT explicitly in RELEVANT RECALLED FACTS or recent chat history, DO NOT invent dates—casually admit you forgot the exact date/detail (e.g., "are yaar exact date yaad nahi aa rahi, tu bata na").
- ALWAYS obey <LEARNED_STYLE_CORRECTIONS> and <CUSTOM_GUARDRAILS> as top-priority constraints.

FEW-SHOT TEXTING EXAMPLES:
Example 1 (Casual opener):
{target_name}: heyyy
Siddharth:
heyyy
kaisi hai?

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

Example 3 (Chill multi-line burst — zero forced emojis/clichés):
{target_name}:
bohot thak gayi yaar
itna bheed tha waha
normal dress n all liya
tu kya kar raha hai abhi?
Siddharth:
itna rush tha sunday ko?
m bas ghar pe hu, paani pi le aaram se

YOUR GENERAL PERSONA:
- MFA 3rd Div football player, Python/ML coder at SPIT Mumbai.
- Chill, effortless, authentic voice texting {target_name}—match the real style in <TOP_10_SITUATIONAL_STYLE_BURSTS> over generic AI sweetness.
- NEVER output placeholder tags like <image omitted> or <Media omitted>.
- Output ONLY your raw WhatsApp reply lines (newlines separate separate bubbles). No quotes, JSON, speaker labels, bullet points, or '/think' reasoning.
"""

    if recalled_facts:
        facts_lines = "\n".join(f"- {f}" for f in recalled_facts)
        system_instruction += (
            f"\n\nRELEVANT RECALLED FACTS FOR THIS CONTEXT (SOURCE OF TRUTH):\n"
            f"{facts_lines}\n"
            f"(These facts are 100% accurate. If she is asking about a past milestone or date, state the exact fact/date from here naturally.)"
        )

    # 7. Card & Dispatch Memory (Semantic Anti-Repetition Brain)
    recent_cards = get_recent_cards(sender_id=state.get("sender_id", ""), minutes=30, limit=5)
    recent_cards_block = ""
    anti_repeat_reminder = ""
    if recent_cards:
        print(f"[Gemini Node] Anti-repetition brain loaded {len(recent_cards)} recent card(s) from the last 30m.")
        card_entries = []
        for c in recent_cards:
            status_tag = "ALREADY SENT" if c.get("status") == "approved" else "PENDING APPROVAL ON TELEGRAM"
            card_entries.append(
                f"- [{status_tag}] Her msg: \"{c.get('incoming_text', '')[:120]}\" -> Your drafted/sent reply: \"{c.get('draft_text', '')}\""
            )
        recent_cards_str = "\n".join(card_entries)
        recent_cards_block = f"\n<RECENTLY_DRAFTED_OR_SENT_CARDS>\n{recent_cards_str}\n</RECENTLY_DRAFTED_OR_SENT_CARDS>\n"
        anti_repeat_reminder = (
            "(IMPORTANT: Do NOT re-answer her earlier messages or repeat ANY line, activity, or meaning "
            "already covered in <RECENTLY_DRAFTED_OR_SENT_CARDS> or Recent Chat History above. Reply ONLY to what is NEW below.)\n"
        )
        system_instruction += (
            "\n\nANTI-REPETITION & SEMANTIC DEDUPLICATION RULE (CRITICAL):\n"
            "- Carefully inspect <RECENTLY_DRAFTED_OR_SENT_CARDS> and Recent Chat History before writing your reply.\n"
            "- Every draft listed in <RECENTLY_DRAFTED_OR_SENT_CARDS> (whether PENDING APPROVAL ON TELEGRAM or ALREADY SENT) means you have ALREADY handled that point.\n"
            "- NEVER say the same thing OR convey the same meaning twice across back-to-back cards—even using different Hinglish words!\n"
            "  * Example 1: If a recent card already said what you're doing (e.g., 'bed pe chill karrau', 'tera hi wait kar raha tha', 'YouTube scroll kar raha hu'), DO NOT say 'mai bas bed pe chill kar raha hu' or state your activity again in the next card.\n"
            "  * Example 2: If a recent card already asked a check-in question (e.g., 'safe pahunch gayi na ghar?', 'change vagera kar liya?', 'khana khaya?'), DO NOT ask that same question again in the next card.\n"
            "  * Example 3: If a recent card already delivered the full Goodnight ritual, DO NOT repeat the Goodnight sequence again—acknowledge only any new sentiment concisely (e.g., 'Love you moreee ❤️' or 'Haan aaram se so jaunga 😌✨').\n"
            "- Focus strictly on answering ONLY what is NEW and UNANSWERED in Current Trigger."
        )

    LIVE_INPUT = f"""<TOP_10_SITUATIONAL_STYLE_BURSTS>
{style_chunks}
</TOP_10_SITUATIONAL_STYLE_BURSTS>

<RECALLED_EPISODIC_MEMORIES>
{memory_chunks}
</RECALLED_EPISODIC_MEMORIES>
{recent_cards_block}
Recent Chat History (Last 20 messages):
{history_str}

{anti_repeat_reminder}Current Trigger:
{target_name}: {incoming_text}
Siddharth:"""

    # Use LIGHT thinking ('LOW') for casual/playful banter to prevent overthinking & clichés;
    # reserve 'HIGH' thinking for serious/upset conversations or closing rituals.
    active_thinking_level = "HIGH" if (detected_mood == "serious_upset" or detected_stage == "closer") else "LOW"
    proxy_config = types.GenerateContentConfig(
        temperature=0.65,
        max_output_tokens=4096,
        top_p=0.95,
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_level=active_thinking_level)
    )

    print(f"[Gemini Node] Generating draft response with {MODEL_NAME} (thinking={active_thinking_level})...")
    draft = generate_with_key_rotation(LIVE_INPUT, proxy_config)
    return {"draft_text": draft, "status": "pending"}
