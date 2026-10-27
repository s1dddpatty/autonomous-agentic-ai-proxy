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
from src.core.fallbacks import get_random_fallback

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

# Primary & Fallback model definitions
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-1.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-1.5-flash-8b")
MODEL_NAME = PRIMARY_MODEL
ACTIVE_PRIMARY_MODEL = "gemini-flash-latest"
ACTIVE_FALLBACK_MODEL = "gemini-flash-lite-latest"
REQUEST_TIMEOUT_SECONDS = 8.0
_active_key_index = 0

def _create_genai_client(api_key: str):
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=10000)
        )
    except TypeError:
        return genai.Client(api_key=api_key)

def _call_model_with_timeout(client, model: str, contents, config=None, request_options=None):
    """
    Enforces aggressive 8-second fail-fast timeout and active model fallback if 1.5 versions are retired.
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
        err_msg = str(e).lower()
        # If the model is retired/not found (404), seamlessly map to active equivalent
        if "404" in str(e) and ("not found" in err_msg or "no longer available" in err_msg):
            alt_model = None
            if "8b" in model:
                alt_model = ACTIVE_FALLBACK_MODEL
            elif "flash" in model:
                alt_model = ACTIVE_PRIMARY_MODEL

            if alt_model and alt_model != model:
                print(f"[Gemini Node] Model '{model}' not found on endpoint. Mapping to '{alt_model}'...")
                return client.models.generate_content(
                    model=alt_model,
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

def generate_with_key_rotation(contents, config=None, model: str = PRIMARY_MODEL) -> str:
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
                model=model,
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
            print(f"[Gemini Node] Successfully generated draft with {model}: '{draft[:60]}...'")
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
                model=model,
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
            print(f"[Gemini Node] Successfully generated draft using Key #{idx + 1} with {model}: '{draft[:60]}...'")
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
                print(f"[Gemini Node] 🔑 Key #{idx + 1} hit quota limit! Rotating to Key #{next_idx + 1} on {model}...")
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
                    print(f"[Gemini Node] Server high demand backoff retries exhausted ({len(backoff_delays)} attempts on {model}).")
                    return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"
            else:
                print(f"[Gemini Node] Model {model} failed with non-quota error:\n{traceback.format_exc()}")
                return f"⚠️ [GEMINI ERROR: {type(e).__name__}: {str(e)[:150]}]"

    if len(quota_exhausted_keys) >= total_keys:
        print(f"[Gemini Node] ⚡ All {total_keys} API keys exhausted! Returning in-character rest fallback.")
        return build_quota_exhausted_draft(str(last_error))
    return f"⚠️ [GEMINI ERROR: {type(last_error).__name__ if last_error else 'Unknown'}: {str(last_error)[:150] if last_error else 'Exhausted retries'}]"

def invoke_fallback_8b(contents, config=None, model: str = FALLBACK_MODEL) -> str:
    """
    Cleanly instantiates and invokes the fallback 8b model across available keys.
    """
    global _active_key_index
    if not GEMINI_KEYS:
        raise Exception("No GEMINI_API_KEYS configured")

    total_keys = len(GEMINI_KEYS)
    last_err = None

    for attempt in range(total_keys):
        idx = (_active_key_index + attempt) % total_keys
        key = GEMINI_KEYS[idx]
        start_time = time.time()
        try:
            client = _create_genai_client(key)
            response = _call_model_with_timeout(
                client=client,
                model=model,
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
            print(f"[Gemini Node] Successfully generated draft using fallback {model} on Key #{idx + 1}: '{draft[:60]}...'")
            return draft
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_api_usage(
                key_index=idx,
                latency_ms=latency_ms,
                status="FAILED",
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )
            print(f"[Gemini Node] Fallback model {model} attempt failed on Key #{idx + 1}: {e}")
            last_err = e
            continue

    raise Exception(f"All keys exhausted for fallback model {model} (last error: {last_err})")

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
    Drafting node: generates WhatsApp responses with strict tone constraints,
    contextualized RAG from Memories.md, and safe High-to-Low model fallback
    (gemini-1.5-flash -> gemini-1.5-flash-8b).
    """
    incoming_text = state.get("incoming_text", "")
    target_name = os.getenv("TARGET_CONTACT_NAME", TARGET_NAME or "PersonName")

    # 1. Ingest vibe, standalone query, and rag_context from Analyzer
    vibe = state.get("vibe")
    rag_context = state.get("rag_context")
    if not vibe or rag_context is None:
        from src.core.analyzer_node import analyzer_node
        analysis = analyzer_node(state)
        vibe = analysis.get("vibe", "dry, casual, and natural")
        rag_context = analysis.get("rag_context", "No relevant memories found in Memories.md.")
        state["vibe"] = vibe
        state["standalone_search_query"] = analysis.get("standalone_search_query", "")
        state["rag_context"] = rag_context

    # 2. Strict Generation Node Prompting (Killing "Cringe" & Forcing Facts)
    # Exact f-string architecture as required:
    system_prompt = f"""You are texting your girlfriend. Your default tone is dry, casual, and natural.

CRITICAL TONE RULES:
1. NO CRINGE: Never use poetic language, forced punchlines, or dramatic flair.
2. EMOJI LIMIT: Maximum one emoji per message.
3. CURRENT VIBE: Match this specific energy for this message -> {vibe}

CRITICAL GROUNDING RULE: If the user's message is vague (e.g., 'boll', 'kab') and the retrieved context does not contain a specific, explicitly matching date or fact, YOU MUST NOT hallucinate, guess, or invent dates (such as February 14). Instead, respond naturally in the established tone asking for clarification (e.g., 'kis baare mein?', 'kya bolu?').

FACTUAL OVERRIDE:
If she asks a question about a past event, memory, or specific fact:
- You MUST playfully accuse her of "testing" you.
- You MUST explicitly state the exact answer retrieved from the CONTEXT below in the very same message.
- NEVER dodge the question to make a joke. NEVER guess, hallucinate, or assume facts.

CONTEXT FROM MEMORIES.MD:
{rag_context}
"""

    # 3. Format recent conversation history
    messages_history = state.get("messages", [])
    if not messages_history:
        messages_history = get_recent_history(limit=20)

    history_lines = []
    for msg in messages_history:
        role = msg.get("role", "")
        role_name = "Siddharth" if role in ["Siddharth", "ai", "model", "gpt"] else target_name
        content = (msg.get("content") or msg.get("text") or "").strip()
        if content:
            history_lines.append(f"{role_name}: {content}")

    history_str = "\n".join(history_lines) if history_lines else "No prior history recorded."

    prompt_content = f"""Recent Chat History:
{history_str}

Current Trigger:
{target_name}: {incoming_text}
Siddharth:"""

    generation_config = types.GenerateContentConfig(
        temperature=0.65,
        max_output_tokens=2048,
        top_p=0.95,
        system_instruction=system_prompt,
    )

    # 4. Safe High-to-Low Model Fallback Execution
    draft = None
    is_fallback = False

    try:
        # Primary Model generation with 5-key rotation mechanism
        print(f"[Drafting Node] Attempting generation with primary model ({PRIMARY_MODEL}) via 5-key rotation...")
        draft = generate_with_key_rotation(prompt_content, generation_config, model=PRIMARY_MODEL)
        if not draft or draft.startswith("⚠️ [GEMINI ERROR:") or is_quota_fallback_text(draft) or draft.startswith("Error:"):
            raise Exception(f"Primary model {PRIMARY_MODEL} exhausted or failed: {draft}")
        is_fallback = False
    except Exception as primary_error:
        print(f"[Drafting Node] ⚠️ Primary model ({PRIMARY_MODEL}) failed: {primary_error}")
        print(f"[Drafting Node] 🔄 Executing Safe Fallback: cleanly instantiating and invoking {FALLBACK_MODEL}...")
        try:
            # Fallback Model: gemini-1.5-flash-8b
            draft = invoke_fallback_8b(prompt_content, generation_config, model=FALLBACK_MODEL)
            if not draft or draft.startswith("⚠️ [GEMINI ERROR:") or draft.startswith("Error:"):
                raise Exception(f"Fallback model {FALLBACK_MODEL} failed: {draft}")
            is_fallback = True
            print(f"[Drafting Node] ✅ Fallback model {FALLBACK_MODEL} generated draft: '{draft[:60]}...'")
        except Exception as fallback_error:
            print(f"[Drafting Node] ❌ Fallback model also failed ({fallback_error}). Triggering safe local excuse.")
            draft = get_random_fallback(error_type=503)
            is_fallback = True

    return {
        "draft_text": draft,
        "status": "pending",
        "is_fallback": is_fallback,
        "vibe": vibe,
        "rag_context": rag_context
    }

drafting_node = gemini_proxy_node
