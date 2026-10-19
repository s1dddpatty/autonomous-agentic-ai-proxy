import os
from typing import List
from dotenv import load_dotenv

# Call load_dotenv() at the top of the file
load_dotenv()

# --- GEMINI CONFIGURATION ---
def _load_gemini_keys() -> List[str]:
    keys = []
    seen = set()

    # 1. Parse comma-separated string from GEMINI_API_KEYS
    raw_keys = os.getenv("GEMINI_API_KEYS", "")
    if raw_keys:
        for k in raw_keys.split(","):
            cleaned = k.strip().strip("'\"")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                keys.append(cleaned)

    # 2. Check individual key variables for backward compatibility
    if not keys:
        individual_vars = [
            "GEMINI_API_KEY",
            "GEMINI_API_KEY_1",
            "GEMINI_API_KEY_2",
            "GEMINI_API_KEY_3",
            "GEMINI_API_KEY_4",
            "GEMINI_API_KEY_5",
        ]
        for var in individual_vars:
            val = os.getenv(var)
            if val:
                cleaned = val.strip().strip("'\"")
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    keys.append(cleaned)

    return keys

# GEMINI_KEYS list
GEMINI_KEYS: List[str] = _load_gemini_keys()

# ENABLE_KEY_ROTATION (boolean flag, default to True)
ENABLE_KEY_ROTATION: bool = os.getenv("ENABLE_KEY_ROTATION", "True").strip().lower() in ("true", "1", "yes")

# --- WHATSAPP CONFIGURATION ---
WHATSAPP_ACCESS_TOKEN: str = (
    os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN") or ""
).strip().strip("'\"")

WHATSAPP_PHONE_ID: str = (
    os.getenv("WHATSAPP_PHONE_ID") or os.getenv("PHONE_NUMBER_ID") or ""
).strip().strip("'\"")

TARGET_PHONE_NUMBER: str = (
    os.getenv("TARGET_PHONE_NUMBER") or os.getenv("RECIPIENT_PHONE_NUMBER") or ""
).strip().strip("'\"")

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip().strip("'\"")

TELEGRAM_CHAT_ID: str = (
    os.getenv("TELEGRAM_CHAT_ID") or os.getenv("AUTHORIZED_TG_USER_ID") or ""
).strip().strip("'\"")

# --- BACKWARD COMPATIBILITY ALIASES ---
AUTHORIZED_TG_USER_ID: str = TELEGRAM_CHAT_ID
WHATSAPP_TOKEN: str = WHATSAPP_ACCESS_TOKEN
PHONE_NUMBER_ID: str = WHATSAPP_PHONE_ID
GEMINI_API_KEYS: List[str] = GEMINI_KEYS
VERIFY_TOKEN: str = os.getenv("VERIFY_TOKEN", "").strip().strip("'\"")
NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "").strip().strip("'\"")
