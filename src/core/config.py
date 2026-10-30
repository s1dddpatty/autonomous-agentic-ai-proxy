import os
import re
from typing import List
from dotenv import load_dotenv

# Call load_dotenv() at the top of the file
load_dotenv()

# --- PERSISTENT VOLUME & STORAGE PATHS ---
# Base project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Cloud servers will inject DATA_DIR (e.g., /mnt/data). Locally, it defaults to the relative "data" folder.
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Update subsequent paths to use this base:
DB_PATH = os.path.join(DATA_DIR, "chats.db")
METRICS_PATH = os.path.join(DATA_DIR, "api_analytics.jsonl")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(DATA_DIR, "chroma_db"))
CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "checkpoint.sqlite")

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

# --- CONTACT REGISTRY & MULTI-USER CONFIGURATION ---
TARGET_CONTACT_NAME: str = (
    os.getenv("TARGET_CONTACT_NAME") or "Aarushi"
).strip().strip("'\"")
TARGET_NAME: str = TARGET_CONTACT_NAME

# Known contacts
AARUSHI_PHONE_NUMBER: str = (
    os.getenv("AARUSHI_PHONE_NUMBER") or "919330959944"
).strip().strip("'\"")

TEST_PHONE_NUMBER: str = (
    os.getenv("TEST_PHONE_NUMBER") or os.getenv("TARGET_PHONE_NUMBER") or "917208337457"
).strip().strip("'\"")
TARGET_PHONE_NUMBER: str = TEST_PHONE_NUMBER
TARGET_PHONE: str = TARGET_PHONE_NUMBER

# --- TIMEZONE CONFIGURATION ---
APP_TIMEZONE: str = os.getenv("APP_TIMEZONE", "Asia/Kolkata").strip().strip("'\"")

# --- WHATSAPP CONFIGURATION & MULTI-TOKEN ROUTING ---
WHATSAPP_ACCESS_TOKEN: str = (
    os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN") or ""
).strip().strip("'\"")

# Optional per-contact tokens for modularity
WHATSAPP_ACCESS_TOKEN_AARUSHI: str = (
    os.getenv("WHATSAPP_ACCESS_TOKEN_AARUSHI") or os.getenv("WHATSAPP_TOKEN_AARUSHI") or ""
).strip().strip("'\"")

WHATSAPP_ACCESS_TOKEN_TEST: str = (
    os.getenv("WHATSAPP_ACCESS_TOKEN_TEST") or os.getenv("WHATSAPP_TOKEN_TEST") or ""
).strip().strip("'\"")

WHATSAPP_PHONE_ID: str = (
    os.getenv("WHATSAPP_PHONE_ID") or os.getenv("PHONE_NUMBER_ID") or ""
).strip().strip("'\"")

def normalize_phone(num: str) -> str:
    """Strips non-digit characters and standardizes country code prefix."""
    if not num:
        return ""
    digits = re.sub(r"\D", "", str(num))
    if digits.startswith("0"):
        digits = digits[1:]
    return digits

def get_contact_display_name(phone_number: str) -> str:
    """Returns human-friendly sender label for Telegram HITL card."""
    norm = normalize_phone(phone_number)
    aarushi_norm = normalize_phone(AARUSHI_PHONE_NUMBER)
    test_norm = normalize_phone(TEST_PHONE_NUMBER)

    if norm and (norm == aarushi_norm or norm.endswith(aarushi_norm[-10:])):
        return f"Aarushi (+{norm})"
    elif norm and (norm == test_norm or norm.endswith(test_norm[-10:])):
        return f"Siddharth / Test (+{norm})"
    elif phone_number:
        return f"+{norm}" if norm else str(phone_number)
    return "Unknown Contact"

def get_whatsapp_token_for_number(phone_number: str) -> str:
    """Selects specific access token if configured for this contact, else falls back to primary."""
    norm = normalize_phone(phone_number)
    aarushi_norm = normalize_phone(AARUSHI_PHONE_NUMBER)
    test_norm = normalize_phone(TEST_PHONE_NUMBER)

    if norm and (norm == aarushi_norm or norm.endswith(aarushi_norm[-10:])) and WHATSAPP_ACCESS_TOKEN_AARUSHI:
        return WHATSAPP_ACCESS_TOKEN_AARUSHI
    if norm and (norm == test_norm or norm.endswith(test_norm[-10:])) and WHATSAPP_ACCESS_TOKEN_TEST:
        return WHATSAPP_ACCESS_TOKEN_TEST
    return WHATSAPP_ACCESS_TOKEN

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
