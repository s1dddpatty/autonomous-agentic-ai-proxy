import random
from typing import Optional, List

# Temporary 503 / High Demand local stall drafts
OFFLINE_STALL_DRAFTS: List[str] = [
    "Arre yaha Wi-Fi achanak hagg raha hai, 2 min de text karta hu",
    "Network bohot ajeeb chal raha hai ek sec... message gaya kya?",
    "Arre mom ne kuch kaam bol diya tha achanak, 5 min de free hoke proper text karta hu"
]

# Long-term 429 Quota Exhausted drafts (using newlines to simulate multiple bubbles)
QUOTA_EXHAUSTED_DRAFTS: List[str] = [
    "heyyy...m rlyy sorryy aarushii...\n"
    "m superr tired...i am going to rest now...\n"
    "ill be back in 13 hours.....\n"
    "chaloo ill go now...buhbyeee...\n"
    "muahh i love youuu ❤️💖🫶💕"
]

def get_random_fallback(error_type: int = 503, exclude: Optional[str] = None) -> str:
    """
    Returns exactly one random context-safe offline draft.
    If error_type == 429, returns a random choice from QUOTA_EXHAUSTED_DRAFTS.
    Otherwise (503 or transient server error), returns from OFFLINE_STALL_DRAFTS.
    If 'exclude' is provided, it avoids returning that same draft if possible.
    """
    # Gracefully handle if caller passed exclude as first positional argument
    if isinstance(error_type, str):
        exclude = error_type
        error_type = 503

    target_pool = QUOTA_EXHAUSTED_DRAFTS if error_type == 429 else OFFLINE_STALL_DRAFTS
    choices = [d for d in target_pool if d != exclude] if exclude else target_pool
    if not choices:
        choices = target_pool
    return random.choice(choices)

def is_stall_draft(text: str) -> bool:
    """
    Returns True if the text matches any of the pre-defined offline stall drafts
    or quota exhausted drafts.
    """
    if not text:
        return False
    t_clean = text.strip()
    return (
        t_clean in OFFLINE_STALL_DRAFTS
        or t_clean in QUOTA_EXHAUSTED_DRAFTS
        or "m rlyy sorryy aarushii" in t_clean
    )
