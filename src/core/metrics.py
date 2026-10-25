import json
import os
from datetime import datetime
from src.core.config import METRICS_PATH, DATA_DIR

LOG_FILE = METRICS_PATH
MAX_LOG_LINES = 200

def log_api_usage(key_index: int, latency_ms: float, status: str, prompt_tokens: int = 0, completion_tokens: int = 0, error: str = None):
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "key_index": key_index,
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "error_type": error
    }
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Load existing logs
    lines = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    # Append the new log
    lines.append(json.dumps(record) + "\n")
    
    # Enforce the rolling limit
    if len(lines) > MAX_LOG_LINES:
        lines = lines[-MAX_LOG_LINES:]
        
    # Write back the truncated list
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
