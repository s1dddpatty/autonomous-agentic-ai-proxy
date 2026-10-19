import sqlite3
import os
from typing import Optional

DB_PATH = "data/chats.db"

def init_db():
    """
    Initializes the database schema including the config table and default settings.
    """
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Configuration table for persistent settings (e.g., mode='training' vs 'autopilot')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Insert default row if not already present
    cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('mode', 'training')")
    
    conn.commit()
    conn.close()

def get_config(key: str, default: Optional[str] = None) -> str:
    """
    Fetches a persistent configuration value by key.
    Defaults to 'training' if key == 'mode' and no row is found.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] is not None:
        return row[0]
    if default is not None:
        return default
    if key == "mode":
        return "training"
    return ""

def set_config(key: str, value: str):
    """
    Sets or updates a persistent configuration value by key.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO config (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    ''', (key, str(value)))
    conn.commit()
    conn.close()

# Initialize tables upon module import
init_db()
