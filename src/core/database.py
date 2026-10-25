import sqlite3
import os
from typing import Optional
from src.core.config import DB_PATH, DATA_DIR

def init_db():
    """
    Initializes the database schema including the config table and default settings.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
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
    
    # 2. SQLite pending message debounce buffer
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_messages_chat ON pending_messages(chat_id)')
    
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

def add_pending_message(chat_id: str, text: str):
    """
    Saves an incoming message directly into the SQLite pending debounce buffer.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO pending_messages (chat_id, text) VALUES (?, ?)",
        (str(chat_id), text)
    )
    conn.commit()
    conn.close()

def pop_pending_messages(chat_id: str) -> list[str]:
    """
    Atomically retrieves and removes all buffered pending messages for a chat_id in order of arrival.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, text FROM pending_messages WHERE chat_id = ? ORDER BY id ASC",
        (str(chat_id),)
    )
    rows = cursor.fetchall()
    if rows:
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(f"DELETE FROM pending_messages WHERE id IN ({placeholders})", ids)
        conn.commit()
    conn.close()
    return [r[1] for r in rows]

def get_pending_messages(chat_id: str) -> list[str]:
    """
    Retrieves all buffered pending messages for a chat_id without removing them.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text FROM pending_messages WHERE chat_id = ? ORDER BY id ASC",
        (str(chat_id),)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def clear_pending_messages(chat_id: str):
    """
    Clears all buffered pending messages for a chat_id.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pending_messages WHERE chat_id = ?", (str(chat_id),))
    conn.commit()
    conn.close()

# Initialize tables upon module import
init_db()

