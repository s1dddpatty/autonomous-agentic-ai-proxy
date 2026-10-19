import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "data/chats.db"

def setup_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(role, content, timestamp)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_draft TEXT NOT NULL,
            human_feedback TEXT NOT NULL,
            revised_draft TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_sessions (
            sender_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            incoming_text TEXT NOT NULL,
            telegram_message_id INTEGER,
            superseded_tg_msg_id INTEGER,
            status TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_cards (
            telegram_message_id INTEGER PRIMARY KEY,
            thread_id TEXT NOT NULL,
            sender_id TEXT NOT NULL,
            incoming_text TEXT NOT NULL,
            draft_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('mode', 'training')")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_cards_thread ON pending_cards(thread_id)')
    conn.commit()
    conn.close()

from src.core.database import get_config, set_config, init_db

def save_message(role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()

def get_total_messages_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0

def get_recent_history(limit: int = 40) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We order by ID DESC to get the most recent, then reverse the list 
    # in Python so it reads chronologically (oldest to newest)
    cursor.execute('''
        SELECT role, content FROM messages 
        ORDER BY id DESC LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Reverse to get chronological order
    history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return history

def log_feedback(original_draft: str, human_feedback: str, revised_draft: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback_log (original_draft, human_feedback, revised_draft) VALUES (?, ?, ?)",
        (original_draft, human_feedback, revised_draft)
    )
    conn.commit()
    conn.close()

def get_recent_feedback(limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT original_draft, human_feedback, revised_draft FROM feedback_log 
        ORDER BY id DESC LIMIT ?
    ''', (limit * 2,))
    rows = cursor.fetchall()
    conn.close()
    
    clean_feedback = []
    for row in rows:
        orig, fb, rev = row[0] or "", row[1] or "", row[2] or ""
        lower_orig, lower_rev = orig.lower(), rev.lower()
        if "sorryy aarushii" in lower_orig or "sorryy aarushii" in lower_rev:
            continue
        if "ill be back in" in lower_orig or "ill be back in" in lower_rev:
            continue
        if orig.startswith("⚠️ [GEMINI ERROR") or rev.startswith("⚠️ [GEMINI ERROR"):
            continue
        clean_feedback.append({"original": orig, "feedback": fb, "revised": rev})
        if len(clean_feedback) >= limit:
            break

    # Reverse to keep chronological learning flow
    return list(reversed(clean_feedback))

import re

def search_past_chats(keyword_string: str, limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Filter out common stop words
    stop_words = {"is", "the", "a", "an", "and", "or", "to", "in", "on", "for", "of", "it", "that", "this", "my", "me", "you", "i", "hai", "kya", "ho", "tu", "aur", "hi", "se", "ko"}
    words = re.findall(r'\w+', keyword_string.lower())
    keywords = [w for w in words if w not in stop_words]
    
    if not keywords:
        conn.close()
        return []
        
    query = "SELECT role, content FROM messages WHERE "
    conditions = []
    params = []
    
    for kw in keywords:
        conditions.append("content LIKE ?")
        params.append(f"%{kw}%")
        
    query += " OR ".join(conditions)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    return [{"role": row[0], "content": row[1]} for row in rows]

def ingest_chat_file(filepath: str):
    """
    Utility function to ingest historical JSON arrays or JSONL files.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.jsonl'):
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if "conversations" in data:
                    for msg in data["conversations"]:
                        role_map = {"human": "Aarushi", "gpt": "Siddharth", "system": "System"}
                        role = role_map.get(msg.get("from"), msg.get("from"))
                        content = msg.get("value", "")
                        if role != "System" and content:
                            cursor.execute("INSERT OR IGNORE INTO messages (role, content) VALUES (?, ?)", (role, content))
                            count += 1
                else:
                    role = data.get("role", data.get("from"))
                    content = data.get("content", data.get("value", ""))
                    if role and content:
                        cursor.execute("INSERT OR IGNORE INTO messages (role, content) VALUES (?, ?)", (role, content))
                        count += 1
        else:
            data = json.load(f)
            for msg in data:
                role = msg.get("role")
                content = msg.get("content", "")
                if role and content:
                    cursor.execute("INSERT OR IGNORE INTO messages (role, content) VALUES (?, ?)", (role, content))
                    count += 1
                    
    conn.commit()
    conn.close()
    print(f"✅ Successfully ingested {count} messages from {filepath} into the memory database.")

def get_pending_session(sender_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sender_id, thread_id, incoming_text, telegram_message_id, superseded_tg_msg_id, status FROM pending_sessions WHERE sender_id = ?", (sender_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "sender_id": row[0],
        "thread_id": row[1],
        "incoming_text": row[2],
        "telegram_message_id": row[3],
        "superseded_tg_msg_id": row[4],
        "status": row[5]
    }

def get_pending_session_by_thread(thread_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sender_id, thread_id, incoming_text, telegram_message_id, superseded_tg_msg_id, status FROM pending_sessions WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "sender_id": row[0],
        "thread_id": row[1],
        "incoming_text": row[2],
        "telegram_message_id": row[3],
        "superseded_tg_msg_id": row[4],
        "status": row[5]
    }

def set_pending_session(sender_id: str, thread_id: str, incoming_text: str, status: str, telegram_message_id=None, superseded_tg_msg_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO pending_sessions (sender_id, thread_id, incoming_text, telegram_message_id, superseded_tg_msg_id, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(sender_id) DO UPDATE SET
            thread_id=excluded.thread_id,
            incoming_text=excluded.incoming_text,
            telegram_message_id=excluded.telegram_message_id,
            superseded_tg_msg_id=excluded.superseded_tg_msg_id,
            status=excluded.status,
            updated_at=CURRENT_TIMESTAMP
    ''', (sender_id, thread_id, incoming_text, telegram_message_id, superseded_tg_msg_id, status))
    conn.commit()
    conn.close()

def update_pending_telegram_msg_id(sender_id: str, telegram_message_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE pending_sessions 
        SET telegram_message_id = ?, status = 'pending_approval', updated_at = CURRENT_TIMESTAMP
        WHERE sender_id = ?
    ''', (telegram_message_id, sender_id))
    conn.commit()
    conn.close()

def pop_superseded_tg_msg_id(sender_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT superseded_tg_msg_id FROM pending_sessions WHERE sender_id = ?", (sender_id,))
    row = cursor.fetchone()
    old_id = row[0] if row else None
    if old_id:
        cursor.execute("UPDATE pending_sessions SET superseded_tg_msg_id = NULL WHERE sender_id = ?", (sender_id,))
        conn.commit()
    conn.close()
    return old_id

def delete_pending_session(sender_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pending_sessions WHERE sender_id = ?", (sender_id,))
    conn.commit()
    conn.close()

# -------------------------------------------------------------
# INDEPENDENT TELEGRAM CARD STORAGE (keyed by telegram_message_id)
# -------------------------------------------------------------
def save_pending_card(telegram_message_id: int, thread_id: str, sender_id: str, incoming_text: str, draft_text: str, status: str = "pending"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO pending_cards (telegram_message_id, thread_id, sender_id, incoming_text, draft_text, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(telegram_message_id) DO UPDATE SET
            thread_id=excluded.thread_id,
            incoming_text=excluded.incoming_text,
            draft_text=excluded.draft_text,
            status=excluded.status,
            updated_at=CURRENT_TIMESTAMP
    ''', (telegram_message_id, thread_id, sender_id, incoming_text, draft_text, status))
    conn.commit()
    conn.close()

def get_pending_card(telegram_message_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT telegram_message_id, thread_id, sender_id, incoming_text, draft_text, status FROM pending_cards WHERE telegram_message_id = ?', (telegram_message_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "telegram_message_id": row[0],
        "thread_id": row[1],
        "sender_id": row[2],
        "incoming_text": row[3],
        "draft_text": row[4],
        "status": row[5]
    }

def get_pending_card_by_thread(thread_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT telegram_message_id, thread_id, sender_id, incoming_text, draft_text, status FROM pending_cards WHERE thread_id = ? ORDER BY telegram_message_id DESC LIMIT 1', (thread_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "telegram_message_id": row[0],
        "thread_id": row[1],
        "sender_id": row[2],
        "incoming_text": row[3],
        "draft_text": row[4],
        "status": row[5]
    }

def update_pending_card_status(telegram_message_id: int, status: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE pending_cards SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_message_id = ?', (status, telegram_message_id))
    conn.commit()
    conn.close()

def update_pending_card_draft(telegram_message_id: int, draft_text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE pending_cards SET draft_text = ?, status = "pending", updated_at = CURRENT_TIMESTAMP WHERE telegram_message_id = ?', (draft_text, telegram_message_id))
    conn.commit()
    conn.close()

def delete_pending_card(telegram_message_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pending_cards WHERE telegram_message_id = ?', (telegram_message_id,))
    conn.commit()
    conn.close()

def get_latest_pending_card():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT telegram_message_id, thread_id, sender_id, incoming_text, draft_text, status FROM pending_cards WHERE status = "pending" ORDER BY telegram_message_id DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "telegram_message_id": row[0],
        "thread_id": row[1],
        "sender_id": row[2],
        "incoming_text": row[3],
        "draft_text": row[4],
        "status": row[5]
    }

# Ensure DB is created on import
setup_db()
