import sqlite3
import os
import sys

def main():
    limit = 15
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data/chats.db")

    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 65)
    print(f"📱 RECENT WHATSAPP CHAT HISTORY (Last {limit} Messages)")
    print("=" * 65)

    cursor.execute("""
        SELECT id, role, content, timestamp 
        FROM messages 
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()

    if not rows:
        print("No messages recorded yet.")
    else:
        for mid, role, content, ts in reversed(rows):
            if role.lower() == "aarushi":
                prefix = f"🌸 [Aarushi | {ts}]"
            else:
                prefix = f"💬 [Siddharth (Dispatched) | {ts}]"
            print(f"\n{prefix} (ID #{mid}):")
            for line in content.split("\n"):
                print(f"   {line}")

    print("\n" + "=" * 65)
    print("🎴 RECENT TELEGRAM APPROVAL CARDS")
    print("=" * 65)

    cursor.execute("""
        SELECT telegram_message_id, status, incoming_text, draft_text, created_at 
        FROM pending_cards 
        ORDER BY created_at DESC LIMIT 5
    """)
    cards = cursor.fetchall()

    if not cards:
        print("No approval cards found.")
    else:
        for t_id, status, incoming, draft, created_at in cards:
            status_icon = "⏳ Pending" if status == "pending" else ("✅ Approved" if status == "approved" else "❌ Discarded")
            print(f"\nCard #{t_id} [{status_icon}] ({created_at}):")
            print(f"   📩 Aarushi: \"{incoming.replace(chr(10), ' / ')}\"")
            first_line_draft = draft.split(chr(10))[0] if draft else ""
            print(f"   🤖 Draft:   \"{first_line_draft}...\"")

    print("\n" + "=" * 65 + "\n")
    conn.close()

if __name__ == "__main__":
    main()
