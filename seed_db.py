import os
from src.core.memory_manager import ingest_chat_file
from src.core.vector_memory import index_chats_to_chroma, retrieve_dual_context

def main():
    print("==================================================")
    print("🚀 SEEDING REPOSITORY DATA (SQLite & Mood-Aware ChromaDB)")
    print("==================================================")

    chats_dir = "chats"
    os.makedirs(chats_dir, exist_ok=True)
    
    # 1. Ingest any JSON/JSONL files in chats/ into SQLite
    for root, _, files in os.walk(chats_dir):
        for file in files:
            if file.endswith('.jsonl') or file.endswith('.json'):
                filepath = os.path.join(root, file)
                print(f"[SQLite] Ingesting {filepath}...")
                ingest_chat_file(filepath)
        
    # 2. Bulk ingest root dataset.jsonl into SQLite
    if os.path.exists("dataset.jsonl"):
        print("[SQLite] Ingesting root dataset.jsonl into SQLite...")
        ingest_chat_file("dataset.jsonl")

    # 3. Bulk group and index Trigger -> Burst pairs into Dual ChromaDB Collections
    print("\n--------------------------------------------------")
    print("🧠 Ingesting and Embedding Burst Pairs into Dual ChromaDB Collections...")
    print("--------------------------------------------------")
    index_chats_to_chroma()

    # 4. Inject fine-grained core facts from Memories.md
    print("\n--------------------------------------------------")
    print("💎 Ingesting and Embedding Core Memories (Memories.md)...")
    print("--------------------------------------------------")
    from scripts.inject_core_facts import inject_memories
    inject_memories()

    # 5. Restore learned style corrections into SQLite feedback_log
    print("\n--------------------------------------------------")
    print("🧠 Restoring Learned Style Corrections (feedback_log)...")
    print("--------------------------------------------------")
    from scripts.restore_feedback import restore_feedback
    restore_feedback()

    print("\n✅ All databases seeded successfully!")

if __name__ == "__main__":
    main()
