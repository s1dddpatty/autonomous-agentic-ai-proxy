import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from src.core.config import CHROMA_PATH, DATA_DIR, BASE_DIR

# Setup paths
DB_DIR = CHROMA_PATH
DATA_FILE = os.path.join(DATA_DIR, "Memories.md")

def inject_memories():
    print(f"Loading ChromaDB from: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Using the primary collection your RAG pipeline queries
    collection = client.get_or_create_collection(name="chat_history")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Granular bullet-point parsing: each milestone gets its own vector embedding
    current_section = "Relationship Timeline"
    documents = []
    ids = []
    metadatas = []

    import hashlib

    # 1. First, include section-level timeline overviews
    raw_chunks = content.split("---")
    for i, chunk in enumerate(raw_chunks):
        c_clean = chunk.strip()
        if c_clean:
            documents.append(c_clean)
            ids.append(f"core_timeline_overview_{i}")
            metadatas.append({"source": "Memories.md", "type": "timeline_overview"})

    # 2. Extract every individual bullet-point milestone
    for line in content.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith("## "):
            current_section = line_clean.replace("## ", "").strip()
            continue
        if line_clean.startswith("* **") or line_clean.startswith("- **"):
            fact_body = line_clean.lstrip("*- ").strip()
            fact_text = f"[{current_section}] {fact_body}"
            doc_id = hashlib.md5(fact_text.lower().encode("utf-8")).hexdigest()[:12]
            documents.append(fact_text)
            ids.append(f"core_fact_{doc_id}")
            metadatas.append({
                "source": "Memories.md",
                "type": "core_fact",
                "section": current_section
            })

    print(f"Embedding {len(documents)} granular core memory facts into the vector space...")
    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    # Also sync into fact_memory so active JIT fact retrieval immediately queries them
    try:
        fact_collection = client.get_or_create_collection(name="fact_memory")
        fact_collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    except Exception as e:
        print(f"Warning syncing with fact_memory: {e}")

    print("✅ Granular core memories successfully injected into ChromaDB!")

if __name__ == "__main__":
    inject_memories()
