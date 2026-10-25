import os
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

    # Split the markdown file by horizontal rules to create semantic timeline chunks
    raw_chunks = content.split("---")
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    documents = []
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        ids.append(f"core_memory_timeline_{i}")
        # Tagging it as a core_fact so it holds semantic weight in retrieval
        metadatas.append({"source": "Memories.md", "type": "core_fact"})

    print(f"Embedding {len(chunks)} core memory chunks into the vector space...")
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

    print("✅ Core memories successfully injected into ChromaDB!")

if __name__ == "__main__":
    inject_memories()
