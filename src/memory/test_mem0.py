import os
from mem0 import Memory

# Ensure the data directory exists
os.makedirs("./data/chroma_db", exist_ok=True)

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "persona_memory",
            "path": "./data/chroma_db"
        }
    }
}

m = Memory.from_config(config)

# Test Ingestion
m.add("My sister adopted a cat named Milo last week.", user_id="sister")
m.add("My sister works as a graphic designer in New York.", user_id="sister")

# Test Retrieval
query = "What does my sister do for a living and does she have pets?"
results = m.search(query, filters={"user_id": "sister"})
print(results)
