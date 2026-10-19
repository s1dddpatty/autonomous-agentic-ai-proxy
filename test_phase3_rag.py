import os
import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.vector_memory import (
    save_semantic_fact,
    retrieve_relevant_facts,
    get_fact_memory_collection,
    FACT_COLLECTION_NAME
)
import src.core.gemini_node as gn
from src.core.fact_extractor import extract_and_store_facts_sync, parse_facts_json

def run_phase3_verification():
    print("=" * 60)
    print("PHASE 3 VERIFICATION: Autonomous Self-Learning & Semantic RAG Facts")
    print("=" * 60)

    # Step 1: Verify fact_memory collection and save_semantic_fact
    print("\n--- Step 1: Save Semantic Fact ---")
    fact_text = "Aarushi's dad loves butterfly gardening"
    fact_id = save_semantic_fact(fact_text)
    print(f"Saved fact: '{fact_text}' with ID: {fact_id}")
    assert fact_id is not None and len(fact_id) == 32, f"Expected 32-char MD5 hash, got {fact_id}"

    # Step 2: Test duplicate prevention (stable hash)
    fact_id_dup = save_semantic_fact("aarushi's dad loves butterfly gardening")
    assert fact_id == fact_id_dup, f"Expected duplicate to produce same hash {fact_id}, got {fact_id_dup}"
    print(f"Duplicate upsert verified: stable hash {fact_id_dup}")

    # Step 3: Test semantic retrieval via vector search
    print("\n--- Step 2: Retrieve Relevant Facts via Semantic Search ---")
    query = "aree kolkata me ek garden me gaye..vaha kitna butterflies dikhe"
    retrieved = retrieve_relevant_facts(query, k=2)
    print(f"Query: '{query}'")
    print(f"Retrieved Facts: {retrieved}")
    assert len(retrieved) > 0, "Expected at least 1 fact retrieved"
    assert any("butterfly" in f.lower() for f in retrieved), f"Expected butterfly fact in {retrieved}"
    print("✅ Semantic fact retrieval successful!")

    # Step 4: Test Fact Extractor Parser
    print("\n--- Step 3: Fact Extractor JSON Parser ---")
    sample_json_markdown = '```json\n["Aarushi has an upcoming trip to Jaipur", "Aarushi is allergic to peanuts"]\n```'
    parsed_facts = parse_facts_json(sample_json_markdown)
    print(f"Parsed facts: {parsed_facts}")
    assert len(parsed_facts) == 2
    assert parsed_facts[0] == "Aarushi has an upcoming trip to Jaipur"
    print("✅ Fact extractor parser verified!")

    # Step 5: Live test of gemini_proxy_node with JIT injected fact
    print("\n--- Step 4: Live gemini_proxy_node with Semantic JIT Injection ---")
    gn._active_key_index = 0
    state = {
        "incoming_text": "aree kolkata me ek garden me gaye..vaha kitna butterflies dikhe",
        "messages": []
    }
    result = gn.gemini_proxy_node(state)
    draft = result.get("draft_text", "")
    print("\nLive Generated Draft on gemini-flash-latest:")
    print("-" * 50)
    print(draft)
    print("-" * 50)
    print(f"Active key index after call: Key #{gn._active_key_index + 1} (index {gn._active_key_index})")
    assert draft and not draft.startswith("⚠️ [GEMINI ERROR"), f"Draft generation failed: {draft}"
    print("✅ Live JIT Fact Injection test successful!")

    # Step 6: Test Modulo Trigger logic in orchestrator
    print("\n--- Step 5: Test Orchestrator Modulo Trigger ---")
    from src.core.memory_manager import get_total_messages_count
    total = get_total_messages_count()
    print(f"Total messages count from database: {total}")
    print(f"Next modulo 8 trigger will occur at total % 8 == 0 (current: {total % 8})")

if __name__ == "__main__":
    run_phase3_verification()
