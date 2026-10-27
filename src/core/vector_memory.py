import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import chromadb

from src.core.config import CHROMA_PATH, DB_PATH, DATA_DIR, BASE_DIR

STYLE_COLLECTION_NAME = "style_bursts"
EPISODIC_COLLECTION_NAME = "episodic_memories"
FACT_COLLECTION_NAME = "fact_memory"

def is_placeholder_line(line: str) -> bool:
    """
    Identifies common WhatsApp export artifact lines that must be stripped out,
    as well as API quota fallback strings so they are never indexed in memory.
    """
    line_clean = line.strip().lower()
    if not line_clean:
        return True

    placeholders = [
        "<image omitted>",
        "<media omitted>",
        "<sticker omitted>",
        "<audio omitted>",
        "<video omitted>",
        "<document omitted>",
        "image omitted",
        "media omitted",
        "sticker omitted",
        "audio omitted",
        "video omitted",
        "document omitted",
        "this message was deleted",
        "you deleted this message",
        "waiting for this message. this may take a while.",
        "waiting for this message",
        f"m rlyy sorryy {os.getenv('TARGET_CONTACT_NAME', 'PersonName').lower()}",
        "m rlyy sorryy aarushii",
        "ill be back in ",
        "m superr tired...i am going to rest now",
    ]
    for p in placeholders:
        if p in line_clean:
            return True

    if re.search(r'<[^>]*omitted[^>]*>', line_clean):
        return True

    return False

def clean_burst_text(text: str) -> str:
    """
    Strips out WhatsApp export placeholder lines and tags from a text burst.
    Returns cleaned multi-line text or empty string if all lines were placeholders.
    """
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    for l in lines:
        l_str = l.strip()
        if not l_str:
            continue
        l_no_tag = re.sub(r'<[^>]*omitted[^>]*>', '', l_str, flags=re.IGNORECASE).strip()
        l_no_tag = re.sub(r'\[Forwarded\]\s*', '', l_no_tag, flags=re.IGNORECASE).strip()
        if is_placeholder_line(l_str) or not l_no_tag:
            if l_no_tag and not is_placeholder_line(l_no_tag):
                cleaned_lines.append(l_no_tag)
            continue
        cleaned_lines.append(l_no_tag)
    return "\n".join(cleaned_lines).strip()

def classify_message_vibe(text: str) -> dict:
    """
    Fast heuristic classifier (zero API calls) that inspects text, punctuation,
    emoji density, and keywords to determine:
    - mood: 'affectionate', 'playful', 'sarcastic', 'serious_upset', or 'casual'
    - conv_stage: 'opener', 'closer', or 'ongoing'
    """
    raw = text.strip()
    lower = raw.lower()

    # 1. Conversation Stage Detection
    closer_pattern = r'\b(g+n+|good\s*night+|so\s*ja+o?|so\s*rahi|so\s*rhi|bye+|byebye+|buhbye+|sweet\s*dreams|tata|tc|sleep\s*tight)\b'
    opener_pattern = r'\b(h+e+y+|h+i+|hello+|hellu+|wassup|watsup|sup|good\s*morning+|g+m+|morning+|uth\s*ga?y[ai]|sun\s*no?)\b'

    if re.search(closer_pattern, lower):
        conv_stage = "closer"
    elif re.search(opener_pattern, lower):
        conv_stage = "opener"
    else:
        conv_stage = "ongoing"

    # 2. Emoji Signals
    affectionate_emojis = {'❤️', '💖', '🫶', '🫶🏻', '💕', '🥺', '😴', '😌', '✨', '✨️', '💓', '😘', '🫂', '🤍', '🥰'}
    playful_emojis = {'😂', '🤣', '😜', '😝', '🤪', '💀', '😭', '🤡'}
    sarcastic_emojis = {'🙄', '😏', '😒', '🙃', '👏', '🫡'}
    serious_emojis = {'😠', '😡', '💔', '😞', '😔', '😢', '😤'}

    aff_score = sum(raw.count(e) for e in affectionate_emojis)
    play_score = sum(raw.count(e) for e in playful_emojis)
    sarc_score = sum(raw.count(e) for e in sarcastic_emojis)
    ser_score = sum(raw.count(e) for e in serious_emojis)

    # 3. Keyword Signals
    aff_words = [
        'love', 'loveyou', 'love you', 'loveya', 'sweet dreams', 'bed bugs',
        'miss you', 'miss u', 'baby', 'babu', 'cutie', 'jaan', 'sardi ka', 'care'
    ]
    ser_words = [
        'kya hua', 'naraz', 'gussa', 'upset', 'hurt', 'serious', 'mat baat kar',
        'chup', 'problem', 'mood off', 'kuch nahi', 'kuch nhi', 'leave it',
        'rehne de', 'rahne de', 'dont talk', 'bore', 'annoying', 'hate', 'fight'
    ]
    sarc_words = [
        'wah', 'waah', 'great', 'claps', 'obviously', 'of course', 'as if',
        'nice excuse', 'sahi hai', 'achha ji', 'acha ji', 'irony', 'sarcasm'
    ]
    play_words = [
        'lmao', 'lmaoo', 'lol', 'haha', 'hahaha', 'hehe', 'rofl', 'pagal',
        'chirkut', 'loser', 'noob', 'dumbo', 'masti', 'joke'
    ]

    for w in aff_words:
        if w in lower:
            aff_score += 2
    for w in ser_words:
        if w in lower:
            ser_score += 3
    for w in sarc_words:
        if w in lower:
            sarc_score += 2
    for w in play_words:
        if w in lower:
            play_score += 2

    # Punctuation Signals
    if re.search(r'\?{2,}|\.{3,}', raw) and ('?' in raw or ser_score > 0):
        ser_score += 1
    if re.search(r'!{2,}|haha', lower):
        play_score += 1

    scores = {
        'affectionate': aff_score,
        'serious_upset': ser_score,
        'sarcastic': sarc_score,
        'playful': play_score
    }
    top_mood = max(scores, key=scores.get)
    if scores[top_mood] > 0:
        mood = top_mood
    else:
        if conv_stage == 'closer' and any(c in lower for c in ['goodnight', 'gn', 'sweet']):
            mood = 'affectionate'
        else:
            mood = 'casual'

    return {"mood": mood, "conv_stage": conv_stage}

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client pointing to chroma_db/.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_style_bursts_collection():
    """
    Collection storing Target Contact Trigger bursts as vectors, with Operator Response bursts in metadata.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=STYLE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def get_episodic_memories_collection():
    """
    Collection storing full exchanges (Target Contact Trigger + Operator Response) for deep context recall.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=EPISODIC_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def get_fact_memory_collection():
    """
    Collection storing permanent facts, relationship details, and personal attributes.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=FACT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def save_semantic_fact(fact_text: str) -> str:
    """
    Upserts a semantic fact into the fact_memory collection with a stable MD5 hash ID.
    Prevents duplicates while keeping the facts searchable via vector embeddings.
    """
    fact_clean = fact_text.strip()
    if not fact_clean:
        return ""
    fact_id = hashlib.md5(fact_clean.lower().encode("utf-8")).hexdigest()
    collection = get_fact_memory_collection()
    collection.upsert(
        ids=[fact_id],
        documents=[fact_clean],
        metadatas=[{"created_at": datetime.now().isoformat()}]
    )
    return fact_id

def retrieve_relevant_facts(query: str, k: int = 3) -> list[str]:
    """
    Retrieves the top k most relevant semantic facts for a given query text.
    Returns an empty list if no facts exist or collection is empty.
    """
    query_clean = query.strip()
    if not query_clean:
        return []
    try:
        collection = get_fact_memory_collection()
        count = collection.count()
        if count == 0:
            return []
        n_results = min(k, count)
        results = collection.query(
            query_texts=[query_clean],
            n_results=n_results
        )
        docs = results.get("documents", [[]])[0]
        return [d.strip() for d in docs if d and d.strip()]
    except Exception as e:
        print(f"[Vector Memory] Error retrieving relevant facts: {e}")
        return []

class ChromaDBManager:
    """
    Manager wrapping ChromaDB collections to expose similarity_search()
    for contextualized retrieval from Memories.md and relationship facts.
    """
    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        query_clean = query.strip()
        if not query_clean:
            return []
        try:
            client = get_chroma_client()
            docs = []

            # 1. Search fact_memory collection (where Memories.md chunks are stored)
            fact_col = get_fact_memory_collection()
            if fact_col.count() > 0:
                n_res = min(k, fact_col.count())
                results = fact_col.query(query_texts=[query_clean], n_results=n_res)
                if results and "documents" in results and results["documents"]:
                    for doc in results["documents"][0]:
                        if doc and doc.strip() and doc.strip() not in docs:
                            docs.append(doc.strip())

            # 2. Search chat_history collection if more documents are needed
            if len(docs) < k:
                try:
                    chat_col = client.get_collection(name="chat_history")
                    if chat_col.count() > 0:
                        n_res = min(k, chat_col.count())
                        results = chat_col.query(query_texts=[query_clean], n_results=n_res)
                        if results and "documents" in results and results["documents"]:
                            for doc in results["documents"][0]:
                                if doc and doc.strip() and doc.strip() not in docs:
                                    docs.append(doc.strip())
                except Exception:
                    pass

            return docs[:k]
        except Exception as e:
            print(f"[ChromaDB] Error in similarity_search: {e}")
            return []

chroma_db = ChromaDBManager()

def _group_bubbles_into_bursts(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Groups consecutive messages from the same speaker into a single multi-line burst.
    """
    grouped = []
    for msg in messages:
        role = msg.get("role", "")
        text = clean_burst_text(msg.get("text", ""))
        if not text:
            continue
        if grouped and grouped[-1]["role"] == role:
            grouped[-1]["text"] += "\n" + text
        else:
            grouped.append({"role": role, "text": text})
    return grouped

def extract_burst_pairs_from_jsonl(filepath: str) -> List[Dict]:
    """
    Parses a JSONL file, filters WhatsApp artifacts, and pairs every Target Contact Burst (Trigger)
    with the immediately following Response Burst.
    """
    pairs = []
    if not os.path.exists(filepath):
        return pairs

    with open(filepath, "r", encoding="utf-8") as f:
        for block_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            convs = [m for m in data.get("conversations", []) if m.get("from") != "system"]
            target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
            raw_messages = []
            for m in convs:
                role = target_name if m.get("from") == "human" else "Siddharth"
                val = clean_burst_text(m.get("value", ""))
                if val:
                    raw_messages.append({"role": role, "text": val})

            bursts = _group_bubbles_into_bursts(raw_messages)
            for i in range(len(bursts) - 1):
                if bursts[i]["role"] in [target_name, "Aarushi", "human"] and bursts[i + 1]["role"] == "Siddharth":
                    a_text = clean_burst_text(bursts[i]["text"])
                    s_text = clean_burst_text(bursts[i + 1]["text"])
                    # Skip pairs where either trigger or response was only placeholder/empty
                    if not a_text or not s_text:
                        continue

                    pair_hash = hashlib.sha256(f"{a_text}:::{s_text}".encode("utf-8")).hexdigest()[:16]
                    pairs.append({
                        "hash": pair_hash,
                        "aarushi_burst": a_text,
                        "siddharth_burst": s_text,
                        "source": f"jsonl_{os.path.basename(filepath)}_b{block_idx}_p{i}",
                        "timestamp": f"jsonl_block_{block_idx}"
                    })
    return pairs

def extract_burst_pairs_from_db(db_path: str) -> List[Dict]:
    """
    Reads SQLite messages table, filters WhatsApp artifacts, groups consecutive speaker
    messages into bursts, and pairs Target Contact Trigger -> Response.
    """
    pairs = []
    if not os.path.exists(db_path):
        return pairs

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, role, content, timestamp FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
    raw_messages = []
    for row in rows:
        db_id, role, content, ts = row
        role_standard = "Siddharth" if role in ["Siddharth", "ai", "model", "gpt"] else target_name
        content_clean = clean_burst_text(content or "")
        if content_clean:
            raw_messages.append({"role": role_standard, "text": content_clean, "ts": str(ts)})

    bursts = _group_bubbles_into_bursts(raw_messages)

    for i in range(len(bursts) - 1):
        if bursts[i]["role"] in [target_name, "Aarushi", "human"] and bursts[i + 1]["role"] == "Siddharth":
            a_text = clean_burst_text(bursts[i]["text"])
            s_text = clean_burst_text(bursts[i + 1]["text"])
            if not a_text or not s_text:
                continue

            pair_hash = hashlib.sha256(f"{a_text}:::{s_text}".encode("utf-8")).hexdigest()[:16]
            pairs.append({
                "hash": pair_hash,
                "aarushi_burst": a_text,
                "siddharth_burst": s_text,
                "source": f"sqlite_chats_db_p{i}",
                "timestamp": bursts[i].get("ts", "sqlite_history")
            })
    return pairs

def extract_burst_pairs_from_chats_dir(chats_dir: str) -> List[Dict]:
    """
    Scans the chats/ folder for any .jsonl or .json files, filters artifacts, and extracts burst pairs.
    """
    pairs = []
    if not os.path.exists(chats_dir):
        return pairs

    for root, _, files in os.walk(chats_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if file.endswith(".jsonl"):
                pairs.extend(extract_burst_pairs_from_jsonl(filepath))
            elif file.endswith(".json"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
                        raw_messages = []
                        for m in data:
                            role = "Siddharth" if m.get("role") in ["Siddharth", "gpt", "model"] else target_name
                            text = clean_burst_text(m.get("content") or m.get("text") or "")
                            if text:
                                raw_messages.append({"role": role, "text": text})
                        bursts = _group_bubbles_into_bursts(raw_messages)
                        for i in range(len(bursts) - 1):
                            if bursts[i]["role"] in [target_name, "Aarushi", "human"] and bursts[i + 1]["role"] == "Siddharth":
                                a_text = clean_burst_text(bursts[i]["text"])
                                s_text = clean_burst_text(bursts[i + 1]["text"])
                                if not a_text or not s_text:
                                    continue

                                pair_hash = hashlib.sha256(f"{a_text}:::{s_text}".encode("utf-8")).hexdigest()[:16]
                                pairs.append({
                                    "hash": pair_hash,
                                    "aarushi_burst": a_text,
                                    "siddharth_burst": s_text,
                                    "source": f"json_{file}_p{i}",
                                    "timestamp": "chats_folder"
                                })
                except Exception as e:
                    print(f"[Vector Memory] Error parsing JSON file {filepath}: {e}")
    return pairs

def index_chats_to_chroma(batch_size: int = 200, clear_existing: bool = True) -> dict:
    """
    Groups all messages from dataset.jsonl, data/chats.db, and chats/ into clean
    Target Contact Trigger -> Response bursts (with media placeholders stripped),
    and indexes them into:
    1. style_bursts (Trigger vector -> Response metadata)
    2. episodic_memories (Full exchange vector -> topic/fact recall)
    """
    client = get_chroma_client()

    # Clear out legacy collections if requested
    if clear_existing:
        for col_name in [STYLE_COLLECTION_NAME, EPISODIC_COLLECTION_NAME, "chat_memories"]:
            try:
                client.delete_collection(col_name)
                print(f"[Vector Memory] Cleared collection '{col_name}'.")
            except Exception:
                pass

    style_col = get_style_bursts_collection()
    episodic_col = get_episodic_memories_collection()

    existing_style_ids = set(style_col.get(include=[])["ids"])
    existing_episodic_ids = set(episodic_col.get(include=[])["ids"])

    print(f"[Vector Memory] Current style_bursts: {len(existing_style_ids)}, episodic_memories: {len(existing_episodic_ids)}")

    # Collect burst pairs across all sources
    all_pairs: Dict[str, Dict] = {}

    # 1. Root dataset.jsonl
    jsonl_path = os.path.join(BASE_DIR, "dataset.jsonl")
    if os.path.exists(jsonl_path):
        for p in extract_burst_pairs_from_jsonl(jsonl_path):
            all_pairs[p["hash"]] = p

    # 2. SQLite chats.db
    db_path = DB_PATH
    if os.path.exists(db_path):
        for p in extract_burst_pairs_from_db(db_path):
            if p["hash"] not in all_pairs:
                all_pairs[p["hash"]] = p

    # 3. chats/ directory
    chats_dir = os.path.join(BASE_DIR, "chats")
    if os.path.exists(chats_dir):
        for p in extract_burst_pairs_from_chats_dir(chats_dir):
            if p["hash"] not in all_pairs:
                all_pairs[p["hash"]] = p

    total_unique_pairs = len(all_pairs)
    print(f"[Vector Memory] Found {total_unique_pairs} pure-text burst pairs across sources.")

    # Prepare batches for style_bursts and episodic_memories
    style_docs, style_ids, style_metas = [], [], []
    episodic_docs, episodic_ids, episodic_metas = [], [], []

    added_style, added_episodic = 0, 0
    skipped_style, skipped_episodic = 0, 0

    for pair_hash, p in all_pairs.items():
        a_burst = p["aarushi_burst"]
        s_burst = p["siddharth_burst"]
        source_id = p["source"]
        timestamp_str = p.get("timestamp", "")

        # Classify vibe of trigger burst
        vibe = classify_message_vibe(a_burst)
        detected_mood = vibe["mood"]
        detected_stage = vibe["conv_stage"]

        style_id = f"style_{pair_hash}"
        if style_id in existing_style_ids:
            skipped_style += 1
        else:
            style_docs.append(a_burst)
            style_ids.append(style_id)
            style_metas.append({
                "siddharth_reply": s_burst[:3500],
                "mood": detected_mood,
                "conv_stage": detected_stage,
                "source": source_id
            })

        episodic_id = f"mem_{pair_hash}"
        if episodic_id in existing_episodic_ids:
            skipped_episodic += 1
        else:
            target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
            episodic_docs.append(f"{target_name}: {a_burst}\nSiddharth: {s_burst}")
            episodic_ids.append(episodic_id)
            episodic_metas.append({
                "mood": detected_mood,
                "timestamp": timestamp_str
            })

        # Batch write style bursts
        if len(style_docs) >= batch_size:
            style_col.add(documents=style_docs, ids=style_ids, metadatas=style_metas)
            added_style += len(style_ids)
            print(f"[Vector Memory] Indexed {added_style} pure style bursts...")
            style_docs.clear()
            style_ids.clear()
            style_metas.clear()

        # Batch write episodic memories
        if len(episodic_docs) >= batch_size:
            episodic_col.add(documents=episodic_docs, ids=episodic_ids, metadatas=episodic_metas)
            added_episodic += len(episodic_ids)
            episodic_docs.clear()
            episodic_ids.clear()
            episodic_metas.clear()

    # Flush remaining
    if style_docs:
        style_col.add(documents=style_docs, ids=style_ids, metadatas=style_metas)
        added_style += len(style_ids)
    if episodic_docs:
        episodic_col.add(documents=episodic_docs, ids=episodic_ids, metadatas=episodic_metas)
        added_episodic += len(episodic_ids)

    print(f"✅ Pure Text Ingestion Complete!")
    print(f"   style_bursts: {added_style} added, {skipped_style} skipped (Total: {style_col.count()})")
    print(f"   episodic_memories: {added_episodic} added, {skipped_episodic} skipped (Total: {episodic_col.count()})")

    return {
        "style_added": added_style,
        "style_skipped": skipped_style,
        "style_total": style_col.count(),
        "episodic_added": added_episodic,
        "episodic_skipped": skipped_episodic,
        "episodic_total": episodic_col.count()
    }

def retrieve_dual_context(
    incoming_text: str,
    n_style: int = 10,
    n_memory: int = 10
) -> dict:
    """
    Mood-Aware Trigger -> Burst Dual RAG retrieval.
    1. Runs local heuristic classification on incoming_text.
    2. Queries style_bursts for Top 10 matches, prioritizing matching stage and mood.
    3. Queries episodic_memories for Top 10 deep episodic context matches.
    Returns:
        {
            "detected_mood": str,
            "detected_stage": str,
            "style_chunks": str,
            "memory_chunks": str
        }
    """
    vibe = classify_message_vibe(incoming_text)
    detected_mood = vibe["mood"]
    detected_stage = vibe["conv_stage"]

    style_col = get_style_bursts_collection()
    episodic_col = get_episodic_memories_collection()

    style_count = style_col.count()
    episodic_count = episodic_col.count()

    # 1. Query Style Bursts (Top 10)
    style_chunks_list = []
    if style_count > 0:
        candidate_k = min(max(n_style * 3, 30), style_count)
        style_results = style_col.query(
            query_texts=[incoming_text],
            n_results=candidate_k
        )

        docs = style_results.get("documents", [[]])[0]
        metas = style_results.get("metadatas", [[]])[0]
        distances = style_results.get("distances", [[]])[0] if "distances" in style_results else [0.0] * len(docs)

        candidates = []
        for doc, meta, dist in zip(docs, metas, distances):
            meta = meta or {}
            stage = meta.get("conv_stage", "ongoing")
            mood = meta.get("mood", "casual")
            siddharth_reply = clean_burst_text(meta.get("siddharth_reply", ""))
            doc_cleaned = clean_burst_text(doc)

            if not siddharth_reply or not doc_cleaned:
                continue

            # Priority tier
            if stage == detected_stage and mood == detected_mood:
                tier = 0
            elif stage == detected_stage:
                tier = 1
            elif mood == detected_mood:
                tier = 2
            else:
                tier = 3

            candidates.append({
                "tier": tier,
                "distance": dist,
                "aarushi_burst": doc_cleaned,
                "siddharth_reply": siddharth_reply,
                "stage": stage,
                "mood": mood
            })

        # Sort by (tier ASC, distance ASC)
        candidates.sort(key=lambda c: (c["tier"], c["distance"]))

        target_name = os.getenv("TARGET_CONTACT_NAME", "PersonName")
        selected_style = candidates[:n_style]
        for idx, item in enumerate(selected_style, 1):
            chunk = (
                f"[Style Match #{idx} | Stage: {item['stage']} | Mood: {item['mood']}]\n"
                f"When {target_name} said: \"{item['aarushi_burst']}\"\n"
                f"Siddharth replied:\n{item['siddharth_reply']}"
            )
            style_chunks_list.append(chunk)

    style_chunks_str = "\n\n".join(style_chunks_list) if style_chunks_list else "No style burst matches found."

    # 2. Query Episodic Memories (Top 10)
    memory_chunks_list = []
    if episodic_count > 0:
        mem_k = min(n_memory * 2, episodic_count)
        mem_results = episodic_col.query(
            query_texts=[incoming_text],
            n_results=mem_k
        )

        mem_docs = mem_results.get("documents", [[]])[0]
        mem_metas = mem_results.get("metadatas", [[]])[0]

        seen_memories = set()
        for doc, meta in zip(mem_docs, mem_metas):
            meta = meta or {}
            mood = meta.get("mood", "casual")
            doc_clean = clean_burst_text(doc)
            if not doc_clean or doc_clean in seen_memories:
                continue
            seen_memories.add(doc_clean)

            idx = len(memory_chunks_list) + 1
            memory_chunks_list.append(
                f"[Memory Match #{idx} | Mood: {mood}]\n{doc_clean}"
            )
            if len(memory_chunks_list) >= n_memory:
                break

    memory_chunks_str = "\n\n".join(memory_chunks_list) if memory_chunks_list else "No episodic memory matches found."

    return {
        "detected_mood": detected_mood,
        "detected_stage": detected_stage,
        "style_chunks": style_chunks_str,
        "memory_chunks": memory_chunks_str
    }

if __name__ == "__main__":
    print("[Vector Memory] Running full clean burst indexing...")
    index_chats_to_chroma(clear_existing=True)
