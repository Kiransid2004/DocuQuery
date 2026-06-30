from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from hybrid import fit_bm25, encode_sparse, encode_sparse_query, hybrid_score_norm
from dotenv import load_dotenv
import numpy as np
import uuid
import os
import json
from functools import lru_cache

load_dotenv()

# ── Embedding model ────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM   = 384 if "small" in EMBED_MODEL or "MiniLM" in EMBED_MODEL else 1024

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"Embedding model loaded! dim={EMBED_DIM}")
    return _model

def encode(texts: list[str]) -> np.ndarray:
    return get_model().encode(
        texts,
        normalize_embeddings = True,
        batch_size           = 128,
        show_progress_bar    = True,
        convert_to_numpy     = True
    )

@lru_cache(maxsize=512)
def encode_query_cached(query: str) -> tuple:
    """Cache query embeddings — same query never re-encoded."""
    vec = get_model().encode(
        [query],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )[0]
    return tuple(vec.tolist())

# ── Pinecone client ────────────────────────────────────────────────────────────

pc            = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME    = os.getenv("PINECONE_INDEX", "docuquery")
REGISTRY_FILE = "data/registry.json"

def ensure_collection():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name      = INDEX_NAME,
            dimension = EMBED_DIM,
            metric    = "dotproduct",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Pinecone index '{INDEX_NAME}' created ({EMBED_DIM}-dim dotproduct)")
    else:
        print(f"Pinecone index '{INDEX_NAME}' ready ({EMBED_DIM}-dim)")

def get_index():
    return pc.Index(INDEX_NAME)

# ── Document registry ──────────────────────────────────────────────────────────

def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict):
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def register_document(doc_id: str, filename: str, chunk_count: int, health: dict):
    registry = load_registry()
    registry[doc_id] = {
        "doc_id":      doc_id,
        "filename":    filename,
        "chunks":      chunk_count,
        "health":      health["health"],
        "health_note": health["note"]
    }
    save_registry(registry)

def list_documents() -> list:
    return list(load_registry().values())

def delete_document(doc_id: str, namespace: str = "default") -> bool:
    registry = load_registry()
    if doc_id not in registry:
        return False
    index = get_index()
    index.delete(
        filter    = {"doc_id": {"$eq": doc_id}},
        namespace = namespace
    )
    del registry[doc_id]
    save_registry(registry)

    # Remove doc's chunks from store and refit BM25 on remaining corpus
    store   = load_chunks_store()
    removed = [k for k, v in store.items() if v.get("doc_id") == doc_id]
    for k in removed:
        del store[k]
    save_chunks_store(store)
    remaining = get_full_corpus_texts()
    if remaining:
        print(f"Refitting BM25 after delete ({len(remaining)} chunks remain)...")
        fit_bm25(remaining)
    return True

# ── Chunk store ────────────────────────────────────────────────────────────────
# Persists every chunk's text to disk. Powers:
#   1. Full-corpus BM25 refit  — refit on ALL chunks, not just the new upload
#   2. Sentence-window retrieval — fetch ±N neighboring chunks for richer LLM context
#   3. Future embedding migrations — re-embed from here without re-uploading PDFs
#   4. RAGAS evaluation — ground-truth context retrieval

CHUNKS_STORE_FILE = "data/chunks_store.json"
_chunks_cache     = None
_chunks_mtime     = None

def load_chunks_store() -> dict:
    global _chunks_cache, _chunks_mtime
    if not os.path.exists(CHUNKS_STORE_FILE):
        return {}
    mtime = os.path.getmtime(CHUNKS_STORE_FILE)
    if _chunks_cache is None or mtime != _chunks_mtime:
        with open(CHUNKS_STORE_FILE, "r", encoding="utf-8") as f:
            _chunks_cache = json.load(f)
        _chunks_mtime = mtime
    return _chunks_cache

def save_chunks_store(store: dict):
    global _chunks_cache, _chunks_mtime
    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    _chunks_cache = store
    if os.path.exists(CHUNKS_STORE_FILE):
        _chunks_mtime = os.path.getmtime(CHUNKS_STORE_FILE)

def get_chunk_key(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::{chunk_index}"

def get_chunk_window(doc_id: str, chunk_index: int, window: int = 1) -> str:
    """
    Sentence-window retrieval: return target chunk ± `window` neighbors
    from the same document, joined in reading order.
    Falls back to "" if the store has no data for this doc (pre-backfill).
    """
    store = load_chunks_store()
    parts = []
    for offset in range(-window, window + 1):
        key = get_chunk_key(doc_id, chunk_index + offset)
        if key in store:
            parts.append(store[key].get("raw_text", store[key].get("text", "")))
    return "\n".join(parts)

def get_full_corpus_texts() -> list[str]:
    """All chunk texts ever indexed — used for full-corpus BM25 refit."""
    return [v["text"] for v in load_chunks_store().values()]

def warmup_bm25():
    """Called at startup — refits BM25 from persisted corpus so sparse
    query encoding matches the full index after a server restart."""
    corpus = get_full_corpus_texts()
    if corpus:
        print(f"Warming up BM25 on {len(corpus)} stored chunks...")
        fit_bm25(corpus)
    else:
        print("Chunk store empty — BM25 will fit on first upload.")

# ── Feedback store ─────────────────────────────────────────────────────────────
# Stores thumbs-up/down feedback per query for weak-query analysis.

FEEDBACK_FILE = "data/feedback.json"

def load_feedback() -> list:
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_feedback(entries: list):
    os.makedirs("data", exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

def record_feedback(query: str, answer: str, rating: str, sources: list = None):
    """
    rating: 'up' | 'down'
    """
    from datetime import datetime
    entries = load_feedback()
    entries.append({
        "timestamp": datetime.now().isoformat(),
        "query":     query,
        "answer":    answer[:300],
        "rating":    rating,
        "sources":   sources or []
    })
    save_feedback(entries)
    print(f"Feedback recorded: {rating} for '{query[:60]}'")

# ── RAGAS-style evaluation ─────────────────────────────────────────────────────
# Lightweight faithfulness + relevancy scoring using LLM-as-judge.
# No external RAGAS library needed — runs on Groq.

EVAL_LOG_FILE = "data/eval_log.json"

def load_eval_log() -> list:
    if os.path.exists(EVAL_LOG_FILE):
        with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_eval_log(entries: list):
    os.makedirs("data", exist_ok=True)
    with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

async def evaluate_answer(query: str, answer: str, context_chunks: list[str]) -> dict:
    """
    LLM-as-judge RAGAS-style scoring:
      - faithfulness:  is the answer supported by the retrieved context?
      - answer_relevancy: does the answer actually address the question?

    Returns dict with scores (0-1) and a brief rationale.
    Runs async — called in background after answer is streamed.
    """
    import litellm
    context = "\n\n---\n\n".join(context_chunks[:5])

    system = (
        "You are a RAG quality evaluator. Score the answer on two dimensions.\n"
        "Return ONLY valid JSON, no markdown, no explanation outside the JSON.\n"
        "{\n"
        '  "faithfulness": 0.0-1.0,  // Is every claim in the answer supported by the context?\n'
        '  "answer_relevancy": 0.0-1.0, // Does the answer directly address the question?\n'
        '  "faithfulness_reason": "one sentence",\n'
        '  "relevancy_reason": "one sentence"\n'
        "}"
    )
    user = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}"
    )

    try:
        r = await litellm.acompletion(
            model    = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile"),
            messages = [{"role": "system", "content": system},
                        {"role": "user",   "content": user}],
            temperature = 0.0,
            max_tokens  = 300
        )
        raw = r.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
        return {
            "faithfulness":        round(float(scores.get("faithfulness",        0)), 2),
            "answer_relevancy":    round(float(scores.get("answer_relevancy",    0)), 2),
            "faithfulness_reason": scores.get("faithfulness_reason",  ""),
            "relevancy_reason":    scores.get("relevancy_reason",     ""),
        }
    except Exception as e:
        print(f"Eval failed: {e}")
        return {"faithfulness": None, "answer_relevancy": None, "error": str(e)}

async def log_eval(query: str, answer: str, context_chunks: list[str], metadata: dict):
    """Evaluate and persist to eval_log.json in the background."""
    from datetime import datetime
    scores = await evaluate_answer(query, answer, context_chunks)
    entries = load_eval_log()
    entries.append({
        "timestamp":        datetime.now().isoformat(),
        "query":            query,
        "answer_preview":   answer[:200],
        "faithfulness":     scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "faithfulness_reason": scores.get("faithfulness_reason", ""),
        "relevancy_reason":    scores.get("relevancy_reason", ""),
        "confidence":       metadata.get("confidence"),
        "alpha_used":       metadata.get("alpha_used"),
        "docs_searched":    metadata.get("docs_searched", 0),
    })
    # Keep last 500 evals
    save_eval_log(entries[-500:])
    print(
        f"Eval logged — faithfulness: {scores.get('faithfulness')} | "
        f"relevancy: {scores.get('answer_relevancy')}"
    )
    return scores

# ── Upsert ─────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], doc_id: str, filename: str, namespace: str = "default"):
    texts = [c["text"] for c in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    dense_vectors = encode(texts)

    # Persist chunks BEFORE fitting BM25 so the full corpus includes them
    store = load_chunks_store()
    for i, c in enumerate(chunks):
        key = get_chunk_key(doc_id, c.get("chunk_index", i))
        store[key] = {
            "text":        c["text"],
            "raw_text":    c.get("raw_text", c["text"]),
            "page":        c.get("page", 0),
            "filename":    filename,
            "doc_id":      doc_id,
            "chunk_index": c.get("chunk_index", i)
        }
    save_chunks_store(store)

    # Full-corpus BM25 refit — IDF stats now reflect ALL indexed chunks
    full_corpus = get_full_corpus_texts()
    print(f"Fitting BM25 on full corpus ({len(full_corpus)} total, +{len(texts)} new)...")
    fit_bm25(full_corpus)
    sparse_vectors = encode_sparse(texts)

    index      = get_index()
    batch_size = 100
    all_vectors = []

    for i, (c, dv, sv) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        all_vectors.append({
            "id":            str(uuid.uuid4()),
            "values":        dv.tolist(),
            "sparse_values": sv,
            "metadata": {
                "text":        c["text"],
                "raw_text":    c.get("raw_text", c["text"]),
                "doc_id":      doc_id,
                "filename":    filename,
                "namespace":   namespace,
                "page":        c.get("page", 0),
                "chunk_index": c.get("chunk_index", i)
            }
        })

    total = len(all_vectors)
    for i in range(0, total, batch_size):
        batch = all_vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1} "
              f"({min(i+batch_size, total)}/{total} chunks)")

    print(f"Done — {total} chunks for: {filename} [namespace: {namespace}]")

# ── Search ─────────────────────────────────────────────────────────────────────

def search(query, top_k=6, doc_ids=None, alpha=0.55, namespace: str = "default"):
    dense_q  = list(encode_query_cached(query))
    sparse_q = encode_sparse_query(query)
    dense_scaled, sparse_scaled = hybrid_score_norm(dense_q, sparse_q, alpha)

    query_filter = None
    if doc_ids and len(doc_ids) > 0:
        query_filter = (
            {"doc_id": {"$eq": doc_ids[0]}}
            if len(doc_ids) == 1
            else {"doc_id": {"$in": doc_ids}}
        )

    index   = get_index()
    results = index.query(
        vector           = dense_scaled,
        sparse_vector    = sparse_scaled,
        top_k            = top_k,
        include_metadata = True,
        filter           = query_filter,
        namespace        = namespace
    )

    return [
        {
            "text":        m.metadata.get("text", ""),
            "raw_text":    m.metadata.get("raw_text", m.metadata.get("text", "")),
            "doc_id":      m.metadata.get("doc_id", ""),
            "filename":    m.metadata.get("filename", "unknown"),
            "page":        m.metadata.get("page", 0),
            "chunk_index": m.metadata.get("chunk_index", 0),
            "score":       round(m.score, 3)
        }
        for m in results.matches
    ]