"""Vector, BM25 and hybrid (RRF) search over the chunks stored by ingest.py.

Try it:  python src/retrieve.py "Koliko stranica mora imati seminarski rad?"
"""

import re
import sys
from functools import cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "docs"
MODEL_NAME = "intfloat/multilingual-e5-base"

# Standard RRF constant from the original paper; damps the gap between rank 1 and rank 2.
RRF_K = 60


def tokenize(text):
    """Lowercase word tokens for BM25. \\w is Unicode-aware, so č, ć, š, ž, đ stay intact."""
    return re.findall(r"\w+", text.lower())


@cache
def _load():
    """Load model, Chroma collection and BM25 index once, then reuse (loading is slow)."""
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)

    # BM25 lives in memory, so we build it from the same chunks Chroma has.
    # That keeps one source of truth: re-run ingest.py and both indexes update.
    data = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"chunk_id": cid, "text": doc, "source": meta["source"], "page": meta["page"]}
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return model, collection, bm25, chunks


def vector_search(query, k=5):
    """Semantic search: finds chunks with similar meaning, even with different words."""
    model, collection, _, _ = _load()
    # e5 needs "query: " for questions (and "passage: " for chunks, done in ingest).
    q_emb = model.encode("query: " + query, normalize_embeddings=True)
    res = collection.query(query_embeddings=[q_emb.tolist()], n_results=k)

    results = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({
            "chunk_id": cid, "text": doc, "source": meta["source"], "page": meta["page"],
            "score": 1 - dist,  # Chroma returns cosine distance; 1 - distance = similarity
        })
    return results


def bm25_search(query, k=5):
    """Keyword search: good for exact terms (names, codes, 'član 12') that embeddings can blur."""
    _, _, bm25, chunks = _load()
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    # Drop zero scores: no query word appears in the chunk at all.
    return [{**chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0]


def hybrid_search(query, k=5, candidates=20):
    """Combine both rankings with Reciprocal Rank Fusion.

    Cosine similarity (0..1) and BM25 scores (0..any) are on different scales,
    so we ignore the raw scores and use only rank: score = sum 1 / (RRF_K + rank).
    A chunk ranked well by BOTH methods ends up on top.
    """
    fused = {}
    for results in (vector_search(query, candidates), bm25_search(query, candidates)):
        for rank, r in enumerate(results, start=1):
            entry = fused.setdefault(r["chunk_id"], {**r, "score": 0.0})
            entry["score"] += 1 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:k]


def _print_results(title, results):
    print(f"\n=== {title} ===")
    for r in results:
        preview = r["text"][:120].replace("\n", " ")
        print(f"{r['score']:.4f} | {r['source']} str. {r['page']} | {preview}...")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Koliko stranica mora imati seminarski rad?"
    print(f"Query: {query}")
    _print_results("Vector", vector_search(query, 3))
    _print_results("BM25", bm25_search(query, 3))
    _print_results("Hybrid (RRF)", hybrid_search(query, 3))
