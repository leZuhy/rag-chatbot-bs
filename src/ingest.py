"""PDF -> page text -> chunks -> e5 embeddings -> persistent Chroma.

Run from the project root:  python src/ingest.py
"""

import re
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("data/pdfs")
DB_DIR = "chroma_db"
COLLECTION = "docs"
MODEL_NAME = "intfloat/multilingual-e5-base"

# e5-base sees at most 512 tokens; 480 leaves room for the "passage: " prefix
# and the special start/end tokens, so nothing gets silently truncated.
CHUNK_TOKENS = 480
OVERLAP_TOKENS = 50

# Course syllabi start each page with "Naziv predmeta: <NAME>".
COURSE_RE = re.compile(r"Naziv predmeta:\s*(.+)")


def make_header(pdf_path, course):
    """Context line prepended to a chunk for embedding and BM25 (not for citations).

    A chunk like "Redovni studenti su obavezni prisustvovati nastavi minimalno 70%..."
    appears in almost every syllabus; without the course name it can't be told apart.
    """
    header = f"Dokument: {pdf_path.stem}."
    if course:
        header += f" Predmet: {course}."
    return header


def load_pages(pdf_path):
    """Return a list of (page_number, text). Page numbers start at 1, like in a PDF viewer."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:  # skip empty / image-only pages
                pages.append((i + 1, text))
    return pages


def chunk_text(text, tokenizer):
    """Split text into overlapping windows of CHUNK_TOKENS tokens.

    We count tokens with the model's own tokenizer (not words), so the size
    matches exactly what the embedding model sees. Offsets let us cut the
    ORIGINAL text, so chunks keep their exact wording and spacing.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []

    chunks = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(offsets), step):
        end = min(start + CHUNK_TOKENS, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(text[char_start:char_end].strip())
        if end == len(offsets):
            break
    return chunks


def main():
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    texts, headers, metadatas, ids = [], [], [], []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        pages = load_pages(pdf_path)
        n_before = len(texts)
        course = None  # carried over to the next page if a syllabus spans pages
        for page_num, page_text in pages:
            match = COURSE_RE.search(page_text)
            if match:
                course = match.group(1).strip()
            header = make_header(pdf_path, course)
            # Chunk per page so every chunk has exactly one page number to cite.
            for i, chunk in enumerate(chunk_text(page_text, tokenizer)):
                chunk_id = f"{pdf_path.name}_p{page_num}_c{i}"
                texts.append(chunk)
                headers.append(header)
                metadatas.append({
                    "source": pdf_path.name, "page": page_num, "chunk_id": chunk_id,
                    "header": header,  # retrieve.py uses it for BM25
                    "course": course or "",  # Chroma metadata can't be None; answer.py puts it in the title
                })
                ids.append(chunk_id)
        print(f"{pdf_path.name}: {len(pages)} pages with text, {len(texts) - n_before} chunks")

    if not texts:
        print("No text found. Are the PDFs scanned images?")
        return

    # e5 was trained with prefixes; without "passage: " retrieval quality drops.
    # The header goes into the embedding only; Chroma stores the clean text for citations.
    to_embed = [f"passage: {h}\n{t}" for h, t in zip(headers, texts)]
    # The header adds ~15-25 tokens, so a full 480-token chunk can pass e5's 512 limit
    # and lose its last few tokens (those are also in the next chunk via the overlap).
    too_long = sum(len(tokenizer(s)["input_ids"]) > 512 for s in to_embed)
    print(f"Chunks over 512 tokens with header (tail truncated): {too_long}/{len(to_embed)}")

    # Normalized vectors -> cosine similarity == dot product.
    embeddings = model.encode(
        to_embed,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    # Rebuild from scratch each run, so old/deleted PDFs don't leave stale chunks.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # Store the raw chunk text (without prefix)
    batch = 1000  # Chroma has a max batch size
    for s in range(0, len(texts), batch):
        collection.add(
            ids=ids[s:s + batch],
            documents=texts[s:s + batch],
            embeddings=embeddings[s:s + batch].tolist(),
            metadatas=metadatas[s:s + batch],
        )

    print(f"\nTotal chunks stored: {collection.count()}")
    for idx in [0, len(texts) // 2]:
        print(f"\n--- Example chunk: {metadatas[idx]} ---")
        print(f"[header] {headers[idx]}")
        print(texts[idx][:600] + ("..." if len(texts[idx]) > 600 else ""))


if __name__ == "__main__":
    main()
