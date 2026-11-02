"""Generate a synthetic test set: pick chunks, ask Claude for one realistic question per chunk.

Run from the project root:  python eval/generate_testset.py
Output: eval/test_set.json (review it by hand before using it for evaluation!)
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

DB_DIR = "chroma_db"
COLLECTION = "docs"
OUT_FILE = Path("eval/test_set.json")
MODEL = "claude-haiku-5-5"

N_QUESTIONS = 25
SEED = 42  # fixed seed: same chunks every run, so the test set is reproducible
MIN_CHARS = 300  # shorter chunks are usually headings or fragments
MIN_WORDS = 50

PROMPT = """Ispod je odlomak iz dokumenta "{source}" (fakultetski pravilnici, upute, nastavni planovi).

Napiši JEDNO pitanje koje bi student stvarno postavio chatbotu, a na koje ovaj odlomak daje jasan odgovor.
Pravila:
- Pitanje mora biti razumljivo samo za sebe, bez spominjanja "odlomka", "teksta" ili "dokumenta".
- Ako se odlomak odnosi na neki predmet, uvijek navedi naziv predmeta u pitanju, nikad "ovaj predmet" ili "na predmetu".
- Ne prepisuj rečenice iz odlomka doslovno; pitaj svojim riječima, kako bi pitao student.
- Odgovor mora biti u potpunosti sadržan u odlomku. Ne dodaji ništa izvan njega.
- expected_answer: kratak i tačan odgovor (1-2 rečenice), na bosanskom.

Odlomak:
\"\"\"
{text}
\"\"\""""

# Structured outputs: the API guarantees valid JSON in exactly this shape,
# so we never have to parse or repair half-broken JSON.
SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "expected_answer": {"type": "string"},
    },
    "required": ["question", "expected_answer"],
    "additionalProperties": False,
}


def is_useful(text):
    """Skip chunks with almost no content: tables of contents, title pages, fragments."""
    if len(text) < MIN_CHARS or len(text.split()) < MIN_WORDS:
        return False
    if re.search(r"\.{5,}|…{2,}", text):  # dot leaders ("Član 5 ........ 12") = table of contents
        return False
    return True


def pick_chunks(data):
    """Round-robin across PDFs, so small documents are represented, not just the biggest one."""
    rng = random.Random(SEED)
    by_source = defaultdict(list)
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if is_useful(text):
            by_source[meta["source"]].append({"chunk_id": cid, "text": text, **meta})

    for chunks in by_source.values():
        rng.shuffle(chunks)

    picked = []
    sources = sorted(by_source)  # sorted, so order doesn't depend on dict/DB order
    while len(picked) < N_QUESTIONS and any(by_source.values()):
        for src in sources:
            if by_source[src] and len(picked) < N_QUESTIONS:
                picked.append(by_source[src].pop())
    return picked


def generate_question(client, chunk):
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(source=chunk["source"], text=chunk["text"])}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main():
    load_dotenv()
    client = anthropic.Anthropic()
    data = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION).get(
        include=["documents", "metadatas"]
    )
    print(f"Chunks in DB: {len(data['ids'])}")

    chunks = pick_chunks(data)
    test_set = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            qa = generate_question(client, chunk)
        except Exception as e:  # one failed call shouldn't kill the whole run
            print(f"  skip {chunk['chunk_id']}: {e}")
            continue
        test_set.append({
            "id": f"gen_{i:02d}",
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "type": "generated",
        })
        print(f"  [{i}/{len(chunks)}] {chunk['source']} str. {chunk['page']}")

    OUT_FILE.write_text(json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGenerated {len(test_set)} questions -> {OUT_FILE}")
    per_source = defaultdict(int)
    for q in test_set:
        per_source[q["source"]] += 1
    for src, n in sorted(per_source.items()):
        print(f"  {n:2d}  {src}")

    for q in random.Random(SEED).sample(test_set, 3):
        print(f"\n[{q['id']}] {q['question']}\n  -> {q['expected_answer']}\n  ({q['source']}, str. {q['page']})")


if __name__ == "__main__":
    main()
