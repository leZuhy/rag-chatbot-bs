"""Retrieve chunks, ask Claude with citations, print answer + sources + token usage.

Try it:  python src/answer.py "Koliko puta student može polagati ispit?"
"""

import sys

import anthropic
from dotenv import load_dotenv

from retrieve import hybrid_search

MODEL = "claude-haiku-5-5"  # cheapest current model; change here to try others
TOP_K = 5
# Enough context for follow-ups like "a koliko to košta?", without the prompt growing forever.
MAX_HISTORY_PAIRS = 3

# Kept byte-for-byte identical across calls: any change would break the prompt cache.
SYSTEM_PROMPT = """Ti si asistent koji odgovara na pitanja studenata isključivo na osnovu priloženih dokumenata (pravilnici, upute, nastavni planovi).

Pravila:
1. Odgovaraj SAMO na osnovu priloženih dokumenata. Ne koristi svoje opće znanje.
2. Ako odgovor nije u dokumentima, odgovori tačno: "Ne znam, ta informacija nije u dokumentima." i ništa više.
3. Ne izmišljaj brojeve, datume, rokove, nazive predmeta ni pravila.
4. Ako dokumenti daju samo djelimičan odgovor, reci šta piše i jasno navedi šta nedostaje.
5. Odgovaraj na bosanskom jeziku, kratko i jasno.
6. Tvoji raniji odgovori u razgovoru zasnovani su na dokumentima iz ranijih poruka, koji ti se više ne šalju. Ne dovodi te odgovore u pitanje i ne ispravljaj ih; za novo pitanje koristi samo trenutno priložene dokumente.
7. Svaku tvrdnju napiši samo jednom: ne ponavljaj istu rečenicu prvo svojim riječima, a zatim kao citat. Ne prepisuj oznake stavova u zagradama, npr. "(5)"."""

# Short, separate prompt for turning a follow-up into a self-contained search query.
REWRITE_PROMPT = """Na osnovu razgovora prepiši posljednje pitanje korisnika u samostalno pitanje koje se razumije bez razgovora (zamijeni "to", "taj", "on" i slično onim na šta se odnose).
Ako je pitanje već samostalno, vrati ga neizmijenjeno.
Vrati SAMO prepisano pitanje na bosanskom, bez objašnjenja."""


def build_documents(chunks):
    """One document block per chunk, so every citation points to exactly one source + page."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c["text"]},
            "title": f"{c['source']}, str. {c['page']}",
            "citations": {"enabled": True},
        }
        for c in chunks
    ]


def rewrite_question(client, question, past):
    """Turn a follow-up ("A u kojem roku to mora uraditi?") into a standalone search query.

    Search only sees the query text, not the chat, so "to" would match nothing useful.
    A tiny, cheap call: no thinking, small max_tokens.
    """
    convo = "\n".join(
        f"{'Korisnik' if m['role'] == 'user' else 'Asistent'}: {m['content']}" for m in past
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        thinking={"type": "disabled"},  # simple task, thinking would only add cost and latency
        system=REWRITE_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Razgovor:\n{convo}\n\nPosljednje pitanje: {question}",
        }],
    )
    rewritten = "".join(b.text for b in response.content if b.type == "text").strip()
    # Fall back to the original question if the rewrite failed for any reason.
    return rewritten or question


def answer(question, history=None, k=TOP_K):
    """Return (Claude response, retrieved chunks, query used for search).

    history: earlier turns as [{"role": "user"|"assistant", "content": "plain text"}, ...].
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    # Earlier turns go as plain text only. Their documents are not re-sent: it would
    # multiply tokens, and old answers already contain the facts that were found.
    past = (history or [])[-2 * MAX_HISTORY_PAIRS:]

    # Only follow-ups need rewriting; the first question saves the extra call.
    search_query = rewrite_question(client, question, past) if past else question
    chunks = hybrid_search(search_query, k=k)
    current = {
        "role": "user",
        # Documents go before the question; the question changes every call.
        "content": build_documents(chunks) + [{"type": "text", "text": question}],
    }

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        # cache_control marks the system prompt as a reusable prefix (cheaper repeated reads).
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        # Claude still answers the ORIGINAL question; the rewrite is only for search.
        messages=past + [current],
    )
    return response, chunks, search_query


def plain_text(response):
    """Answer text without [n] markers - what we send back as history."""
    return "".join(b.text for b in response.content if b.type == "text")


def parse_response(response, chunks):
    """Return (answer text with [n] markers, list of citations).

    With citations on, the answer comes back split into several text blocks;
    the cited ones carry a `citations` list. We append [n] after each cited block.
    The same quote (same text, PDF and page) gets one number and is listed once.
    """
    text_parts, citations = [], []
    number_of = {}  # (text, source, page) -> citation number
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        markers = []
        for cit in block.citations or []:
            # document_index maps back to our chunk, which holds the real source + page.
            chunk = chunks[cit.document_index]
            key = (cit.cited_text.strip(), chunk["source"], chunk["page"])
            if key not in number_of:
                citations.append({"cited_text": key[0], "source": key[1], "page": key[2]})
                number_of[key] = len(citations)
            if number_of[key] not in markers:
                markers.append(number_of[key])
        text_parts.extend(f" [{n}]" for n in markers)
    return "".join(text_parts), citations


def print_result(question, response, chunks):
    print(f"\n{'=' * 70}\nPitanje: {question}\n{'=' * 70}")

    if response.stop_reason == "refusal":
        print("Model je odbio odgovoriti.")
        return

    answer_text, citations = parse_response(response, chunks)
    print(answer_text)

    print("\nCitati:")
    for n, c in enumerate(citations, start=1):
        cited = c["cited_text"].replace("\n", " ")
        print(f'  [{n}] "{cited[:200]}"\n      -> {c["source"]}, str. {c["page"]}')
    if not citations:
        print("  (nema citata)")

    u = response.usage
    print(
        f"\nUsage: input={u.input_tokens}, output={u.output_tokens}, "
        f"cache_write={u.cache_creation_input_tokens}, cache_read={u.cache_read_input_tokens}"
    )


if __name__ == "__main__":
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "Koliko puta student može polagati ispit?"
    response, chunks, _ = answer(question)
    print_result(question, response, chunks)
