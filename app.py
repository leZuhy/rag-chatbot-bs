"""Minimal Streamlit chat UI.  Run from the project root:  streamlit run app.py"""

import sys

import streamlit as st
from dotenv import load_dotenv

# Our modules live in src/ and import each other by plain name (from retrieve import ...),
# so we put src/ on the path instead of turning it into a package.
sys.path.insert(0, "src")
from answer import answer, parse_response, plain_text  # noqa: E402

load_dotenv()

st.set_page_config(page_title="RAG chatbot - dokumenti", page_icon="📄")
st.title("📄 Pitaj dokumente")
st.caption("Odgovori samo iz priloženih PDF-ova, uz izvor (dokument + strana).")

# Streamlit re-runs this whole script on every interaction, so the chat
# must live in session_state to survive between runs.
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_assistant(msg):
    """Render one assistant message with its sources and debug info."""
    st.markdown(msg["content"])
    if msg.get("citations"):
        with st.expander(f"Izvori ({len(msg['citations'])})"):
            for n, c in enumerate(msg["citations"], start=1):
                st.markdown(f"**[{n}] {c['source']}, str. {c['page']}**")
                st.caption(f"„{c['cited_text']}“")
    # Debug view: was a wrong answer caused by retrieval or by generation?
    if msg.get("chunks"):
        with st.expander("Pronađeni chunkovi i tokeni"):
            st.markdown(f"**Upit za pretragu:** {msg['search_query']}")
            for c in msg["chunks"]:
                st.markdown(f"**{c['source']}, str. {c['page']}** (RRF skor {c['score']:.4f})")
                st.text(c["text"][:800])
            st.write(msg["usage"])


# Re-draw the whole conversation on each run.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_assistant(msg)
        else:
            st.markdown(msg["content"])

if question := st.chat_input("Postavi pitanje o dokumentima..."):
    with st.chat_message("user"):
        st.markdown(question)

    # Claude gets plain text of earlier turns (no [n] markers, no citations).
    history = [
        {"role": m["role"], "content": m.get("plain", m["content"])}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        # retrieve.py caches the model and indexes, so only the first question is slow.
        with st.spinner("Tražim u dokumentima..."):
            response, chunks, search_query = answer(question, history)

        if response.stop_reason == "refusal":
            text = "Model je odbio odgovoriti na ovo pitanje."
            msg = {"role": "assistant", "content": text, "plain": text}
        else:
            text, citations = parse_response(response, chunks)
            u = response.usage
            msg = {
                "role": "assistant",
                "content": text,
                "plain": plain_text(response),
                "citations": citations,
                "chunks": chunks,
                "search_query": search_query,
                "usage": {
                    "input": u.input_tokens,
                    "output": u.output_tokens,
                    "cache_write": u.cache_creation_input_tokens,
                    "cache_read": u.cache_read_input_tokens,
                },
            }
        show_assistant(msg)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(msg)
