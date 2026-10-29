"""Streamlit UI: upload PDF -> Jina OCR -> Markdown -> Elasticsearch -> hybrid search + RAG.

Run with:  streamlit run app.py
"""

import streamlit as st

import es_pipeline as ep

st.set_page_config(page_title="Jina OCR + Hybrid RAG", page_icon="📄", layout="wide")
st.title("📄 PDF → Jina OCR → Elasticsearch hybrid search + RAG")

if not ep.ES_URL or not ep.ES_API_KEY:
    st.error("Set ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY in .env")
    st.stop()

docs = ep.list_documents()
doc_indices = [d["index"] for d in docs]

with st.sidebar:
    st.header("Search / chat scope")
    scope = st.multiselect("Documents to search", doc_indices, default=doc_indices, label_visibility="collapsed")
    st.divider()
    st.header("Indexed documents")
    if docs:
        for d in docs:
            with st.container(border=True):
                st.markdown(f"**{d['title']}**  \n`{d['index']}`")
                size = f" · {d['chars']:,} chars" if d.get("chars") else ""
                st.caption(f"✅ indexed · {d['page_count']} pages{size}  \n{d['ingested_at'][:16].replace('T', ' ')} UTC")
                if st.button("Delete", key=f"del-{d['index']}"):
                    ep.delete_document(d["index"])
                    st.rerun()
    else:
        st.caption("Nothing indexed yet. Upload a PDF first.")

tab_upload, tab_search, tab_chat = st.tabs(["1 · Upload & index", "2 · Hybrid search", "3 · Ask (RAG)"])

# ───────────────────────── upload ─────────────────────────
with tab_upload:
    file = st.file_uploader("Upload a PDF", type=["pdf"])
    if file and st.button("Parse and index", type="primary"):
        index = ep.slugify_index_name(file.name)
        try:
            with st.status(f"Processing {file.name}", expanded=True) as status:
                st.write("Rendering PDF pages to images…")
                pages = ep.render_pdf(file.getvalue())
                st.write(f"{len(pages)} pages rendered. Running Jina OCR…")
                bar = st.progress(0.0)
                md_pages = ep.ocr_pages(pages, on_page=lambda done, total: bar.progress(done / total))
                st.write(f"Creating hybrid index `{index}` and indexing (Elasticsearch chunks + embeds)…")
                result = ep.index_document(index, file.name, md_pages)
                status.update(label=f"Indexed `{index}`: {result['pages']} pages", state="complete")
            st.session_state["last_markdown"] = result["markdown"]
            st.rerun()
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the user
            st.error(str(e))

    if "last_markdown" in st.session_state:
        st.success("Indexed. Use the Search and Ask tabs.")
        with st.expander("Extracted Markdown"):
            md = st.session_state["last_markdown"]
            st.download_button("Download .md", md, file_name="document.md")
            st.markdown(md, unsafe_allow_html=True)

# ───────────────────────── search ─────────────────────────
with tab_search:
    c1, c2 = st.columns([4, 1])
    query = c1.text_input("Search query", key="q")
    mode = c2.selectbox("Mode", ["hybrid", "lexical", "semantic"])
    if query:
        if not scope:
            st.info("Select at least one document in the sidebar.")
        else:
            try:
                for r in ep.search(scope, query, mode):
                    with st.container(border=True):
                        st.subheader(r["doc"])
                        st.caption(f"score {r['score']:.4f} · {mode}")
                        for p in r["passages"]:
                            page = f"p. {p['page']} · " if p["page"] else ""
                            st.markdown(f"_{page}{p['kind']}_")
                            st.markdown(p["text"].replace("<!--", "").replace("-->", ""), unsafe_allow_html=False)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

# ───────────────────────── RAG chat ─────────────────────────
with tab_chat:
    history = st.session_state.setdefault("history", [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if question := st.chat_input("Ask a question about the indexed documents"):
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if not scope:
                answer = "Select at least one document in the sidebar."
                st.markdown(answer)
            else:
                try:
                    passages, stream = ep.rag_answer(scope, question)
                    with st.expander(f"Sources ({len(passages)})"):
                        for p in passages:
                            st.markdown(f"**{p['doc']}**{', p. ' + p['page'] if p['page'] else ''} _({p['kind']})_")
                            st.caption(p["text"][:400])
                    answer = st.write_stream(stream)
                except Exception as e:  # noqa: BLE001
                    answer = f"Error: {e}"
                    st.error(answer)
        history.append({"role": "assistant", "content": answer})
