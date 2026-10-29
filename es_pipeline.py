"""Elasticsearch helpers: OCR a PDF with Jina OCR (via EIS), index it, hybrid search, RAG.

Pipeline: render PDF pages to PNG -> OCR each page through the EIS
`.jina-ocr-v1-chat_completion` endpoint (parallel, order preserved) -> stitch
Markdown with <!-- page N --> markers -> create a hybrid index (text field
copy_to semantic_text) -> index the document.
"""

import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pymupdf
import requests
from dotenv import load_dotenv

load_dotenv()

ES_URL = os.getenv("ELASTICSEARCH_URL", "").rstrip("/")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
OCR_INFERENCE_ID = os.getenv("OCR_INFERENCE_ID", ".jina-ocr-v1-chat_completion")
LLM_INFERENCE_ID = os.getenv("LLM_INFERENCE_ID", "")
CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))
VERIFY_TLS = os.getenv("ELASTICSEARCH_INSECURE_TLS", "").lower() not in ("1", "true", "yes")
REGISTRY_INDEX = "doc-registry"

OCR_PROMPT = (
    "Transcribe the provided document image into a clean Markdown format, "
    "preserving the natural reading order. Represent tables as HTML tables."
)

if not VERIFY_TLS:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"Authorization": f"ApiKey {ES_API_KEY}", "Content-Type": "application/json"}


def _req(method, path, **kwargs):
    return requests.request(method, f"{ES_URL}/{path.lstrip('/')}", headers=HEADERS, verify=VERIFY_TLS, **kwargs)


def slugify_index_name(filename: str) -> str:
    base = re.sub(r"\.[^.]+$", "", filename)
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:100]
    return slug or "uploaded-document"


def display_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", re.sub(r"\.[^.]+$", "", title or ""))).strip()


def render_pdf(pdf_bytes: bytes, scale: float = 2.0) -> list[bytes]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png") for page in doc]


def _stream_chat(inference_id: str, messages: list):
    """Yield content deltas from the EIS chat_completion SSE stream."""
    resp = _req(
        "POST",
        f"_inference/chat_completion/{inference_id}/_stream",
        data=json.dumps({"messages": messages}),
        stream=True,
    )
    if not resp.ok:
        raise RuntimeError(f"{inference_id} call failed: {resp.status_code} {resp.text}")
    # SSE responses carry no charset, so requests would guess Latin-1 and garble
    # non-ASCII text (e.g. Chinese). Decode the raw bytes as UTF-8 ourselves.
    for raw in resp.iter_lines():
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError):
            continue
        if isinstance(delta, str):
            yield delta


def ocr_page(png: bytes) -> str:
    b64 = base64.b64encode(png).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    return "".join(_stream_chat(OCR_INFERENCE_ID, messages))


def ocr_pages(pages: list[bytes], on_page=None) -> list[str]:
    """OCR all pages in parallel, preserving order. on_page(completed, total) per page."""
    results = [""] * len(pages)
    with ThreadPoolExecutor(max_workers=max(1, min(CONCURRENCY, len(pages)))) as pool:
        futures = {pool.submit(ocr_page, p): i for i, p in enumerate(pages)}
        for done, fut in enumerate(as_completed(futures), start=1):
            results[futures[fut]] = fut.result()
            if on_page:
                on_page(done, len(pages))
    return results


def create_hybrid_index(index: str):
    _req("DELETE", index)  # overwrite on re-upload
    resp = _req(
        "PUT",
        index,
        data=json.dumps(
            {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "copy_to": "content_semantic"},
                        "content_semantic": {"type": "semantic_text"},
                        "title": {"type": "keyword"},
                        "source_file": {"type": "keyword"},
                        "page_count": {"type": "integer"},
                        "ingested_at": {"type": "date"},
                    }
                }
            }
        ),
    )
    if not resp.ok:
        raise RuntimeError(f"Index creation failed: {resp.text}")


def index_document(index: str, filename: str, pages_md: list[str]) -> dict:
    markdown = "\n\n".join(f"<!-- page {i} -->\n{md}" for i, md in enumerate(pages_md, start=1))
    create_hybrid_index(index)
    doc = {
        "content": markdown,
        "title": filename,
        "source_file": filename,
        "page_count": len(pages_md),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = _req("POST", f"{index}/_doc?refresh=wait_for", data=json.dumps(doc))
    if not resp.ok:
        raise RuntimeError(f"Indexing failed: {resp.text}")
    _req(
        "PUT",
        f"{REGISTRY_INDEX}/_doc/{index}?refresh=true",
        data=json.dumps(
            {
                "index": index,
                "title": filename,
                "page_count": len(pages_md),
                "chars": len(markdown),
                "ingested_at": doc["ingested_at"],
            }
        ),
    )
    return {"index": index, "markdown": markdown, "pages": len(pages_md)}


def delete_document(index: str):
    """Remove the document's index and its registry entry."""
    _req("DELETE", index)
    _req("DELETE", f"{REGISTRY_INDEX}/_doc/{index}?refresh=true")


def list_documents() -> list[dict]:
    try:
        resp = _req(
            "POST",
            f"{REGISTRY_INDEX}/_search",
            data=json.dumps({"size": 50, "sort": [{"ingested_at": "desc"}], "query": {"match_all": {}}}),
        )
        if not resp.ok:
            return []
        return [h["_source"] for h in resp.json()["hits"]["hits"]]
    except requests.RequestException:
        return []


def _extract_page(fragment: str):
    m = re.search(r"<!--\s*page\s+(\d+)\s*-->", fragment)
    return m.group(1) if m else None


def search(indices: list[str], query: str, mode: str = "hybrid", size: int = 5) -> list[dict]:
    """mode: 'hybrid' (BM25 + semantic fused with RRF), 'lexical' or 'semantic'.

    Returns one result per matching document with highlighted passages.
    """
    lexical = {"standard": {"query": {"match": {"content": query}}}}
    semantic = {"standard": {"query": {"semantic": {"field": "content_semantic", "query": query}}}}
    if mode == "lexical":
        retriever = lexical
    elif mode == "semantic":
        retriever = semantic
    else:
        retriever = {"rrf": {"rank_window_size": 30, "retrievers": [lexical, semantic]}}

    resp = _req(
        "POST",
        f"{','.join(indices)}/_search?ignore_unavailable=true",
        data=json.dumps(
            {
                "retriever": retriever,
                "size": size,
                "_source": ["title", "page_count"],
                "highlight": {
                    "fields": {
                        "content_semantic": {"number_of_fragments": 4, "order": "score"},
                        "content": {"fragment_size": 350, "number_of_fragments": 2, "order": "score"},
                    }
                },
            }
        ),
    )
    if not resp.ok:
        raise RuntimeError(f"Search failed: {resp.text}")

    results = []
    for hit in resp.json()["hits"]["hits"]:
        passages, seen = [], set()
        for kind, frags in (
            ("semantic", hit.get("highlight", {}).get("content_semantic", [])),
            ("lexical", hit.get("highlight", {}).get("content", [])),
        ):
            for frag in frags:
                if kind == "lexical":
                    frag = frag.replace("<em>", "**").replace("</em>", "**")
                key = frag[:80]
                if key not in seen:
                    seen.add(key)
                    passages.append({"text": frag, "page": _extract_page(frag), "kind": kind})
        results.append(
            {
                "doc": display_title(hit["_source"].get("title", hit["_index"])),
                "index": hit["_index"],
                "score": hit["_score"],
                "passages": passages,
            }
        )
    return results


def resolve_llm() -> str:
    if LLM_INFERENCE_ID:
        return LLM_INFERENCE_ID
    resp = _req("GET", "_inference/chat_completion/_all")
    resp.raise_for_status()
    for ep in resp.json().get("endpoints", []):
        if "ocr" not in ep["inference_id"]:
            return ep["inference_id"]
    raise RuntimeError("No chat LLM endpoint found. Set LLM_INFERENCE_ID in .env")


def build_prompt(question: str, passages: list[dict]) -> str:
    sources = "\n\n".join(
        f"[Source: {p['doc']}{', p. ' + p['page'] if p['page'] else ''}]\n{p['text']}" for p in passages
    )
    return "\n".join(
        [
            "You are a helpful assistant answering questions about uploaded documents.",
            "Answer the question using ONLY the reference sources below.",
            "Cite sources inline by document name and page, for example (Native and Adaptive Plants, p. 50). "
            'Never use the word "passage".',
            "If the sources do not answer the question, say so plainly. Be concise and practical.",
            "",
            "REFERENCE SOURCES:",
            sources or "(none)",
            "",
            f"QUESTION: {question}",
        ]
    )


def rag_answer(indices: list[str], question: str):
    """Return (passages, token generator). Retrieval is hybrid; the LLM is the EIS chat endpoint."""
    hits = search(indices, question, mode="hybrid", size=3)
    passages = [{**p, "doc": h["doc"]} for h in hits for p in h["passages"]][:5]
    if not passages:
        return passages, iter(["Nothing in the selected documents matched that question."])
    messages = [{"role": "user", "content": build_prompt(question, passages)}]
    return passages, _stream_chat(resolve_llm(), messages)
