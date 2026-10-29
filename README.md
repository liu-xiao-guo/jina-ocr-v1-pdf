# Jina OCR + Elasticsearch Hybrid Search & RAG (Python / Streamlit)

Upload a PDF, convert every page to Markdown with **Jina OCR v1** (served by the
Elastic Inference Service), index the result into Elasticsearch, then run
**hybrid search** (BM25 + semantic, fused with RRF) and **RAG chat** over it.

This is a Python/Streamlit port of the `jina-ocr-v1-hybrid-rag-elasticsearch`
project, without the product-catalog seed data.

## How it works

1. **Render:** each PDF page is rendered to a PNG with PyMuPDF (2x scale).
2. **OCR:** each PNG is sent to the `.jina-ocr-v1-chat_completion` inference
   endpoint (parallel, order preserved). Tables come back as HTML tables.
3. **Stitch:** pages are joined into one Markdown document with
   `<!-- page N -->` markers so answers can cite page numbers.
4. **Index:** an index named after the file is created with
   `content` (`text`, `copy_to` -> `content_semantic`) and `content_semantic`
   (`semantic_text`). Elasticsearch chunks and embeds the text itself.
   The file is also recorded in a `doc-registry` index that drives the sidebar.
5. **Search:** `hybrid` (RRF over a `match` and a `semantic` query), `lexical`
   or `semantic`, with highlighted passages and page numbers.
6. **RAG:** the top passages are retrieved with hybrid search and passed to the
   chat LLM (`LLM_INFERENCE_ID`), which streams a cited answer.

## Requirements

- Python 3.10+
- An Elasticsearch deployment (Serverless or 9.x) with the Elastic Inference
  Service, an API key with index read/write and inference privileges, and
  access to the OCR and chat endpoints below.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env (see below)
streamlit run app.py
```

### Configuration (`.env`)

| Variable | Purpose | Default |
| --- | --- | --- |
| `ELASTICSEARCH_URL` | Cluster URL | required |
| `ELASTICSEARCH_API_KEY` | API key | required |
| `OCR_INFERENCE_ID` | OCR endpoint | `.jina-ocr-v1-chat_completion` |
| `OCR_CONCURRENCY` | Parallel OCR calls per upload (lower it on HTTP 429) | `4` |
| `LLM_INFERENCE_ID` | Chat LLM for RAG answers; use a `chat_completion` endpoint | auto-detected if unset |
| `ELASTICSEARCH_INSECURE_TLS` | `true` skips certificate checks (local self-signed clusters only) | off |

## Using the app

- **Upload & index:** choose a PDF and click *Parse and index*. Re-uploading a
  file with the same name replaces its index.
- **Hybrid search:** type a query, pick a mode, and compare results.
- **Ask (RAG):** ask questions about the selected documents.
- **Sidebar:** lists indexed documents (pages, size, ingest time), lets you
  delete one, and sets the documents used for search and chat.

## Project layout

```
app.py            Streamlit UI
es_pipeline.py    PDF rendering, OCR, indexing, search, RAG
requirements.txt  Python dependencies
docs/             Sample PDFs
```

## Notes

- The default `content` analyzer is language-agnostic, so keyword matching on
  Chinese text is character-level; the semantic half of hybrid search carries
  most of the relevance. A `smartcn`/`ik` analyzer can improve this if the
  plugin is available on your cluster.
- Large PDFs take time and tokens: one OCR call is made per page.

## License

Released under the [MIT License](LICENSE).
