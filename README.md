
# DocuQuery

**Production-grade Retrieval-Augmented Generation (RAG) system for document Q&A** — built as a portfolio project demonstrating end-to-end ML engineering: hybrid search, agentic orchestration, security hardening, and observability.

Ask questions across PDFs, DOCX, and PPTX files. Get cited, grounded answers with confidence scores, source images, and hallucination risk indicators — all running on a free-tier stack.

---

## What it does

Upload a document, ask a question, get an answer with exact page citations — not a black box. Every response shows you *why* it's confident (or not), *where* the information came from, and *whether* it might be hallucinating.

```
Q: What is LangGraph?
→ Answer cites Learning LangChain.pdf, page 165
→ Confidence: high · Docs searched: 1 · Hallucination risk: low
→ Shows the actual diagram from that page
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Chainlit   │────▶│   FastAPI    │────▶│  Pinecone   │
│     UI      │◀────│   Backend    │◀────│ (hybrid idx)│
└─────────────┘     └──────┬───────┘     └─────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          Cross-Encoder  LangGraph    Groq
            Reranker       Agent     LLaMA 3.3
```

**Retrieval pipeline:** query → dense (BGE embeddings) + sparse (BM25) hybrid search → cross-encoder reranking → sentence-window context expansion → LLM generation → background RAGAS-style evaluation.

---

## Core features

### Retrieval & Search
- **Hybrid search** — dense embeddings (`BAAI/bge-small-en-v1.5`) combined with BM25 sparse retrieval via Pinecone, weighted by an auto-detected alpha (keyword vs. semantic)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-12-v2` re-scores top candidates for precision
- **Sentence-window retrieval** — expands each matched chunk with neighboring context before sending to the LLM, fixing fragmented answers
- **Full-corpus BM25 refit** — sparse index refits on the entire corpus on every upload, not just new chunks
- **Query decomposition** — complex questions are split into sub-queries and retrieved in parallel
- **Guaranteed document diversity** — multi-doc queries pin results across all selected documents, not just the top scorer

### Intelligence Layer
- **LangGraph agentic routing** — a state machine decides between direct answers, simple RAG, and multi-step agentic retrieval with self-critique retry on low confidence
- **Intent detection** — dedicated handlers for summarization (single/all docs), image requests, conversation history queries, and factual lookups (author/publisher) that bypass semantic drift
- **Conversation memory** — follow-up questions resolve pronouns against prior context

### Trust & Safety
- **Hallucination risk scoring** — synchronous heuristic (confidence + source count) shown immediately, plus async RAGAS-style faithfulness/relevancy scoring via LLM-as-judge
- **Prompt injection detection** — weighted pattern scanner flags risky queries; the real defense is a hardened system prompt that treats all document/history content as data, never instructions
- **Source citations** — every answer traces back to exact filename + page number

### Production Hardening
- **Rate limiting** — `slowapi` enforced at the ASGI layer (20 req/min queries, 5/hour uploads)
- **Security headers** — CSP, X-Frame-Options, XSS protection on every response
- **Input sanitization** — control-character stripping, length caps, path-traversal guards
- **Password auth** — Chainlit session-based login, no public access without credentials

### Developer Experience
- **Developer mode** — toggle in-app to see live server logs, injection risk scores, token usage, and aggregate stats
- **Token usage meter** — per-query and cumulative session tracking
- **Image extraction** — PyMuPDF pulls diagrams/charts from PDFs, filtered against logos/icons via entropy + aspect-ratio heuristics, surfaced inline with answers
- **Quality dashboard** — faithfulness/relevancy bars, thumbs up/down feedback aggregation

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| UI | Chainlit |
| Vector DB | Pinecone (serverless, hybrid dense+sparse) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| LLM | Groq (LLaMA 3.3 70B) via LiteLLM |
| Agent orchestration | LangGraph |
| Chunking | Chonkie (semantic chunking) |
| Document parsing | pypdf, Docling (DOCX/PPTX), PyMuPDF (images) |
| Rate limiting | slowapi |
| Testing | pytest, pytest-asyncio |
| Containerization | Docker, Docker Compose |
| Deployment | Render (free tier) |

---

## Quick start (local)

```bash
git clone https://github.com/<your-username>/docuquery.git
cd docuquery
pip install -r requirements.txt

# Set up .env (see .env.example)
cp .env.example .env
# Add PINECONE_API_KEY, GROQ_API_KEY, etc.

uvicorn main:app --reload --port 8000     # Terminal 1
chainlit run app.py --port 8080           # Terminal 2
```

Visit `http://localhost:8080`.

## Quick start (Docker)

```bash
docker compose up --build
```

API at `localhost:8000/docs`, UI at `localhost:8080`.

---

## Testing

```bash
python run_tests.py            # all suites
python run_tests.py security   # vulnerability + injection + hallucination probes
python run_tests.py langgraph  # agentic endpoint + image extraction
```

23+ automated security tests covering prompt injection resistance, rate limiting, path traversal, and hallucination prevention.

---

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /upload/` | Index a document |
| `POST /ask/` | Standard RAG query |
| `POST /ask/stream` | Streaming RAG query (SSE) |
| `POST /ask/agentic` | LangGraph-orchestrated query with self-critique |
| `GET /documents/` | List indexed documents |
| `POST /feedback/` | Submit thumbs up/down |
| `GET /eval-log/` | RAGAS-style quality scores |
| `GET /dev/stats/` | Developer mode aggregate stats |
| `POST /rebuild-images/` | Re-extract images from existing PDFs |




## License

MIT
