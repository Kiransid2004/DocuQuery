"""
DocuQuery v4.4 — Production FastAPI Backend

New in this version:
  - POST /rebuild-images/ — re-scan PDFs on disk and re-extract images
    using current image_extractor.py thresholds (recovery tool).
  - GET /dev/logs/ — last N log lines for developer mode panel.
  - GET /dev/stats/ — aggregate stats (tokens, evals, feedback, doc health).
  - DevLogHandler — captures all logging into an in-memory ring buffer.

Security (unchanged from previous version):
  - slowapi rate limiting via SlowAPIMiddleware
  - security headers middleware with RateLimitExceeded re-raise
  - input sanitisation, upload size guard, secrets from env only
"""

from fastapi import FastAPI, UploadFile, HTTPException, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, validator
from typing import List, Optional
from collections import deque
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from ingest import load_document_with_pages, chunk_pages, analyze_document_health
from utils import (
    ensure_collection,
    upsert_chunks,
    list_documents,
    delete_document,
    register_document,
    record_feedback,
    load_feedback,
    load_eval_log,
    warmup_bm25,
)
from image_extractor import (
    extract_images_from_pdf,
    get_images_for_page,
    get_images_for_doc,
    get_image_by_id,
    delete_images_for_doc,
    rebuild_image_index,
)
from query import (
    generate_answer,
    generate_answer_stream,
    generate_suggested_questions,
    generate_document_summary,
)
from agent_graph import run_agentic_query
from dotenv import load_dotenv
import shutil
import os
import re
import hashlib
import json
import asyncio
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# ── Rate Limiter ───────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Developer mode: in-memory log ring buffer ───────────────────────────────────

_dev_log_buffer = deque(maxlen=200)

class DevLogHandler(logging.Handler):
    """Captures log records into a ring buffer for GET /dev/logs/."""
    def emit(self, record):
        try:
            _dev_log_buffer.append({
                "timestamp": self.formatTime(record, "%H:%M:%S"),
                "level":     record.levelname,
                "logger":    record.name,
                "message":   record.getMessage()[:300],
            })
        except Exception:
            pass

_dev_handler = DevLogHandler()
_dev_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_dev_handler)

# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    logger.info("Preloading embedding model...")
    from utils import get_model
    get_model()
    logger.info("Preloading reranker...")
    from reranker import get_reranker
    get_reranker()
    logger.info("Warming up BM25 on full corpus...")
    warmup_bm25()
    docs = list_documents()
    logger.info(f"DocuQuery ready — {len(docs)} document(s) already indexed.")
    yield

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DocuQuery",
    description = "RAG-powered document Q&A — production grade.",
    version     = "4.4.0",
    lifespan    = lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ALLOWED_ORIGINS,
    allow_methods = ["GET", "POST", "DELETE"],
    allow_headers = ["*"],
)

# ── Security Headers Middleware ────────────────────────────────────────────────

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    try:
        response = await call_next(request)
    except RateLimitExceeded:
        raise
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"]          = "no-store"
    return response

# ── Upload size guard ──────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# ── Input sanitisation ─────────────────────────────────────────────────────────

def sanitise_query(query: str) -> str:
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    return query[:2000].strip()

def sanitise_filename(name: str) -> str:
    return re.sub(r"[^\w\s\-.]", "", name)[:200].strip()

# ── Request models ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query:   str
    doc_ids: list[str]       = []
    alpha:   Optional[float] = None
    history: list[dict]      = []

    @validator("query")
    def query_not_empty(cls, v):
        v = sanitise_query(v)
        if not v:
            raise ValueError("Query cannot be empty.")
        return v

    @validator("history")
    def limit_history(cls, v):
        return v[-20:] if v else []

class CompareRequest(BaseModel):
    query:    str
    doc_id_a: str
    doc_id_b: str

class AnalyseRequest(BaseModel):
    doc_id:         str
    filename:       str
    chunks_preview: list[str] = []

class FeedbackRequest(BaseModel):
    query:   str
    answer:  str
    rating:  str
    sources: list[dict] = []

    @validator("rating")
    def valid_rating(cls, v):
        if v not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        return v

# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "DocuQuery API is running", "version": "4.4.0", "docs": "/docs"}

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    docs     = list_documents()
    feedback = load_feedback()
    evals    = load_eval_log()

    faith_scores = [e["faithfulness"]     for e in evals if e.get("faithfulness")     is not None]
    relev_scores = [e["answer_relevancy"] for e in evals if e.get("answer_relevancy") is not None]

    return {
        "status":            "ok",
        "version":           "4.4.0",
        "indexed_documents": len(docs),
        "documents":         docs,
        "feedback": {
            "total":       len(feedback),
            "thumbs_up":   sum(1 for f in feedback if f.get("rating") == "up"),
            "thumbs_down": sum(1 for f in feedback if f.get("rating") == "down"),
        },
        "eval_quality": {
            "total_evals":          len(evals),
            "avg_faithfulness":     round(sum(faith_scores)/len(faith_scores), 2) if faith_scores else None,
            "avg_answer_relevancy": round(sum(relev_scores)/len(relev_scores), 2) if relev_scores else None,
        },
    }

# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/upload/")
@limiter.limit(f"{os.getenv('RATE_LIMIT_UPLOAD_PER_HOUR', '5')}/hour")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    allowed  = (".pdf", ".docx", ".pptx", ".txt", ".md")
    results, errors, skipped = [], [], []
    existing = {d["doc_id"]: d for d in list_documents()}

    for file in files:
        safe_name = sanitise_filename(file.filename)
        doc_id    = hashlib.md5(safe_name.encode()).hexdigest()[:8]

        if doc_id in existing:
            skipped.append({"filename": safe_name, "doc_id": doc_id, "reason": "Already indexed"})
            continue

        if not safe_name.lower().endswith(allowed):
            errors.append({"filename": safe_name, "error": f"Unsupported type. Allowed: {allowed}"})
            continue

        try:
            os.makedirs("data", exist_ok=True)
            path = f"data/{safe_name}"

            content = await file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                errors.append({"filename": safe_name, "error": "File exceeds 50 MB limit"})
                continue
            with open(path, "wb") as f:
                f.write(content)

            file_size_mb = len(content) / (1024 * 1024)
            logger.info(f"Processing: {safe_name} ({file_size_mb:.1f} MB)")

            pages  = load_document_with_pages(path)
            chunks = chunk_pages(pages, filename=safe_name)

            if not chunks:
                errors.append({"filename": safe_name, "error": "Could not extract text."})
                os.remove(path)
                continue

            health = analyze_document_health(chunks)
            upsert_chunks(chunks, doc_id, safe_name)
            register_document(doc_id, safe_name, len(chunks), health)

            images = extract_images_from_pdf(path, doc_id, safe_name)

            results.append({
                "message":          "Processed successfully",
                "doc_id":           doc_id,
                "filename":         safe_name,
                "file_size_mb":     round(file_size_mb, 1),
                "chunks":           len(chunks),
                "pages_detected":   len(pages),
                "document_health":  health["health"],
                "health_note":      health["note"],
                "chunks_preview":   [c.get("raw_text", c["text"]) for c in chunks[:10]],
                "images_extracted": len(images),
            })
            logger.info(f"Done: {safe_name} — {len(chunks)} chunks / {len(pages)} pages / {len(images)} images")

        except Exception as e:
            logger.error(f"Upload error for {safe_name}: {e}")
            errors.append({"filename": safe_name, "error": str(e)})

    return {
        "processed":     len(results),
        "skipped":       len(skipped),
        "failed":        len(errors),
        "results":       results,
        "skipped_docs":  skipped,
        "errors":        errors,
        "total_indexed": len(list_documents()),
    }

# ── Post-upload analysis ───────────────────────────────────────────────────────

@app.post("/upload/analyse")
@limiter.limit("30/minute")
async def analyse_document(request: Request, req: AnalyseRequest):
    if not req.doc_id or not req.filename:
        raise HTTPException(400, "doc_id and filename required")
    summary, questions = await asyncio.gather(
        generate_document_summary(req.filename, req.chunks_preview),
        generate_suggested_questions(req.filename, req.chunks_preview),
    )
    return {"doc_id": req.doc_id, "filename": req.filename, "summary": summary, "questions": questions}

# ── List / delete documents ────────────────────────────────────────────────────

@app.get("/documents/")
async def get_documents():
    docs = list_documents()
    return {"total": len(docs), "documents": docs}

@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    if not delete_document(doc_id):
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    delete_images_for_doc(doc_id)
    return {"message": f"Document '{doc_id}' deleted.", "total_indexed": len(list_documents())}

# ── Rebuild image index (NEW — recovery tool) ───────────────────────────────────

@app.post("/rebuild-images/")
@limiter.limit("3/hour")
async def rebuild_images(request: Request):
    """
    Re-scan every PDF already on disk (data/*.pdf) using the registry,
    and re-extract images with the CURRENT thresholds in image_extractor.py.

    Use this when:
      - image_index.json was lost or reset
      - You changed filtering thresholds and want to re-apply them
        to documents already indexed (no need to re-upload via UI)
    """
    try:
        result = rebuild_image_index(data_dir="data")
        logger.info(f"Image index rebuilt: {result['rebuilt']} document(s)")
        return {
            "message":           f"Rebuilt image index for {result['rebuilt']} document(s)",
            "documents_rebuilt": result["rebuilt"],
            "details":           result["documents"],
        }
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(500, f"Rebuild failed: {e}")

# ── Ask ────────────────────────────────────────────────────────────────────────

@app.post("/ask/")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '20')}/minute")
async def ask(request: Request, req: AskRequest):
    all_docs = list_documents()
    try:
        return await generate_answer(
            query    = req.query,
            doc_ids  = req.doc_ids or None,
            alpha    = req.alpha,
            history  = req.history,
            all_docs = all_docs,
        )
    except Exception as e:
        logger.error(f"/ask/ failed: {e}", exc_info=True)
        return {
            "answer": "## Answer\nAn error occurred while processing your question. Please try rephrasing.\n\n## References\nNone.",
            "sources": [], "confidence": "low", "avg_relevance_score": 0.0,
            "docs_searched": 0, "sub_queries_used": [req.query],
            "alpha_used": req.alpha, "alpha_mode": "error",
            "search_mode": "error", "hallucination_risk": "high",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

@app.post("/ask/agentic")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '20')}/minute")
async def ask_agentic(request: Request, req: AskRequest):
    all_docs = list_documents()
    return await run_agentic_query(
        query    = req.query,
        doc_ids  = req.doc_ids or None,
        alpha    = req.alpha,
        history  = req.history,
        all_docs = all_docs,
    )

@app.post("/ask/stream")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '20')}/minute")
async def ask_stream(request: Request, req: AskRequest):
    doc_ids  = req.doc_ids or None
    all_docs = list_documents()

    async def event_generator():
        try:
            async for token in generate_answer_stream(
                query    = req.query,
                doc_ids  = doc_ids,
                alpha    = req.alpha,
                history  = req.history,
                all_docs = all_docs,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"/ask/stream failed: {e}", exc_info=True)
            error_msg = "## Answer\nAn error occurred while processing your question.\n\n## References\nNone."
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )

# ── Compare ────────────────────────────────────────────────────────────────────

@app.post("/compare/")
@limiter.limit("10/minute")
async def compare_documents(request: Request, req: CompareRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    result_a, result_b = await asyncio.gather(
        generate_answer(req.query, [req.doc_id_a], None),
        generate_answer(req.query, [req.doc_id_b], None),
    )
    docs = {d["doc_id"]: d["filename"] for d in list_documents()}
    return {
        "query": req.query,
        "document_a": {"doc_id": req.doc_id_a, "filename": docs.get(req.doc_id_a, "unknown"),
                       "answer": result_a["answer"], "confidence": result_a.get("confidence"),
                       "sources": result_a.get("sources", [])},
        "document_b": {"doc_id": req.doc_id_b, "filename": docs.get(req.doc_id_b, "unknown"),
                       "answer": result_b["answer"], "confidence": result_b.get("confidence"),
                       "sources": result_b.get("sources", [])},
    }

# ── Images ─────────────────────────────────────────────────────────────────────

@app.get("/documents/{doc_id}/images")
async def list_document_images(doc_id: str):
    images = get_images_for_doc(doc_id)
    return {"doc_id": doc_id, "total": len(images), "images": images}

@app.get("/documents/{doc_id}/images/page/{page}")
async def list_page_images(doc_id: str, page: int):
    images = get_images_for_page(doc_id, page)
    return {"doc_id": doc_id, "page": page, "images": images}

@app.get("/images/{image_id}")
async def get_image_file(image_id: str):
    img = get_image_by_id(image_id)
    if not img or not os.path.exists(img["path"]):
        raise HTTPException(404, "Image not found")
    return FileResponse(img["path"])

# ── Feedback ───────────────────────────────────────────────────────────────────

@app.post("/feedback/")
@limiter.limit("60/minute")
async def submit_feedback(request: Request, req: FeedbackRequest):
    record_feedback(query=req.query, answer=req.answer, rating=req.rating, sources=req.sources)
    return {"status": "recorded", "rating": req.rating}

@app.get("/feedback/")
async def get_feedback():
    entries = load_feedback()
    return {
        "total":       len(entries),
        "thumbs_up":   sum(1 for e in entries if e.get("rating") == "up"),
        "thumbs_down": sum(1 for e in entries if e.get("rating") == "down"),
        "entries":     entries[-50:],
    }

# ── Eval log ───────────────────────────────────────────────────────────────────

@app.get("/eval-log/")
async def get_eval_log(limit: int = 50):
    entries      = load_eval_log()
    faith_scores = [e["faithfulness"]     for e in entries if e.get("faithfulness")     is not None]
    relev_scores = [e["answer_relevancy"] for e in entries if e.get("answer_relevancy") is not None]
    return {
        "total_evals": len(entries),
        "averages": {
            "faithfulness":     round(sum(faith_scores)/len(faith_scores), 2) if faith_scores else None,
            "answer_relevancy": round(sum(relev_scores)/len(relev_scores), 2) if relev_scores else None,
        },
        "recent": entries[-limit:],
    }

# ── Developer Mode (NEW) ─────────────────────────────────────────────────────

@app.get("/dev/logs/")
async def get_dev_logs(limit: int = 50):
    """Last N captured log lines for the developer mode panel."""
    logs = list(_dev_log_buffer)[-limit:]
    return {"total": len(_dev_log_buffer), "logs": logs}

@app.get("/dev/stats/")
async def get_dev_stats():
    """Aggregate stats: doc health, feedback, eval quality, log buffer size."""
    docs     = list_documents()
    feedback = load_feedback()
    evals    = load_eval_log()

    faith_scores = [e["faithfulness"]     for e in evals if e.get("faithfulness")     is not None]
    relev_scores = [e["answer_relevancy"] for e in evals if e.get("answer_relevancy") is not None]

    health_counts = {}
    for d in docs:
        h = d.get("health", "unknown")
        health_counts[h] = health_counts.get(h, 0) + 1

    return {
        "documents": {"total": len(docs), "health_breakdown": health_counts},
        "feedback": {
            "total":       len(feedback),
            "thumbs_up":   sum(1 for f in feedback if f.get("rating") == "up"),
            "thumbs_down": sum(1 for f in feedback if f.get("rating") == "down"),
        },
        "eval_quality": {
            "total_evals":          len(evals),
            "avg_faithfulness":     round(sum(faith_scores)/len(faith_scores), 2) if faith_scores else None,
            "avg_answer_relevancy": round(sum(relev_scores)/len(relev_scores), 2) if relev_scores else None,
        },
        "log_buffer_size": len(_dev_log_buffer),
    }